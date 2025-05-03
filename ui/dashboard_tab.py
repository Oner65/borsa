"""
Gerçek zamanlı piyasa verilerini gösteren dashboard sekmesi.
Canlı fiyat hareketleri, haber akışı, takvim olayları ve piyasa genel görünümünü içerir.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import logging
import traceback
import threading

# Canlı veri modüllerini import et
try:
    from data.live_market_data import get_market_snapshot, get_live_stock_data, get_tradingview_analysis
    # Interval sınıfını da try bloğu içinde import etmeye çalışalım
    try:
        from tradingview_ta import Interval
        TRADINGVIEW_INTERVAL_IMPORTED = True
    except ImportError:
        TRADINGVIEW_INTERVAL_IMPORTED = False
        # Fallback olarak Interval sınıfını kullanmak yerine string değerleri kullanalım
    
    LIVE_DATA_AVAILABLE = True
except ImportError:
    LIVE_DATA_AVAILABLE = False
    logging.warning("live_market_data modülü yüklenemedi. Canlı veri özellikleri devre dışı.")

# WebSocket modüllerini import et
try:
    from data.websocket_data import initialize_websocket, register_callback, get_latest_price, get_realtime_data
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logging.warning("websocket_data modülü yüklenemedi. Gerçek zamanlı veri özellikleri devre dışı.")

# Ekonomik takvim modüllerini import et
try:
    from data.economic_calendar import get_economic_calendar, get_central_bank_announcements, get_key_economic_indicators
    CALENDAR_AVAILABLE = True
except ImportError:
    CALENDAR_AVAILABLE = False
    logging.warning("economic_calendar modülü yüklenemedi. Ekonomik takvim özellikleri devre dışı.")

# Kullanıcı yönetim modüllerini import et
try:
    from utils.user_management import get_user_preferences, update_user_preferences
    USER_MANAGEMENT_AVAILABLE = True
except ImportError:
    USER_MANAGEMENT_AVAILABLE = False
    logging.warning("user_management modülü yüklenemedi. Kullanıcı tercihleri özelliği devre dışı.")

# Loglama yapılandırması
logger = logging.getLogger(__name__)

# WebSocket istemcisini başlat
websocket_client = None
if WEBSOCKET_AVAILABLE:
    try:
        websocket_client = initialize_websocket()
    except Exception as e:
        logger.error(f"WebSocket istemcisi başlatılırken hata: {str(e)}")
        logger.error(traceback.format_exc())
        websocket_client = None
        WEBSOCKET_AVAILABLE = False

def render_dashboard_tab(username=None):
    """
    Ana dashboard sekmesini oluşturur
    
    Args:
        username (str, optional): Giriş yapmış kullanıcı adı
    """
    st.header("Borsa İstanbul Dashboard")
    
    # Kullanıcı tercihlerini yükle
    user_prefs = None
    if USER_MANAGEMENT_AVAILABLE and username is not None:
        try:
            user_prefs = get_user_preferences(username)
        except Exception as e:
            logger.error(f"Kullanıcı tercihleri yüklenirken hata: {str(e)}")
    
    # SOL BÖLÜM
    
    # 1. Piyasa Genel Görünümü
    st.subheader("Piyasa Genel Görünümü")
    if LIVE_DATA_AVAILABLE:
        render_market_snapshot()
    else:
        st.warning("Canlı piyasa verileri için gerekli modüller yüklenemedi.")
    
    # 2. Gerçek Zamanlı Hisse Fiyatları
    st.subheader("Gerçek Zamanlı Hisse Fiyatları")
    if WEBSOCKET_AVAILABLE:
        render_realtime_chart()
    else:
        st.warning("Gerçek zamanlı veriler için gerekli modüller yüklenemedi.")
    
    # SAĞ BÖLÜM
    
    # 3. Ekonomik Takvim
    st.subheader("Ekonomik Takvim")
    if CALENDAR_AVAILABLE:
        render_economic_events()
    else:
        st.warning("Ekonomik takvim için gerekli modüller yüklenemedi.")
    
    # 4. Piyasa Haberleri
    st.subheader("Piyasa Haberleri")
    render_market_news()
    
    # 5. Teknik Analiz Görünümü
    st.subheader("Teknik Analiz Görünümü")
    if LIVE_DATA_AVAILABLE:
        render_technical_dashboard()
    else:
        st.warning("Teknik analiz için gerekli modüller yüklenemedi.")

def render_market_snapshot():
    """Piyasa özet bilgilerini gösterir"""
    if LIVE_DATA_AVAILABLE:
        try:
            # Piyasa özeti verilerini al
            snapshot = get_market_snapshot()
            
            # Hata varsa göster
            if "error" in snapshot:
                st.error(f"Piyasa verileri alınırken hata: {snapshot['error']}")
                return
            
            # BIST100 metriği
            last_value = snapshot["BIST100"]["last"]
            change_pct = snapshot["BIST100"]["change_pct"]
            color = "green" if change_pct > 0 else "red" if change_pct < 0 else "gray"
            
            st.metric(
                label="BIST100", 
                value=f"{last_value:,.2f}", 
                delta=f"{change_pct:+.2f}%",
                delta_color="normal"
            )
            
            # USD/TRY metriği
            last_value = snapshot["USD/TRY"]["last"]
            change_pct = snapshot["USD/TRY"]["change_pct"]
            
            st.metric(
                label="USD/TRY", 
                value=f"{last_value:.4f}", 
                delta=f"{change_pct:+.2f}%",
                delta_color="inverse"  # Yükseliş kötü, düşüş iyi
            )
            
            # Altın metriği
            last_value = snapshot["Altın"]["last"]
            change_pct = snapshot["Altın"]["change_pct"]
            
            st.metric(
                label="Altın (Ons/$)", 
                value=f"{last_value:,.2f}", 
                delta=f"{change_pct:+.2f}%",
                delta_color="normal"
            )
            
            # Petrol metriği  
            last_value = snapshot["Petrol"]["last"]
            change_pct = snapshot["Petrol"]["change_pct"]
            
            st.metric(
                label="Brent Petrol", 
                value=f"{last_value:.2f}", 
                delta=f"{change_pct:+.2f}%",
                delta_color="normal"
            )
            
            # Piyasa saati ve son güncelleme zamanı
            st.caption(f"Son güncelleme: {snapshot.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}")
        
        except Exception as e:
            st.error(f"Piyasa verileri işlenirken hata: {str(e)}")
            logger.error(traceback.format_exc())
    else:
        st.warning("Canlı piyasa verileri için gerekli modüller yüklenemedi.")

def render_realtime_chart():
    """Gerçek zamanlı fiyat grafiğini gösterir"""
    # Sembol seçimi
    default_symbols = ["GARAN.IS", "THYAO.IS", "AKBNK.IS", "TUPRS.IS", "EREGL.IS"]
    selected_symbol = st.selectbox("Hisse Senedi", default_symbols)
    
    # İzleme butonu
    watch_button = st.button("İzle", key="realtime_watch")
    if watch_button and websocket_client and selected_symbol:
        websocket_client.subscribe(selected_symbol)
        st.success(f"{selected_symbol} gerçek zamanlı izleniyor")
    
    # Grafik alanı için placeholder
    chart_placeholder = st.empty()
    
    if websocket_client and selected_symbol:
        # Son fiyatı al
        latest_price = get_latest_price(selected_symbol)
        
        # Gerçek zamanlı verileri al
        realtime_data = get_realtime_data(selected_symbol, limit=30)
        
        if realtime_data is not None and not realtime_data.empty:
            # Grafik oluştur
            fig = go.Figure()
            
            # Çizgi grafik ekle
            fig.add_trace(go.Scatter(
                x=realtime_data["datetime"],
                y=realtime_data["price"],
                mode='lines+markers',
                name=selected_symbol,
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Layout düzenle
            fig.update_layout(
                title=f"{selected_symbol} Gerçek Zamanlı Fiyat",
                xaxis_title="Saat",
                yaxis_title="Fiyat (TL)",
                height=400,
                margin=dict(l=0, r=0, t=40, b=0),
                hovermode="x unified",
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(230, 230, 230, 0.5)',
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(230, 230, 230, 0.5)',
                    tickformat='.2f'
                ),
                plot_bgcolor='rgba(255, 255, 255, 0.9)',
            )
            
            # Grafiği göster
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Son fiyatı göster
            st.metric(
                label=f"{selected_symbol} Son Fiyat", 
                value=f"{latest_price:.2f} TL",
                delta=f"{realtime_data['price'].iloc[-1] - realtime_data['price'].iloc[0]:.2f} TL",
                delta_color="normal"
            )
        else:
            chart_placeholder.info(f"{selected_symbol} için gerçek zamanlı veri henüz yok. Lütfen bekleyin veya farklı bir hisse seçin.")
    else:
        chart_placeholder.warning("Gerçek zamanlı veri alınamıyor. Seçilen hisse için veri olmayabilir veya bağlantı sorunu yaşanıyor.")

def render_economic_events():
    """Ekonomik takvim olaylarını gösterir"""
    if CALENDAR_AVAILABLE:
        try:
            # Bugünü ve tarih aralığını belirle
            today = datetime.now()
            start_date = today.strftime('%Y/%m/%d')
            end_date = (today + timedelta(days=3)).strftime('%Y/%m/%d')
            
            # Ekonomik takvimi al
            calendar = get_economic_calendar(start_date, end_date, country="turkey", importance="high")
            
            if calendar is not None and not calendar.empty:
                # Tarih gruplarına göre olayları göster
                for date, group in calendar.groupby('event_date'):
                    st.markdown(f"**{date.strftime('%d %B %Y')}**")
                    
                    for _, event in group.iterrows():
                        # Önem renkleri
                        importance_color = "red" if event['importance'] == "high" else "orange" if event['importance'] == "medium" else "gray"
                        
                        # Olay bilgilerini göster
                        st.markdown(f"""
                        <div style="padding: 5px; margin-bottom: 5px; border-left: 3px solid {importance_color}; background-color: rgba(240, 240, 240, 0.5);">
                            <div style="font-size: 0.9em; color: #555;">{event['event_time']} - {event['currency']}</div>
                            <div style="font-weight: bold;">{event['event_name']}</div>
                            <div style="font-size: 0.9em;">
                                Tahmin: <b>{event['forecast_value']}</b> | 
                                Önceki: <b>{event['previous_value']}</b>
                                {f" | <span style='color: green;'>Gerçekleşen: <b>{event['actual_value']}</b></span>" if pd.notna(event['actual_value']) else ""}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.caption("Ekonomik takvim verileri")
            else:
                st.info("Bugün ve önümüzdeki günlerde önemli bir ekonomik olay bulunmuyor.")
        
        except Exception as e:
            st.error(f"Ekonomik takvim verileri alınırken hata: {str(e)}")
            logger.error(traceback.format_exc())
    else:
        st.warning("Ekonomik takvim için gerekli modüller yüklenemedi.")

def render_market_news():
    """Son piyasa haberlerini gösterir"""
    try:
        # Haberler için örnek veriler
        news = [
            {
                "title": "BIST100 yükselişle kapattı",
                "source": "Ekonomi Haberleri",
                "summary": "Borsa İstanbul, günü %1.2 yükselişle tamamladı. Bankacılık ve enerji hisseleri öne çıktı.",
                "url": "https://example.com/news1",
                "published_at": "2024-05-30 16:45"
            },
            {
                "title": "TCMB Başkanı'ndan faiz açıklaması",
                "source": "Merkez Bankası",
                "summary": "TCMB Başkanı, enflasyonla mücadele kararlılığının süreceğini ve sıkı para politikasının devam edeceğini belirtti.",
                "url": "https://example.com/news2",
                "published_at": "2024-05-30 14:30"
            },
            {
                "title": "Otomotiv şirketlerinin ihracat rakamları açıklandı",
                "source": "Otomotiv Sanayi Derneği",
                "summary": "Otomotiv sektörü Mayıs ayında geçen yılın aynı dönemine göre %8.5 artışla 2.8 milyar dolar ihracat gerçekleştirdi.",
                "url": "https://example.com/news3",
                "published_at": "2024-05-30 12:15"
            },
            {
                "title": "Dolar/TL yeni rekor kırdı",
                "source": "Finans Haberleri",
                "summary": "Dolar/TL kuru, küresel piyasalardaki gelişmeler ve yurtiçi faktörlerin etkisiyle yeni bir rekor seviyeye ulaştı.",
                "url": "https://example.com/news4",
                "published_at": "2024-05-30 11:20"
            }
        ]
        
        # Haberleri göster
        for item in news:
            st.markdown(f"""
            <div style="padding: 10px; margin-bottom: 10px; border: 1px solid #eee; border-radius: 5px;">
                <div style="font-weight: bold;">{item['title']}</div>
                <div style="font-size: 0.8em; color: #777; margin-bottom: 5px;">{item['source']} · {item['published_at']}</div>
                <div style="font-size: 0.9em; margin-bottom: 5px;">{item['summary']}</div>
                <div style="font-size: 0.8em;"><a href="{item['url']}" target="_blank">Devamını oku</a></div>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Haberler yüklenirken hata: {str(e)}")
        logger.error(traceback.format_exc())

def render_technical_dashboard():
    """Teknik analiz panosu gösterir"""
    # Sembol seçimi
    default_symbols = ["GARAN.IS", "THYAO.IS", "AKBNK.IS", "TUPRS.IS", "EREGL.IS"]
    selected_symbol = st.selectbox("Hisse Senedi", default_symbols, key="technical_symbol")
    
    # Teknik analiz
    if selected_symbol:
        try:
            # TradingView teknik analizi
            analysis = get_tradingview_analysis(selected_symbol)
            
            if "error" not in analysis:
                # Özet bilgileri göster
                summary = analysis["summary"]
                oscillators = analysis["oscillators"]
                moving_averages = analysis["moving_averages"]
                
                # Teknik analiz sonuçlarını göster
                st.markdown(f"""
                <div style="padding: 15px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 15px;">
                    <h4 style="margin-top: 0;">{selected_symbol} Teknik Analiz</h4>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <div style="text-align: center; flex: 1;">
                            <div style="font-size: 0.9em; color: #555;">Özet</div>
                            <div style="font-weight: bold; font-size: 1.1em; color: {'green' if 'BUY' in summary['recommendation'] else 'red' if 'SELL' in summary['recommendation'] else 'gray'};">
                                {summary['recommendation']}
                            </div>
                            <div style="font-size: 0.8em;">
                                Al: {summary['buy_signals']} | Sat: {summary['sell_signals']} | Nötr: {summary['neutral_signals']}
                            </div>
                        </div>
                        <div style="text-align: center; flex: 1;">
                            <div style="font-size: 0.9em; color: #555;">Osilatörler</div>
                            <div style="font-weight: bold; font-size: 1.1em; color: {'green' if 'BUY' in oscillators['recommendation'] else 'red' if 'SELL' in oscillators['recommendation'] else 'gray'};">
                                {oscillators['recommendation']}
                            </div>
                            <div style="font-size: 0.8em;">
                                Al: {oscillators['buy_signals']} | Sat: {oscillators['sell_signals']} | Nötr: {oscillators['neutral_signals']}
                            </div>
                        </div>
                        <div style="text-align: center; flex: 1;">
                            <div style="font-size: 0.9em; color: #555;">Hareketli Ort.</div>
                            <div style="font-weight: bold; font-size: 1.1em; color: {'green' if 'BUY' in moving_averages['recommendation'] else 'red' if 'SELL' in moving_averages['recommendation'] else 'gray'};">
                                {moving_averages['recommendation']}
                            </div>
                            <div style="font-size: 0.8em;">
                                Al: {moving_averages['buy_signals']} | Sat: {moving_averages['sell_signals']} | Nötr: {moving_averages['neutral_signals']}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # İndikatör değerlerini göster
                indicators = analysis["indicators"]
                
                st.markdown("**Teknik Göstergeler**")
                
                # Önemli indikatörleri belirle
                key_indicators = {
                    "RSI": "RSI",
                    "RSI[1]": "RSI(1)",
                    "MACD.macd": "MACD",
                    "MACD.signal": "MACD Sinyal",
                    "Stoch.K": "Stokastik K",
                    "Stoch.D": "Stokastik D",
                    "CCI20": "CCI(20)",
                    "ADX": "ADX",
                    "ATR": "ATR",
                    "W.R": "Williams %R",
                    "Volatility": "Volatilite",
                    "Mom": "Momentum"
                }
                
                # İndikatörleri göster
                for indicator_key, indicator_name in key_indicators.items():
                    if indicator_key in indicators:
                        value = indicators[indicator_key]
                        
                        # RSI için renk kodlaması
                        if indicator_key == "RSI":
                            color = "red" if value > 70 else "green" if value < 30 else "black"
                        elif indicator_key == "Stoch.K" or indicator_key == "Stoch.D":
                            color = "red" if value > 80 else "green" if value < 20 else "black"
                        else:
                            color = "black"
                        
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <div>{indicator_name}:</div>
                            <div style="font-weight: bold; color: {color};">{value:.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning(f"Teknik analiz verileri alınamadı: {analysis.get('error', 'Bilinmeyen hata')}")
                
        except Exception as e:
            st.error(f"Teknik analiz verileri işlenirken hata: {str(e)}")
            logger.error(traceback.format_exc())
    
        try:
            # Canlı hisse grafik
            # Veriyi al
            period = "1d"
            interval = "5m"
            stock_data = get_live_stock_data(selected_symbol, interval=interval)
            
            if stock_data is not None and not stock_data.empty:
                # Grafik oluştur
                fig = go.Figure()
                
                # Mum grafik ekle
                fig.add_trace(go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close'],
                    name=selected_symbol
                ))
                
                # Layout düzenle
                fig.update_layout(
                    title=f"{selected_symbol} - {period} / {interval}",
                    xaxis_title="Tarih",
                    yaxis_title="Fiyat (TL)",
                    height=400,
                    margin=dict(l=0, r=0, t=40, b=0),
                    xaxis=dict(
                        rangeslider=dict(visible=False),
                        showgrid=True,
                        gridcolor='rgba(230, 230, 230, 0.5)',
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(230, 230, 230, 0.5)',
                        tickformat='.2f'
                    ),
                    plot_bgcolor='rgba(255, 255, 255, 0.9)',
                )
                
                # Grafiği göster
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"{selected_symbol} için günlük veri bulunamadı.")
        
        except Exception as e:
            st.error(f"Hisse verileri yüklenirken hata: {str(e)}")
            logger.error(traceback.format_exc()) 