import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import time
import logging
import traceback
from datetime import datetime, timedelta
import sqlite3
import random
import warnings
import json

# Streamlit Cloud için sayfa konfigürasyonu
st.set_page_config(
    page_title="Kompakt Borsa Analiz Uygulaması",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Uyarıları bastır
warnings.filterwarnings("ignore")

# Loglama yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cloud ortamı için path ayarları
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Ana modüllerden fonksiyonları import etmeye çalış
try:
    from ui.stock_tab import render_stock_tab
    from ui.bist100_tab import render_bist100_tab
    from ui.ml_prediction_tab import render_ml_prediction_tab
    from data.db_utils import DB_FILE, get_analysis_results, save_analysis_result
    from data.stock_data import get_market_summary, get_popular_stocks
    from data.announcements import get_announcements, get_all_announcements  
    from data.utils import get_analysis_result, save_analysis_result, get_favorites
    MODULES_AVAILABLE = True
    logger.info("Ana modüller başarıyla yüklendi")
except ImportError as e:
    MODULES_AVAILABLE = False
    logger.warning(f"Ana modüller yüklenemedi: {e}")

# Eğer ana modüller yüklenemezse, kendi render fonksiyonlarımızı tanımlayacağız
if not MODULES_AVAILABLE:
    # Ana uygulamadan kopyalanan render fonksiyonları burada tanımlanacak
    def render_stock_tab():
        """Ana uygulamadan kopyalanan hisse analizi sekmesi"""
        st.header("Hisse Senedi Teknik Analizi")
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        # Session state'den selected_stock_for_analysis'i kontrol et
        initial_stock = ""
        if 'selected_stock_for_analysis' in st.session_state and st.session_state.selected_stock_for_analysis:
            initial_stock = st.session_state.selected_stock_for_analysis
            st.session_state.selected_stock_for_analysis = ""
        
        with col1:
            stock_symbol = st.text_input("Hisse Senedi Kodu", value=initial_stock)
        
        with col2:
            period = st.selectbox("Zaman Aralığı", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
        
        with col3:
            interval = st.selectbox("Veri Aralığı", ["1d", "1wk"], index=0)
        
        with col4:
            st.write("")
            analyze = st.button("Analiz Et", use_container_width=True)
        
        # Analiz yap
        if analyze and stock_symbol:
            with st.spinner(f"{stock_symbol} analiz ediliyor..."):
                try:
                    stock_symbol = stock_symbol.upper().strip()
                    df = get_stock_data(stock_symbol, period, interval)
                    
                    if not df.empty:
                        df = calculate_indicators(df)
                        signals = get_signals(df)
                        company_info = get_company_info(stock_symbol)
                        
                        # Temel bilgiler
                        last_price = df['Close'].iloc[-1]
                        prev_price = df['Close'].iloc[-2] if len(df) > 1 else last_price
                        change = ((last_price - prev_price) / prev_price) * 100
                        
                        col_info, col_fav = st.columns([4, 1])
                        
                        with col_info:
                            col_metrics = st.columns(4)
                            
                            with col_metrics[0]:
                                st.metric("Son Fiyat", f"{last_price:.2f} TL", f"{change:.2f}%")
                            
                            with col_metrics[1]:
                                rsi = df['RSI'].iloc[-1]
                                st.metric("RSI", f"{rsi:.1f}")
                            
                            with col_metrics[2]:
                                macd = df['MACD'].iloc[-1]
                                st.metric("MACD", f"{macd:.4f}")
                            
                            with col_metrics[3]:
                                volume = df['Volume'].iloc[-1]
                                st.metric("Hacim", f"{volume:,.0f}")
                        
                        with col_fav:
                            st.write("")
                            if st.button("⭐ Favorilere Ekle", use_container_width=True):
                                if add_to_favorites(stock_symbol):
                                    st.success("Favorilere eklendi!")
                                    st.rerun()
                                else:
                                    st.warning("Zaten favorilerde")
                        
                        # Şirket bilgileri
                        if company_info.get("name") != stock_symbol:
                            st.info(f"**{company_info['name']}** - {company_info.get('sector', 'N/A')} sektörü")
                        
                        # Grafik
                        fig = create_stock_chart(df, stock_symbol)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Teknik göstergeler tablosu
                        st.subheader("Teknik Göstergeler")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write("**Hareketli Ortalamalar**")
                            ma_data = {
                                "Gösterge": ["SMA5", "SMA20", "SMA50", "SMA200"],
                                "Değer": [
                                    f"{df['SMA5'].iloc[-1]:.2f}",
                                    f"{df['SMA20'].iloc[-1]:.2f}",
                                    f"{df['SMA50'].iloc[-1]:.2f}",
                                    f"{df['SMA200'].iloc[-1]:.2f}"
                                ],
                                "Sinyal": [
                                    "AL" if signals['SMA5_Signal'].iloc[-1] > 0 else "SAT",
                                    "AL" if signals['SMA20_Signal'].iloc[-1] > 0 else "SAT",
                                    "AL" if signals['SMA50_Signal'].iloc[-1] > 0 else "SAT",
                                    "AL" if signals['SMA200_Signal'].iloc[-1] > 0 else "SAT"
                                ]
                            }
                            
                            def color_signal(val):
                                if val == "AL":
                                    return 'background-color: green; color: white'
                                elif val == "SAT":
                                    return 'background-color: red; color: white'
                                return ''
                            
                            ma_df = pd.DataFrame(ma_data)
                            st.dataframe(ma_df.style.map(color_signal, subset=['Sinyal']), hide_index=True, use_container_width=True)
                        
                        with col2:
                            st.write("**Osilatörler**")
                            osc_data = {
                                "Gösterge": ["RSI", "MACD", "Stochastic", "Williams %R"],
                                "Değer": [
                                    f"{df['RSI'].iloc[-1]:.1f}",
                                    f"{df['MACD'].iloc[-1]:.4f}",
                                    f"{df['Stoch_%K'].iloc[-1]:.1f}",
                                    f"{df['Williams_%R'].iloc[-1]:.1f}"
                                ],
                                "Sinyal": [
                                    "AL" if signals['RSI_Signal'].iloc[-1] > 0 else ("SAT" if signals['RSI_Signal'].iloc[-1] < 0 else "NÖTR"),
                                    "AL" if signals['MACD_Signal'].iloc[-1] > 0 else "SAT",
                                    "AL" if signals['Stoch_Signal'].iloc[-1] > 0 else ("SAT" if signals['Stoch_Signal'].iloc[-1] < 0 else "NÖTR"),
                                    "AL" if signals['Williams_%R_Signal'].iloc[-1] > 0 else ("SAT" if signals['Williams_%R_Signal'].iloc[-1] < 0 else "NÖTR")
                                ]
                            }
                            
                            def color_signal_osc(val):
                                if val == "AL":
                                    return 'background-color: green; color: white'
                                elif val == "SAT":
                                    return 'background-color: red; color: white'
                                else:
                                    return 'background-color: gray; color: white'
                            
                            osc_df = pd.DataFrame(osc_data)
                            st.dataframe(osc_df.style.map(color_signal_osc, subset=['Sinyal']), hide_index=True, use_container_width=True)
                        
                        with col3:
                            st.write("**Analiz Özeti**")
                            
                            # Sinyalleri say
                            ma_signals = ['SMA5_Signal', 'SMA20_Signal', 'SMA50_Signal', 'SMA200_Signal']
                            osc_signals = ['RSI_Signal', 'MACD_Signal', 'Stoch_Signal', 'Williams_%R_Signal']
                            
                            ma_buy = sum(1 for sig in ma_signals if signals[sig].iloc[-1] > 0)
                            ma_sell = sum(1 for sig in ma_signals if signals[sig].iloc[-1] < 0)
                            
                            osc_buy = sum(1 for sig in osc_signals if signals[sig].iloc[-1] > 0)
                            osc_sell = sum(1 for sig in osc_signals if signals[sig].iloc[-1] < 0)
                            
                            total_buy = ma_buy + osc_buy
                            total_sell = ma_sell + osc_sell
                            total_signals = len(ma_signals) + len(osc_signals)
                            
                            summary_data = {
                                "Kategori": ["Hareketli Ort.", "Osilatörler", "Genel"],
                                "Al": [f"{ma_buy}/4", f"{osc_buy}/4", f"{total_buy}/8"],
                                "Sat": [f"{ma_sell}/4", f"{osc_sell}/4", f"{total_sell}/8"],
                                "Sonuç": [
                                    "AL" if ma_buy > ma_sell else ("SAT" if ma_sell > ma_buy else "NÖTR"),
                                    "AL" if osc_buy > osc_sell else ("SAT" if osc_sell > osc_buy else "NÖTR"),
                                    "GÜÇLÜ AL" if total_buy > 6 else ("AL" if total_buy > total_sell else ("GÜÇLÜ SAT" if total_sell > 6 else ("SAT" if total_sell > total_buy else "NÖTR")))
                                ]
                            }
                            
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df.style.map(color_signal_osc, subset=['Sonuç']), hide_index=True, use_container_width=True)
                            
                            # Final recommendation
                            final_rec = summary_df['Sonuç'].iloc[2]
                            rec_color = "green" if "AL" in final_rec else ("red" if "SAT" in final_rec else "gray")
                            st.markdown(f"<h3 style='text-align: center; color: {rec_color};'>{final_rec}</h3>", unsafe_allow_html=True)
                        
                        # Performans analizi
                        st.subheader("Performans Analizi")
                        
                        perf_cols = st.columns(4)
                        periods = [5, 20, 60, 252]
                        period_names = ["1 Hafta", "1 Ay", "3 Ay", "1 Yıl"]
                        
                        for i, (period, name) in enumerate(zip(periods, period_names)):
                            with perf_cols[i]:
                                if len(df) > period:
                                    perf = ((df['Close'].iloc[-1] / df['Close'].iloc[-period-1]) - 1) * 100
                                    st.metric(name, f"{perf:.2f}%")
                                else:
                                    st.metric(name, "N/A")
                    
                    else:
                        st.error("Hisse senedi verisi bulunamadı")
                        
                except Exception as e:
                    st.error(f"Analiz hatası: {str(e)}")
                    logger.error(f"Analiz hatası: {str(e)}")
        
        else:
            st.info("Hisse senedi kodunu girin ve 'Analiz Et' butonuna tıklayın.")
    
    def render_bist100_tab():
        """Ana uygulamadan kopyalanan BIST100 sekmesi"""
        st.header("BIST 100 Genel Bakış", divider="blue")
        
        # BIST-100 verilerini al
        with st.spinner("BIST-100 verileri yükleniyor..."):
            bist100_data = get_stock_data("XU100", "6mo")
            
            if not bist100_data.empty:
                bist100_data = calculate_indicators(bist100_data)
                
                # Günlük veriler
                last_price = bist100_data['Close'].iloc[-1]
                prev_price = bist100_data['Close'].iloc[-2] if len(bist100_data) > 1 else last_price
                change = ((last_price - prev_price) / prev_price) * 100
                today_high = bist100_data['High'].iloc[-1]
                today_low = bist100_data['Low'].iloc[-1]
                volume = bist100_data['Volume'].iloc[-1]
                
                # Metrikler
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("BIST 100", f"{last_price:.0f}", f"{change:.2f}%")
                
                with col2:
                    st.metric("Günlük Aralık", f"{today_low:.0f} - {today_high:.0f}")
                
                with col3:
                    st.metric("Hacim", f"{volume:,.0f}")
                
                with col4:
                    # Haftalık değişim
                    if len(bist100_data) >= 6:
                        weekly_change = ((last_price - bist100_data['Close'].iloc[-6]) / bist100_data['Close'].iloc[-6]) * 100
                        st.metric("Haftalık", f"{weekly_change:.2f}%")
                    else:
                        st.metric("Haftalık", "N/A")
                
                # Grafik
                fig = create_stock_chart(bist100_data, "BIST-100")
                st.plotly_chart(fig, use_container_width=True)
                
                # Teknik analiz özeti
                st.subheader("Teknik Analiz Özeti")
                signals = get_signals(bist100_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Trend Analizi**")
                    trend_data = {
                        "Gösterge": ["SMA20 vs SMA50", "SMA50 vs SMA200", "MACD"],
                        "Durum": [
                            "Yükseliş" if bist100_data['SMA20'].iloc[-1] > bist100_data['SMA50'].iloc[-1] else "Düşüş",
                            "Yükseliş" if bist100_data['SMA50'].iloc[-1] > bist100_data['SMA200'].iloc[-1] else "Düşüş",
                            "Pozitif" if bist100_data['MACD'].iloc[-1] > bist100_data['MACD_Signal'].iloc[-1] else "Negatif"
                        ]
                    }
                    st.dataframe(pd.DataFrame(trend_data), hide_index=True)
                
                with col2:
                    st.write("**Momentum Göstergeleri**")
                    momentum_data = {
                        "Gösterge": ["RSI", "Stochastic %K", "Williams %R"],
                        "Değer": [
                            f"{bist100_data['RSI'].iloc[-1]:.1f}",
                            f"{bist100_data['Stoch_%K'].iloc[-1]:.1f}",
                            f"{bist100_data['Williams_%R'].iloc[-1]:.1f}"
                        ],
                        "Durum": [
                            "Aşırı Alım" if bist100_data['RSI'].iloc[-1] > 70 else ("Aşırı Satım" if bist100_data['RSI'].iloc[-1] < 30 else "Normal"),
                            "Aşırı Alım" if bist100_data['Stoch_%K'].iloc[-1] > 80 else ("Aşırı Satım" if bist100_data['Stoch_%K'].iloc[-1] < 20 else "Normal"),
                            "Aşırı Satım" if bist100_data['Williams_%R'].iloc[-1] < -80 else ("Aşırı Alım" if bist100_data['Williams_%R'].iloc[-1] > -20 else "Normal")
                        ]
                    }
                    st.dataframe(pd.DataFrame(momentum_data), hide_index=True)
                
            else:
                st.error("BIST-100 verileri alınamadı")
        
        # Popüler hisseler
        st.subheader("Popüler Hisseler")
        
        with st.spinner("Popüler hisseler yükleniyor..."):
            popular_stocks = get_popular_stocks()
            
            # 5'er sütun halinde göster
            for i in range(0, len(popular_stocks), 5):
                cols = st.columns(5)
                for j, col in enumerate(cols):
                    if i + j < len(popular_stocks):
                        stock = popular_stocks[i + j]
                        with col:
                            change_color = "green" if stock["change_percent"] > 0 else ("red" if stock["change_percent"] < 0 else "gray")
                            arrow = "↑" if stock["change_percent"] > 0 else ("↓" if stock["change_percent"] < 0 else "→")
                            
                            st.markdown(f"""
                            <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 10px; text-align: center;">
                                <h5 style="margin: 0; color: {change_color};">{stock["symbol"]} {arrow}</h5>
                                <p style="margin: 0; font-size: 12px; color: gray;">{stock["name"][:15]}...</p>
                                <p style="margin: 0; color: {change_color}; font-weight: bold;">{stock["change_percent"]:.2f}%</p>
                                <p style="margin: 0; font-size: 12px;">{stock["value"]:.2f} TL</p>
                            </div>
                            """, unsafe_allow_html=True)
    
    def render_ml_prediction_tab():
        """Ana uygulamadan kopyalanan ML tahmin sekmesi"""
        st.header("ML Yükseliş Tahmini", divider="rainbow")
        
        st.markdown("""
        Bu bölüm, makine öğrenmesi algoritmaları kullanarak hisse senetlerinin gelecekteki performansını tahmin eder.
        
        **Nasıl Çalışır:**
        - Teknik göstergeler, fiyat hareketleri ve hacim verileri analiz edilir
        - RandomForest algoritması ile model eğitilir
        - Belirlediğiniz eşik değerini aşma olasılığı hesaplanır
        
        **Not:** Bu tahminler yatırım tavsiyesi değildir, sadece analiz amaçlıdır.
        """)
        
        # Parametreler
        col1, col2, col3 = st.columns(3)
        
        with col1:
            threshold = st.slider("Yükseliş Eşiği (%)", 1.0, 10.0, 3.0, 0.5) / 100
        
        with col2:
            scan_option = st.selectbox("Tarama Türü", ["BIST 30", "BIST 50", "Popüler Hisseler", "Özel Liste"])
        
        with col3:
            prediction_days = st.selectbox("Tahmin Süresi", [1, 5, 10, 20], index=1)
        
        # Özel liste için
        custom_stocks = ""
        if scan_option == "Özel Liste":
            custom_stocks = st.text_area("Hisse Kodları (virgülle ayırın)", placeholder="THYAO, GARAN, ASELS")
        
        # Hisse listesini belirle
        if scan_option == "BIST 30":
            stock_list = ["THYAO", "GARAN", "AKBNK", "EREGL", "TUPRS", "FROTO", "ASELS", "SASA", "KOZAA", "VAKBN",
                         "HALKB", "ISCTR", "YKBNK", "MGROS", "BIMAS", "TOASO", "KCHOL", "SAHOL", "DOHOL", "TCELL"]
        elif scan_option == "BIST 50":
            stock_list = ["THYAO", "GARAN", "AKBNK", "EREGL", "TUPRS", "FROTO", "ASELS", "SASA", "KOZAA", "VAKBN",
                         "HALKB", "ISCTR", "YKBNK", "MGROS", "BIMAS", "TOASO", "KCHOL", "SAHOL", "DOHOL", "TCELL",
                         "TTKOM", "PGSUS", "SOKM", "MAVI", "TAVHL", "KARSN", "OTKAR", "ARCLK", "VESTL", "PETKM"]
        elif scan_option == "Popüler Hisseler":
            stock_list = ["THYAO", "GARAN", "ASELS", "AKBNK", "EREGL", "VAKBN", "TUPRS", "FROTO", "SASA", "KOZAA"]
        else:  # Özel Liste
            if custom_stocks:
                stock_list = [s.strip().upper() for s in custom_stocks.split(",") if s.strip()]
            else:
                stock_list = []
        
        if st.button("ML Tarama Başlat", type="primary", use_container_width=True):
            if not stock_list:
                st.error("Analiz edilecek hisse bulunamadı")
                return
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            
            try:
                # ML kütüphanelerini kontrol et
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import accuracy_score
                
                for i, symbol in enumerate(stock_list):
                    status_text.text(f"Analiz ediliyor: {symbol} ({i+1}/{len(stock_list)})")
                    progress_bar.progress((i + 1) / len(stock_list))
                    
                    try:
                        # Veri al
                        df = get_stock_data(symbol, "2y")
                        
                        if df.empty or len(df) < 100:
                            continue
                        
                        # Teknik göstergeleri hesapla
                        df = calculate_indicators(df)
                        
                        # Özellikleri hazırla
                        features = prepare_features(df)
                        target = create_ml_target(df, threshold, prediction_days)
                        
                        # Model eğit
                        model, status = train_simple_model(features, target)
                        
                        if model is not None:
                            # Son veri için tahmin yap
                            last_features = features.iloc[-1:].select_dtypes(include=[np.number]).fillna(0)
                            last_features = last_features.replace([np.inf, -np.inf], 0)
                            
                            if len(last_features.columns) > 0:
                                prediction_proba = model.predict_proba(last_features)[0]
                                confidence = prediction_proba[1] if len(prediction_proba) > 1 else 0.5
                                
                                # Son fiyat bilgileri
                                last_price = df['Close'].iloc[-1]
                                daily_change = ((last_price / df['Close'].iloc[-2]) - 1) * 100 if len(df) > 1 else 0
                                
                                results.append({
                                    "Hisse": symbol,
                                    "Fiyat": last_price,
                                    "Günlük %": daily_change,
                                    "Yükseliş Olasılığı": confidence * 100,
                                    "Model Durumu": status,
                                    "Tavsiye": "AL" if confidence > 0.6 else ("DİKKAT" if confidence > 0.4 else "SAT")
                                })
                    
                    except Exception as e:
                        logger.error(f"{symbol} analiz hatası: {str(e)}")
                        continue
                
                progress_bar.empty()
                status_text.empty()
                
                if results:
                    # Sonuçları sırala
                    results_df = pd.DataFrame(results)
                    results_df = results_df.sort_values("Yükseliş Olasılığı", ascending=False)
                    
                    st.subheader("ML Tahmin Sonuçları")
                    
                    # Renklendirme fonksiyonu
                    def color_prediction(val):
                        if isinstance(val, str):
                            if val == "AL":
                                return 'background-color: green; color: white'
                            elif val == "SAT":
                                return 'background-color: red; color: white'
                            elif val == "DİKKAT":
                                return 'background-color: orange; color: white'
                        return ''
                    
                    def color_probability(val):
                        if isinstance(val, (int, float)):
                            if val > 70:
                                return 'background-color: darkgreen; color: white'
                            elif val > 60:
                                return 'background-color: green; color: white'
                            elif val > 40:
                                return 'background-color: orange; color: white'
                            else:
                                return 'background-color: red; color: white'
                        return ''
                    
                    # Tabloyu göster
                    styled_df = results_df.style.map(color_prediction, subset=['Tavsiye'])
                    styled_df = styled_df.map(color_probability, subset=['Yükseliş Olasılığı'])
                    
                    st.dataframe(styled_df, hide_index=True, use_container_width=True)
                    
                    # Özet istatistikler
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        buy_count = len(results_df[results_df['Tavsiye'] == 'AL'])
                        st.metric("AL Tavsiyesi", buy_count)
                    
                    with col2:
                        attention_count = len(results_df[results_df['Tavsiye'] == 'DİKKAT'])
                        st.metric("DİKKAT", attention_count)
                    
                    with col3:
                        sell_count = len(results_df[results_df['Tavsiye'] == 'SAT'])
                        st.metric("SAT Tavsiyesi", sell_count)
                    
                    # En yüksek olasılığa sahip hisseler
                    st.subheader("En Yüksek Potansiyelli Hisseler")
                    top_picks = results_df.head(5)
                    
                    for _, row in top_picks.iterrows():
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                        
                        with col1:
                            st.write(f"**{row['Hisse']}**")
                        
                        with col2:
                            st.write(f"{row['Fiyat']:.2f} TL")
                        
                        with col3:
                            change_color = "green" if row['Günlük %'] > 0 else "red"
                            st.markdown(f"<span style='color: {change_color}'>{row['Günlük %']:.2f}%</span>", unsafe_allow_html=True)
                        
                        with col4:
                            prob_color = "green" if row['Yükseliş Olasılığı'] > 60 else ("orange" if row['Yükseliş Olasılığı'] > 40 else "red")
                            st.markdown(f"<span style='color: {prob_color}'>{row['Yükseliş Olasılığı']:.1f}%</span>", unsafe_allow_html=True)
                    
                else:
                    st.warning("Analiz edilebilir hisse bulunamadı")
                    
            except ImportError:
                st.error("ML analizi için scikit-learn kütüphanesi gerekli. Lütfen yükleyin: pip install scikit-learn")
            except Exception as e:
                st.error(f"ML analizi hatası: {str(e)}")
                logger.error(f"ML analizi hatası: {str(e)}")

import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import re
import traceback
from pathlib import Path

# Loglama yapılandırması
logger = logging.getLogger(__name__)

# ======================
# VERİ ALMA FONKSİYONLARI
# ======================

@st.cache_data(ttl=300)
def get_stock_data(symbol, period="1y", interval="1d"):
    """Hisse senedi verilerini al"""
    try:
        if not symbol.endswith('.IS'):
            symbol = symbol + '.IS'
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            logger.warning(f"Boş veri: {symbol}")
            return pd.DataFrame()
        
        # Tarih indeksini düzenle
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        return data
    except Exception as e:
        logger.error(f"Veri alınırken hata ({symbol}): {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_company_info(symbol):
    """Şirket bilgilerini al"""
    try:
        if not symbol.endswith('.IS'):
            symbol = symbol + '.IS'
        
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        return {
            "name": info.get("longName", symbol),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "marketCap": info.get("marketCap", 0),
            "employees": info.get("fullTimeEmployees", 0)
        }
    except Exception as e:
        logger.error(f"Şirket bilgisi alınırken hata ({symbol}): {e}")
        return {"name": symbol}

def get_market_summary():
    """Piyasa özetini al"""
    try:
        # BIST 100 verisi
        bist100 = yf.Ticker("XU100.IS")
        bist_data = bist100.history(period="5d")
        
        if not bist_data.empty:
            last_price = bist_data['Close'].iloc[-1]
            prev_price = bist_data['Close'].iloc[-2] if len(bist_data) > 1 else last_price
            change = ((last_price - prev_price) / prev_price) * 100
            volume = bist_data['Volume'].iloc[-1]
            
            status = "yükseliş" if change > 0 else ("düşüş" if change < 0 else "sabit")
            
            return {
                "bist100": {
                    "value": last_price,
                    "change": change,
                    "change_percent": change,
                    "volume": volume / 1e9,  # Milyar TL
                    "status": status
                }
            }
        else:
            raise Exception("BIST 100 verisi alınamadı")
            
    except Exception as e:
        logger.error(f"Piyasa özeti alınırken hata: {e}")
        return {
            "bist100": {
                "value": 10000,
                "change": 0,
                "change_percent": 0,
                "volume": 0,
                "status": "bilinmiyor"
            }
        }

def get_popular_stocks():
    """Popüler hisse listesi"""
    popular_list = [
        "THYAO", "GARAN", "ASELS", "AKBNK", "EREGL",
        "VAKBN", "TUPRS", "FROTO", "SASA", "KOZAA"
    ]
    
    stocks = []
    for symbol in popular_list:
        try:
            data = get_stock_data(symbol, "5d")
            if not data.empty:
                last_price = data['Close'].iloc[-1]
                prev_price = data['Close'].iloc[-2] if len(data) > 1 else last_price
                change_percent = ((last_price - prev_price) / prev_price) * 100
                
                stocks.append({
                    "symbol": symbol,
                    "name": get_company_info(symbol)["name"],
                    "value": last_price,
                    "change_percent": change_percent
                })
            else:
                stocks.append({
                    "symbol": symbol,
                    "name": symbol,
                    "value": 100.0,
                    "change_percent": random.uniform(-5, 5)
                })
        except:
            stocks.append({
                "symbol": symbol,
                "name": symbol,
                "value": 100.0,
                "change_percent": random.uniform(-5, 5)
            })
    
    return stocks

# ======================
# TEKNİK ANALİZ FONKSİYONLARI
# ======================

def calculate_indicators(df):
    """Teknik göstergeleri hesapla"""
    if df.empty:
        return df
    
    try:
        # Hareketli ortalamalar
        df['SMA5'] = df['Close'].rolling(window=5).mean()
        df['SMA10'] = df['Close'].rolling(window=10).mean()
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA100'] = df['Close'].rolling(window=100).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        
        df['EMA5'] = df['Close'].ewm(span=5).mean()
        df['EMA10'] = df['Close'].ewm(span=10).mean()
        df['EMA20'] = df['Close'].ewm(span=20).mean()
        df['EMA50'] = df['Close'].ewm(span=50).mean()
        df['EMA100'] = df['Close'].ewm(span=100).mean()
        df['EMA200'] = df['Close'].ewm(span=200).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['Middle_Band'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['Upper_Band'] = df['Middle_Band'] + (bb_std * 2)
        df['Lower_Band'] = df['Middle_Band'] - (bb_std * 2)
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_%K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
        df['Stoch_%D'] = df['Stoch_%K'].rolling(window=3).mean()
        
        # Williams %R
        df['Williams_%R'] = -100 * (high_14 - df['Close']) / (high_14 - low_14)
        
        return df
    except Exception as e:
        logger.error(f"Göstergeler hesaplanırken hata: {e}")
        return df

def get_signals(df):
    """Alım/satım sinyallerini hesapla"""
    signals = pd.DataFrame(index=df.index)
    
    try:
        # MA sinyalleri
        signals['SMA5_Signal'] = np.where(df['Close'] > df['SMA5'], 1, -1)
        signals['SMA10_Signal'] = np.where(df['Close'] > df['SMA10'], 1, -1)
        signals['SMA20_Signal'] = np.where(df['Close'] > df['SMA20'], 1, -1)
        signals['SMA50_Signal'] = np.where(df['Close'] > df['SMA50'], 1, -1)
        signals['SMA100_Signal'] = np.where(df['Close'] > df['SMA100'], 1, -1)
        signals['SMA200_Signal'] = np.where(df['Close'] > df['SMA200'], 1, -1)
        
        signals['EMA5_Signal'] = np.where(df['Close'] > df['EMA5'], 1, -1)
        signals['EMA10_Signal'] = np.where(df['Close'] > df['EMA10'], 1, -1)
        signals['EMA20_Signal'] = np.where(df['Close'] > df['EMA20'], 1, -1)
        signals['EMA50_Signal'] = np.where(df['Close'] > df['EMA50'], 1, -1)
        signals['EMA100_Signal'] = np.where(df['Close'] > df['EMA100'], 1, -1)
        signals['EMA200_Signal'] = np.where(df['Close'] > df['EMA200'], 1, -1)
        
        # Oscillator sinyalleri
        signals['RSI_Signal'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
        signals['MACD_Signal'] = np.where(df['MACD'] > df['MACD_Signal'], 1, -1)
        signals['Stoch_Signal'] = np.where(df['Stoch_%K'] < 20, 1, np.where(df['Stoch_%K'] > 80, -1, 0))
        signals['Williams_%R_Signal'] = np.where(df['Williams_%R'] < -80, 1, np.where(df['Williams_%R'] > -20, -1, 0))
        
        # Toplam sinyaller
        ma_signals = ['SMA5_Signal', 'SMA10_Signal', 'SMA20_Signal', 'SMA50_Signal', 'SMA100_Signal', 'SMA200_Signal',
                     'EMA5_Signal', 'EMA10_Signal', 'EMA20_Signal', 'EMA50_Signal', 'EMA100_Signal', 'EMA200_Signal']
        osc_signals = ['RSI_Signal', 'MACD_Signal', 'Stoch_Signal', 'Williams_%R_Signal']
        
        signals['Total_MA_Signal'] = signals[ma_signals].sum(axis=1)
        signals['Total_Oscillator_Signal'] = signals[osc_signals].sum(axis=1)
        signals['Total_Signal'] = signals['Total_MA_Signal'] + signals['Total_Oscillator_Signal']
        
        return signals
    except Exception as e:
        logger.error(f"Sinyaller hesaplanırken hata: {e}")
        return signals

def create_stock_chart(df, symbol):
    """Hisse senedi grafiği oluştur"""
    try:
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f"{symbol} Fiyat Grafiği", "RSI", "MACD")
        )
        
        # Mum grafiği
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=symbol
            ), row=1, col=1
        )
        
        # Hareketli ortalamalar
        for sma in ['SMA20', 'SMA50', 'SMA200']:
            if sma in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[sma],
                        mode='lines',
                        name=sma,
                        line=dict(width=1)
                    ), row=1, col=1
                )
        
        # Bollinger Bands
        if all(col in df.columns for col in ['Upper_Band', 'Lower_Band']):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Upper_Band'],
                    mode='lines',
                    name='BB Üst',
                    line=dict(color='gray', dash='dash')
                ), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Lower_Band'],
                    mode='lines',
                    name='BB Alt',
                    line=dict(color='gray', dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(0,100,80,0.1)'
                ), row=1, col=1
            )
        
        # RSI
        if 'RSI' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='orange')
                ), row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        if all(col in df.columns for col in ['MACD', 'MACD_Signal']):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue')
                ), row=3, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MACD_Signal'],
                    mode='lines',
                    name='MACD Signal',
                    line=dict(color='red')
                ), row=3, col=1
            )
        
        fig.update_layout(
            height=800,
            xaxis_rangeslider_visible=False,
            template="plotly_white"
        )
        
        return fig
    except Exception as e:
        logger.error(f"Grafik oluşturulurken hata: {e}")
        return go.Figure()

# ======================
# ML TAHMİN FONKSİYONLARI
# ======================

def prepare_features(df):
    """ML için özellikleri hazırla"""
    features = df.copy()
    
    # Temel özellikler
    features['Price_Change_1d'] = features['Close'].pct_change()
    features['Price_Change_5d'] = features['Close'].pct_change(5)
    features['Volume_Change'] = features['Volume'].pct_change()
    
    # Teknik göstergeler normalleştirilmiş
    features['RSI_normalized'] = (features['RSI'] - 50) / 50
    features['MACD_normalized'] = features['MACD'] / features['Close']
    
    # Bollinger Bands pozisyonu
    features['BB_Position'] = (features['Close'] - features['Lower_Band']) / (features['Upper_Band'] - features['Lower_Band'])
    
    # Trend göstergeleri
    features['SMA20_Trend'] = (features['SMA20'] > features['SMA20'].shift(1)).astype(int)
    features['SMA50_Trend'] = (features['SMA50'] > features['SMA50'].shift(1)).astype(int)
    
    # Momentum
    features['Momentum_5d'] = features['Close'] / features['Close'].shift(5) - 1
    features['Momentum_10d'] = features['Close'] / features['Close'].shift(10) - 1
    
    return features

def create_ml_target(df, threshold=0.02, days=5):
    """ML için hedef değişkeni oluştur"""
    future_return = df['Close'].shift(-days) / df['Close'] - 1
    target = (future_return > threshold).astype(int)
    return target

def train_simple_model(features, target):
    """Basit ML modeli eğit"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        
        # NaN değerleri temizle
        mask = ~(features.isna().any(axis=1) | target.isna())
        X = features[mask]
        y = target[mask]
        
        if len(X) < 50:
            return None, "Yeterli veri yok"
        
        # Sadece numerik sütunları seç
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        # Sonsuz değerleri temizle
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Model eğitimi
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Test performansı
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return model, f"Accuracy: {accuracy:.3f}"
        
    except Exception as e:
        return None, f"Model eğitimi hatası: {str(e)}"

# ======================
# FAVORİ FONKSİYONLARI
# ======================

def add_to_favorites(stock_symbol):
    """Favorilere hisse ekle"""
    try:
        stock_symbol = stock_symbol.upper().strip()
        
        if 'favorite_stocks' not in st.session_state:
            st.session_state.favorite_stocks = []
        
        if stock_symbol not in st.session_state.favorite_stocks:
            st.session_state.favorite_stocks.append(stock_symbol)
            logger.info(f"{stock_symbol} favorilere eklendi")
            return True
        
        return False
    except Exception as e:
        logger.error(f"Favori eklenirken hata: {str(e)}")
        return False

def remove_from_favorites(stock_symbol):
    """Favorilerden hisse çıkar"""
    try:
        if 'favorite_stocks' in st.session_state and stock_symbol in st.session_state.favorite_stocks:
            st.session_state.favorite_stocks.remove(stock_symbol)
            logger.info(f"{stock_symbol} favorilerden çıkarıldı")
            return True
        return False
    except Exception as e:
        logger.error(f"Favori çıkarılırken hata: {str(e)}")
        return False

def is_favorite(stock_symbol):
    """Hissenin favorilerde olup olmadığını kontrol et"""
    if 'favorite_stocks' not in st.session_state:
        return False
    return stock_symbol.upper().strip() in st.session_state.favorite_stocks

# ======================
# RENDER FONKSİYONLARI
# ======================

def render_bist100_tab():
    """BIST 100 genel bakış sekmesi"""
    st.header("BIST 100 Genel Bakış", divider="blue")
    
    # BIST-100 verilerini al
    with st.spinner("BIST-100 verileri yükleniyor..."):
        bist100_data = get_stock_data("XU100", "6mo")
        
        if not bist100_data.empty:
            bist100_data = calculate_indicators(bist100_data)
            
            # Günlük veriler
            last_price = bist100_data['Close'].iloc[-1]
            prev_price = bist100_data['Close'].iloc[-2] if len(bist100_data) > 1 else last_price
            change = ((last_price - prev_price) / prev_price) * 100
            today_high = bist100_data['High'].iloc[-1]
            today_low = bist100_data['Low'].iloc[-1]
            volume = bist100_data['Volume'].iloc[-1]
            
            # Metrikler
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("BIST 100", f"{last_price:.0f}", f"{change:.2f}%")
            
            with col2:
                st.metric("Günlük Aralık", f"{today_low:.0f} - {today_high:.0f}")
            
            with col3:
                st.metric("Hacim", f"{volume:,.0f}")
            
            with col4:
                # Haftalık değişim
                if len(bist100_data) >= 6:
                    weekly_change = ((last_price - bist100_data['Close'].iloc[-6]) / bist100_data['Close'].iloc[-6]) * 100
                    st.metric("Haftalık", f"{weekly_change:.2f}%")
                else:
                    st.metric("Haftalık", "N/A")
            
            # Grafik
            fig = create_stock_chart(bist100_data, "BIST-100")
            st.plotly_chart(fig, use_container_width=True)
            
            # Teknik analiz özeti
            st.subheader("Teknik Analiz Özeti")
            signals = get_signals(bist100_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Trend Analizi**")
                trend_data = {
                    "Gösterge": ["SMA20 vs SMA50", "SMA50 vs SMA200", "MACD"],
                    "Durum": [
                        "Yükseliş" if bist100_data['SMA20'].iloc[-1] > bist100_data['SMA50'].iloc[-1] else "Düşüş",
                        "Yükseliş" if bist100_data['SMA50'].iloc[-1] > bist100_data['SMA200'].iloc[-1] else "Düşüş",
                        "Pozitif" if bist100_data['MACD'].iloc[-1] > bist100_data['MACD_Signal'].iloc[-1] else "Negatif"
                    ]
                }
                st.dataframe(pd.DataFrame(trend_data), hide_index=True)
            
            with col2:
                st.write("**Momentum Göstergeleri**")
                momentum_data = {
                    "Gösterge": ["RSI", "Stochastic %K", "Williams %R"],
                    "Değer": [
                        f"{bist100_data['RSI'].iloc[-1]:.1f}",
                        f"{bist100_data['Stoch_%K'].iloc[-1]:.1f}",
                        f"{bist100_data['Williams_%R'].iloc[-1]:.1f}"
                    ],
                    "Durum": [
                        "Aşırı Alım" if bist100_data['RSI'].iloc[-1] > 70 else ("Aşırı Satım" if bist100_data['RSI'].iloc[-1] < 30 else "Normal"),
                        "Aşırı Alım" if bist100_data['Stoch_%K'].iloc[-1] > 80 else ("Aşırı Satım" if bist100_data['Stoch_%K'].iloc[-1] < 20 else "Normal"),
                        "Aşırı Satım" if bist100_data['Williams_%R'].iloc[-1] < -80 else ("Aşırı Alım" if bist100_data['Williams_%R'].iloc[-1] > -20 else "Normal")
                    ]
                }
                st.dataframe(pd.DataFrame(momentum_data), hide_index=True)
            
        else:
            st.error("BIST-100 verileri alınamadı")
    
    # Popüler hisseler
    st.subheader("Popüler Hisseler")
    
    with st.spinner("Popüler hisseler yükleniyor..."):
        popular_stocks = get_popular_stocks()
        
        # 5'er sütun halinde göster
        for i in range(0, len(popular_stocks), 5):
            cols = st.columns(5)
            for j, col in enumerate(cols):
                if i + j < len(popular_stocks):
                    stock = popular_stocks[i + j]
                    with col:
                        change_color = "green" if stock["change_percent"] > 0 else ("red" if stock["change_percent"] < 0 else "gray")
                        arrow = "↑" if stock["change_percent"] > 0 else ("↓" if stock["change_percent"] < 0 else "→")
                        
                        st.markdown(f"""
                        <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 10px; text-align: center;">
                            <h5 style="margin: 0; color: {change_color};">{stock["symbol"]} {arrow}</h5>
                            <p style="margin: 0; font-size: 12px; color: gray;">{stock["name"][:15]}...</p>
                            <p style="margin: 0; color: {change_color}; font-weight: bold;">{stock["change_percent"]:.2f}%</p>
                            <p style="margin: 0; font-size: 12px;">{stock["value"]:.2f} TL</p>
                        </div>
                        """, unsafe_allow_html=True)

def render_stock_tab():
    """Hisse analizi sekmesi"""
    st.header("Hisse Senedi Teknik Analizi")
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    # Session state'den selected_stock_for_analysis'i kontrol et
    initial_stock = ""
    if 'selected_stock_for_analysis' in st.session_state and st.session_state.selected_stock_for_analysis:
        initial_stock = st.session_state.selected_stock_for_analysis
        st.session_state.selected_stock_for_analysis = ""
    
    with col1:
        stock_symbol = st.text_input("Hisse Senedi Kodu", value=initial_stock)
    
    with col2:
        period = st.selectbox("Zaman Aralığı", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    
    with col3:
        interval = st.selectbox("Veri Aralığı", ["1d", "1wk"], index=0)
    
    with col4:
        st.write("")
        analyze = st.button("Analiz Et", use_container_width=True)
    
    # Analiz yap
    if analyze and stock_symbol:
        with st.spinner(f"{stock_symbol} analiz ediliyor..."):
            try:
                stock_symbol = stock_symbol.upper().strip()
                df = get_stock_data(stock_symbol, period, interval)
                
                if not df.empty:
                    df = calculate_indicators(df)
                    signals = get_signals(df)
                    company_info = get_company_info(stock_symbol)
                    
                    # Temel bilgiler
                    last_price = df['Close'].iloc[-1]
                    prev_price = df['Close'].iloc[-2] if len(df) > 1 else last_price
                    change = ((last_price - prev_price) / prev_price) * 100
                    
                    col_info, col_fav = st.columns([4, 1])
                    
                    with col_info:
                        col_metrics = st.columns(4)
                        
                        with col_metrics[0]:
                            st.metric("Son Fiyat", f"{last_price:.2f} TL", f"{change:.2f}%")
                        
                        with col_metrics[1]:
                            rsi = df['RSI'].iloc[-1]
                            st.metric("RSI", f"{rsi:.1f}")
                        
                        with col_metrics[2]:
                            macd = df['MACD'].iloc[-1]
                            st.metric("MACD", f"{macd:.4f}")
                        
                        with col_metrics[3]:
                            volume = df['Volume'].iloc[-1]
                            st.metric("Hacim", f"{volume:,.0f}")
                    
                    with col_fav:
                        st.write("")
                        if st.button("⭐ Favorilere Ekle", use_container_width=True):
                            if add_to_favorites(stock_symbol):
                                st.success("Favorilere eklendi!")
                                st.rerun()
                            else:
                                st.warning("Zaten favorilerde")
                    
                    # Şirket bilgileri
                    if company_info.get("name") != stock_symbol:
                        st.info(f"**{company_info['name']}** - {company_info.get('sector', 'N/A')} sektörü")
                    
                    # Grafik
                    fig = create_stock_chart(df, stock_symbol)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Teknik göstergeler tablosu
                    st.subheader("Teknik Göstergeler")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Hareketli Ortalamalar**")
                        ma_data = {
                            "Gösterge": ["SMA5", "SMA20", "SMA50", "SMA200"],
                            "Değer": [
                                f"{df['SMA5'].iloc[-1]:.2f}",
                                f"{df['SMA20'].iloc[-1]:.2f}",
                                f"{df['SMA50'].iloc[-1]:.2f}",
                                f"{df['SMA200'].iloc[-1]:.2f}"
                            ],
                            "Sinyal": [
                                "AL" if signals['SMA5_Signal'].iloc[-1] > 0 else "SAT",
                                "AL" if signals['SMA20_Signal'].iloc[-1] > 0 else "SAT",
                                "AL" if signals['SMA50_Signal'].iloc[-1] > 0 else "SAT",
                                "AL" if signals['SMA200_Signal'].iloc[-1] > 0 else "SAT"
                            ]
                        }
                        
                        def color_signal(val):
                            if val == "AL":
                                return 'background-color: green; color: white'
                            elif val == "SAT":
                                return 'background-color: red; color: white'
                            return ''
                        
                        ma_df = pd.DataFrame(ma_data)
                        st.dataframe(ma_df.style.map(color_signal, subset=['Sinyal']), hide_index=True, use_container_width=True)
                    
                    with col2:
                        st.write("**Osilatörler**")
                        osc_data = {
                            "Gösterge": ["RSI", "MACD", "Stochastic", "Williams %R"],
                            "Değer": [
                                f"{df['RSI'].iloc[-1]:.1f}",
                                f"{df['MACD'].iloc[-1]:.4f}",
                                f"{df['Stoch_%K'].iloc[-1]:.1f}",
                                f"{df['Williams_%R'].iloc[-1]:.1f}"
                            ],
                            "Sinyal": [
                                "AL" if signals['RSI_Signal'].iloc[-1] > 0 else ("SAT" if signals['RSI_Signal'].iloc[-1] < 0 else "NÖTR"),
                                "AL" if signals['MACD_Signal'].iloc[-1] > 0 else "SAT",
                                "AL" if signals['Stoch_Signal'].iloc[-1] > 0 else ("SAT" if signals['Stoch_Signal'].iloc[-1] < 0 else "NÖTR"),
                                "AL" if signals['Williams_%R_Signal'].iloc[-1] > 0 else ("SAT" if signals['Williams_%R_Signal'].iloc[-1] < 0 else "NÖTR")
                            ]
                        }
                        
                        def color_signal_osc(val):
                            if val == "AL":
                                return 'background-color: green; color: white'
                            elif val == "SAT":
                                return 'background-color: red; color: white'
                            else:
                                return 'background-color: gray; color: white'
                        
                        osc_df = pd.DataFrame(osc_data)
                        st.dataframe(osc_df.style.map(color_signal_osc, subset=['Sinyal']), hide_index=True, use_container_width=True)
                    
                    with col3:
                        st.write("**Analiz Özeti**")
                        
                        # Sinyalleri say
                        ma_signals = ['SMA5_Signal', 'SMA20_Signal', 'SMA50_Signal', 'SMA200_Signal']
                        osc_signals = ['RSI_Signal', 'MACD_Signal', 'Stoch_Signal', 'Williams_%R_Signal']
                        
                        ma_buy = sum(1 for sig in ma_signals if signals[sig].iloc[-1] > 0)
                        ma_sell = sum(1 for sig in ma_signals if signals[sig].iloc[-1] < 0)
                        
                        osc_buy = sum(1 for sig in osc_signals if signals[sig].iloc[-1] > 0)
                        osc_sell = sum(1 for sig in osc_signals if signals[sig].iloc[-1] < 0)
                        
                        total_buy = ma_buy + osc_buy
                        total_sell = ma_sell + osc_sell
                        total_signals = len(ma_signals) + len(osc_signals)
                        
                        summary_data = {
                            "Kategori": ["Hareketli Ort.", "Osilatörler", "Genel"],
                            "Al": [f"{ma_buy}/4", f"{osc_buy}/4", f"{total_buy}/8"],
                            "Sat": [f"{ma_sell}/4", f"{osc_sell}/4", f"{total_sell}/8"],
                            "Sonuç": [
                                "AL" if ma_buy > ma_sell else ("SAT" if ma_sell > ma_buy else "NÖTR"),
                                "AL" if osc_buy > osc_sell else ("SAT" if osc_sell > osc_buy else "NÖTR"),
                                "GÜÇLÜ AL" if total_buy > 6 else ("AL" if total_buy > total_sell else ("GÜÇLÜ SAT" if total_sell > 6 else ("SAT" if total_sell > total_buy else "NÖTR")))
                            ]
                        }
                        
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df.style.map(color_signal_osc, subset=['Sonuç']), hide_index=True, use_container_width=True)
                        
                        # Final recommendation
                        final_rec = summary_df['Sonuç'].iloc[2]
                        rec_color = "green" if "AL" in final_rec else ("red" if "SAT" in final_rec else "gray")
                        st.markdown(f"<h3 style='text-align: center; color: {rec_color};'>{final_rec}</h3>", unsafe_allow_html=True)
                    
                    # Performans analizi
                    st.subheader("Performans Analizi")
                    
                    perf_cols = st.columns(4)
                    periods = [5, 20, 60, 252]
                    period_names = ["1 Hafta", "1 Ay", "3 Ay", "1 Yıl"]
                    
                    for i, (period, name) in enumerate(zip(periods, period_names)):
                        with perf_cols[i]:
                            if len(df) > period:
                                perf = ((df['Close'].iloc[-1] / df['Close'].iloc[-period-1]) - 1) * 100
                                st.metric(name, f"{perf:.2f}%")
                            else:
                                st.metric(name, "N/A")
                
                else:
                    st.error("Hisse senedi verisi bulunamadı")
                    
            except Exception as e:
                st.error(f"Analiz hatası: {str(e)}")
                logger.error(f"Analiz hatası: {str(e)}")

def render_ml_prediction_tab():
    """ML tahmin sekmesi"""
    st.header("ML Yükseliş Tahmini", divider="rainbow")
    
    st.markdown("""
    Bu bölüm, makine öğrenmesi algoritmaları kullanarak hisse senetlerinin gelecekteki performansını tahmin eder.
    
    **Nasıl Çalışır:**
    - Teknik göstergeler, fiyat hareketleri ve hacim verileri analiz edilir
    - RandomForest algoritması ile model eğitilir
    - Belirlediğiniz eşik değerini aşma olasılığı hesaplanır
    
    **Not:** Bu tahminler yatırım tavsiyesi değildir, sadece analiz amaçlıdır.
    """)
    
    # Parametreler
    col1, col2, col3 = st.columns(3)
    
    with col1:
        threshold = st.slider("Yükseliş Eşiği (%)", 1.0, 10.0, 3.0, 0.5) / 100
    
    with col2:
        scan_option = st.selectbox("Tarama Türü", ["BIST 30", "BIST 50", "Popüler Hisseler", "Özel Liste"])
    
    with col3:
        prediction_days = st.selectbox("Tahmin Süresi", [1, 5, 10, 20], index=1)
    
    # Özel liste için
    custom_stocks = ""
    if scan_option == "Özel Liste":
        custom_stocks = st.text_area("Hisse Kodları (virgülle ayırın)", placeholder="THYAO, GARAN, ASELS")
    
    # Hisse listesini belirle
    if scan_option == "BIST 30":
        stock_list = ["THYAO", "GARAN", "AKBNK", "EREGL", "TUPRS", "FROTO", "ASELS", "SASA", "KOZAA", "VAKBN",
                     "HALKB", "ISCTR", "YKBNK", "MGROS", "BIMAS", "TOASO", "KCHOL", "SAHOL", "DOHOL", "TCELL"]
    elif scan_option == "BIST 50":
        stock_list = ["THYAO", "GARAN", "AKBNK", "EREGL", "TUPRS", "FROTO", "ASELS", "SASA", "KOZAA", "VAKBN",
                     "HALKB", "ISCTR", "YKBNK", "MGROS", "BIMAS", "TOASO", "KCHOL", "SAHOL", "DOHOL", "TCELL",
                     "TTKOM", "PGSUS", "SOKM", "MAVI", "TAVHL", "KARSN", "OTKAR", "ARCLK", "VESTL", "PETKM"]
    elif scan_option == "Popüler Hisseler":
        stock_list = ["THYAO", "GARAN", "ASELS", "AKBNK", "EREGL", "VAKBN", "TUPRS", "FROTO", "SASA", "KOZAA"]
    else:  # Özel Liste
        if custom_stocks:
            stock_list = [s.strip().upper() for s in custom_stocks.split(",") if s.strip()]
        else:
            stock_list = []
    
    if st.button("ML Tarama Başlat", type="primary", use_container_width=True):
        if not stock_list:
            st.error("Analiz edilecek hisse bulunamadı")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []
        
        try:
            # ML kütüphanelerini kontrol et
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            
            for i, symbol in enumerate(stock_list):
                status_text.text(f"Analiz ediliyor: {symbol} ({i+1}/{len(stock_list)})")
                progress_bar.progress((i + 1) / len(stock_list))
                
                try:
                    # Veri al
                    df = get_stock_data(symbol, "2y")
                    
                    if df.empty or len(df) < 100:
                        continue
                    
                    # Teknik göstergeleri hesapla
                    df = calculate_indicators(df)
                    
                    # Özellikleri hazırla
                    features = prepare_features(df)
                    target = create_ml_target(df, threshold, prediction_days)
                    
                    # Model eğit
                    model, status = train_simple_model(features, target)
                    
                    if model is not None:
                        # Son veri için tahmin yap
                        last_features = features.iloc[-1:].select_dtypes(include=[np.number]).fillna(0)
                        last_features = last_features.replace([np.inf, -np.inf], 0)
                        
                        if len(last_features.columns) > 0:
                            prediction_proba = model.predict_proba(last_features)[0]
                            confidence = prediction_proba[1] if len(prediction_proba) > 1 else 0.5
                            
                            # Son fiyat bilgileri
                            last_price = df['Close'].iloc[-1]
                            daily_change = ((last_price / df['Close'].iloc[-2]) - 1) * 100 if len(df) > 1 else 0
                            
                            results.append({
                                "Hisse": symbol,
                                "Fiyat": last_price,
                                "Günlük %": daily_change,
                                "Yükseliş Olasılığı": confidence * 100,
                                "Model Durumu": status,
                                "Tavsiye": "AL" if confidence > 0.6 else ("DİKKAT" if confidence > 0.4 else "SAT")
                            })
                
                except Exception as e:
                    logger.error(f"{symbol} analiz hatası: {str(e)}")
                    continue
            
            progress_bar.empty()
            status_text.empty()
            
            if results:
                # Sonuçları sırala
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values("Yükseliş Olasılığı", ascending=False)
                
                st.subheader("ML Tahmin Sonuçları")
                
                # Renklendirme fonksiyonu
                def color_prediction(val):
                    if isinstance(val, str):
                        if val == "AL":
                            return 'background-color: green; color: white'
                        elif val == "SAT":
                            return 'background-color: red; color: white'
                        elif val == "DİKKAT":
                            return 'background-color: orange; color: white'
                    return ''
                
                def color_probability(val):
                    if isinstance(val, (int, float)):
                        if val > 70:
                            return 'background-color: darkgreen; color: white'
                        elif val > 60:
                            return 'background-color: green; color: white'
                        elif val > 40:
                            return 'background-color: orange; color: white'
                        else:
                            return 'background-color: red; color: white'
                    return ''
                
                # Tabloyu göster
                styled_df = results_df.style.map(color_prediction, subset=['Tavsiye'])
                styled_df = styled_df.map(color_probability, subset=['Yükseliş Olasılığı'])
                
                st.dataframe(styled_df, hide_index=True, use_container_width=True)
                
                # Özet istatistikler
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    buy_count = len(results_df[results_df['Tavsiye'] == 'AL'])
                    st.metric("AL Tavsiyesi", buy_count)
                
                with col2:
                    attention_count = len(results_df[results_df['Tavsiye'] == 'DİKKAT'])
                    st.metric("DİKKAT", attention_count)
                
                with col3:
                    sell_count = len(results_df[results_df['Tavsiye'] == 'SAT'])
                    st.metric("SAT Tavsiyesi", sell_count)
                
                # En yüksek olasılığa sahip hisseler
                st.subheader("En Yüksek Potansiyelli Hisseler")
                top_picks = results_df.head(5)
                
                for _, row in top_picks.iterrows():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    
                    with col1:
                        st.write(f"**{row['Hisse']}**")
                    
                    with col2:
                        st.write(f"{row['Fiyat']:.2f} TL")
                    
                    with col3:
                        change_color = "green" if row['Günlük %'] > 0 else "red"
                        st.markdown(f"<span style='color: {change_color}'>{row['Günlük %']:.2f}%</span>", unsafe_allow_html=True)
                    
                    with col4:
                        prob_color = "green" if row['Yükseliş Olasılığı'] > 60 else ("orange" if row['Yükseliş Olasılığı'] > 40 else "red")
                        st.markdown(f"<span style='color: {prob_color}'>{row['Yükseliş Olasılığı']:.1f}%</span>", unsafe_allow_html=True)
                
            else:
                st.warning("Analiz edilebilir hisse bulunamadı")
                
        except ImportError:
            st.error("ML analizi için scikit-learn kütüphanesi gerekli. Lütfen yükleyin: pip install scikit-learn")
        except Exception as e:
            st.error(f"ML analizi hatası: {str(e)}")
            logger.error(f"ML analizi hatası: {str(e)}")

# ======================
# ANA UYGULAMA
# ======================

def main():
    """Ana uygulama fonksiyonu"""
    
    # Sayfa konfigürasyonu
    st.set_page_config(
        page_title="Kompakt Borsa Analizi",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Session state başlat
    if 'favorite_stocks' not in st.session_state:
        st.session_state.favorite_stocks = []
    
    # Ana başlık
    st.title("📈 Kompakt Borsa Analizi")
    st.markdown("*Hisse senedi teknik analizi, BIST 100 takibi ve ML tahminleri*")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Kontrol Paneli")
        
        # Favoriler bölümü
        st.subheader("⭐ Favoriler")
        if st.session_state.favorite_stocks:
            for fav_stock in st.session_state.favorite_stocks:
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(f"📊 {fav_stock}", key=f"fav_{fav_stock}", use_container_width=True):
                        st.session_state.selected_stock_for_analysis = fav_stock
                        st.rerun()
                with col2:
                    if st.button("❌", key=f"del_{fav_stock}"):
                        remove_from_favorites(fav_stock)
                        st.rerun()
        else:
            st.info("Henüz favori hisse yok")
        
        st.divider()
        
        # Hızlı erişim
        st.subheader("🚀 Hızlı Erişim")
        
        if st.button("📈 BIST 100 Görünümü", use_container_width=True):
            st.session_state.tab_selection = "BIST 100"
            st.rerun()
        
        if st.button("🔍 Hisse Analizi", use_container_width=True):
            st.session_state.tab_selection = "Hisse Analizi"
            st.rerun()
        
        if st.button("🤖 ML Tahminleri", use_container_width=True):
            st.session_state.tab_selection = "ML Tahminleri"
            st.rerun()
        
        st.divider()
        
        # Piyasa durumu
        st.subheader("📊 Piyasa Durumu")
        try:
            bist100_data = get_stock_data("XU100", "5d")
            if not bist100_data.empty:
                last_price = bist100_data['Close'].iloc[-1]
                prev_price = bist100_data['Close'].iloc[-2] if len(bist100_data) > 1 else last_price
                change = ((last_price - prev_price) / prev_price) * 100
                
                st.metric("BIST 100", f"{last_price:.0f}", f"{change:.2f}%")
                
                # Son 5 güne göre trend
                if len(bist100_data) >= 5:
                    weekly_change = ((last_price - bist100_data['Close'].iloc[-5]) / bist100_data['Close'].iloc[-5]) * 100
                    trend_icon = "📈" if weekly_change > 0 else "📉"
                    st.write(f"{trend_icon} 5 Günlük: {weekly_change:.2f}%")
                
            else:
                st.warning("Piyasa verileri yüklenemedi")
        except:
            st.warning("Piyasa verileri yüklenemedi")
        
        st.divider()
        
        # Bilgi
        st.subheader("ℹ️ Bilgi")
        st.markdown("""
        **Kompakt Borsa Analizi**
        
        Bu uygulama:
        - ✅ BIST 100 takibi
        - ✅ Teknik analiz
        - ✅ ML tahminleri
        - ✅ Favori hisseler
        
        *Veriler yfinance'dan alınır*
        """)
    
    # Ana sekmeler
    if 'tab_selection' not in st.session_state:
        st.session_state.tab_selection = "BIST 100"
    
    # Tab kontrolü
    tab1, tab2, tab3 = st.tabs(["📊 BIST 100", "🔍 Hisse Analizi", "🤖 ML Tahminleri"])
    
    with tab1:
        if st.session_state.tab_selection == "BIST 100" or True:  # Her zaman göster
            render_bist100_tab()
    
    with tab2:
        if st.session_state.tab_selection == "Hisse Analizi" or True:  # Her zaman göster
            render_stock_tab()
    
    with tab3:
        if st.session_state.tab_selection == "ML Tahminleri" or True:  # Her zaman göster
            render_ml_prediction_tab()

if __name__ == "__main__":
    main()

def main():
    """
    Kompakt borsa uygulaması - BIST 100, Hisse Analizi ve ML Tahminleri
    """
    
    # CSS stil ekle - Sade tasarım
    st.markdown("""
    <style>
        /* Ana stil ayarları */
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            color: #333;
            background-color: #f9f9f9;
        }
        
        /* Streamlit başlık stillerini düzenle */
        h1 {
            font-size: 1.5rem !important;
            margin-top: 0 !important;
            margin-bottom: 0.5rem !important;
            padding-top: 0 !important;
        }
        
        h2 {
            font-size: 1.3rem !important;
            margin-top: 0 !important;
            margin-bottom: 0.4rem !important;
        }
        
        h3 {
            font-size: 1.1rem !important;
            margin-top: 0 !important;
            margin-bottom: 0.3rem !important;
        }
        
        /* Sade header */
        .main-header {
            background-color: #f5f5f5;
            padding: 1.5rem;
            border-radius: 5px;
            border: 1px solid #e0e0e0;
            margin-bottom: 1.5rem;
            text-align: center;
        }
        
        .main-header h1 {
            font-weight: 600;
            margin-bottom: 0.3rem;
            font-size: 1.8rem;
            color: #333;
        }
        
        .main-header p {
            font-size: 1rem;
            color: #666;
        }
        
        /* Sade sekme tasarımı */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.1rem;
            border-radius: 5px;
            padding: 0.2rem;
            background-color: #f5f5f5;
            border: 1px solid #e0e0e0;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 4px;
            padding: 0.3rem 0.7rem;
            font-weight: 500;
            background-color: #f5f5f5;
            transition: all 0.2s ease;
            border: none !important;
            font-size: 0.8rem;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e9e9e9;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #e0e0e0 !important;
            color: #333 !important;
        }
        
        /* Sidebar stil ayarları */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
            border-right: 1px solid #e0e0e0;
        }
        
        [data-testid="stSidebar"] > div:first-child {
            background-color: #f8f9fa;
            padding-top: 1rem;
        }
        
        /* Ana içerik alanını genişlet */
        .main .block-container {
            max-width: 100%;
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
        }
        
        /* Sade kartlar */
        div[data-testid="stBlock"] {
            background-color: #fff;
            padding: 0.5rem;
            border-radius: 5px;
            border: 1px solid #e0e0e0;
            margin-bottom: 0.5rem;
        }
        
        /* Sade metrik kartları */
        [data-testid="stMetric"] {
            background-color: #f9f9f9;
            padding: 0.5rem;
            border-radius: 5px;
            border: 1px solid #e0e0e0;
        }
        
        /* Sade butonlar */
        .stButton button {
            border-radius: 4px;
            font-weight: 500;
            background-color: #f0f0f0;
            color: #333;
            border: 1px solid #ddd;
            padding: 0.5rem 1rem;
            transition: all 0.2s ease;
        }
        
        .stButton button:hover {
            background-color: #e5e5e5;
        }
        
        /* Analiz kartları */
        .analysis-card {
            background-color: #fff;
            border-radius: 5px;
            border: 1px solid #e0e0e0;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        .card-header {
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 0.8rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #eee;
            color: #333;
        }
        
        footer {
            visibility: hidden;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Session state değişkenlerini başlat
    if 'analyzed_stocks' not in st.session_state:
        st.session_state.analyzed_stocks = {}
    
    if 'favorite_stocks' not in st.session_state:
        st.session_state.favorite_stocks = []
    
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    if 'last_predictions' not in st.session_state:
        st.session_state.last_predictions = {}
    
    if 'realtime_data' not in st.session_state:
        st.session_state.realtime_data = {}
    
    if 'signals_cache' not in st.session_state:
        st.session_state.signals_cache = {}
    
    if 'trend_cache' not in st.session_state:
        st.session_state.trend_cache = {}
    
    if 'stock_analysis_results' not in st.session_state:
        st.session_state.stock_analysis_results = {}
    
    # Piyasa verilerini çek
    try:
        market_data = get_market_summary()
        popular_stocks_data = get_popular_stocks()
    except Exception as e:
        logger.error(f"Piyasa verileri alınırken hata: {str(e)}")
        # Streamlit Cloud için varsayılan veriler
        market_data = {
            "bist100": {
                "value": 10000, 
                "change": 0, 
                "change_percent": 0, 
                "volume": 0, 
                "status": "bilinmiyor"
            },
            "usdtry": {
                "value": 34.0,
                "change": 0,
                "change_percent": 0,
                "range": "33.5-34.5",
                "status": "bilinmiyor"
            },
            "gold": {
                "value": 2650.0,
                "change": 0,
                "change_percent": 0,
                "range": "2640-2660",
                "status": "bilinmiyor"
            }
        }
        popular_stocks_data = [
            {"symbol": "THYAO", "name": "Türk Hava Yolları", "value": 250.0, "change_percent": 2.5},
            {"symbol": "GARAN", "name": "Garanti BBVA", "value": 85.0, "change_percent": -1.2},
            {"symbol": "ASELS", "name": "Aselsan", "value": 75.0, "change_percent": 1.8},
            {"symbol": "AKBNK", "name": "Akbank", "value": 55.0, "change_percent": -0.5},
            {"symbol": "EREGL", "name": "Ereğli Demir Çelik", "value": 40.0, "change_percent": 3.2}
        ]
    
    # Uygulamanın ana başlığı
    st.markdown('<div class="main-header"><h1>📊 Kompakt Borsa Analiz Uygulaması</h1><p>BIST 100 Genel Bakış • Hisse Analizi • ML Yükseliş Tahminleri</p></div>', unsafe_allow_html=True)
    
    # Favori hisseler bölümü - yatay düzen
    if st.session_state.favorite_stocks:
        st.markdown('<div style="padding: 0.5rem; background-color: #f9f9f9; border-radius: 5px; border: 1px solid #e0e0e0; margin-bottom: 1rem;">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-bottom: 0.5rem;'>⭐ Favori Hisseler</h4>", unsafe_allow_html=True)
        
        cols = st.columns(min(len(st.session_state.favorite_stocks), 5))  
        for idx, stock_symbol in enumerate(st.session_state.favorite_stocks):
            col_idx = idx % 5
            with cols[col_idx]:
                st.markdown(f"<div style='text-align: center; margin-bottom: 0.5rem;'><strong>{stock_symbol}</strong></div>", unsafe_allow_html=True)       
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔍", key=f"analyze_{stock_symbol}", help="Hisseyi analiz et"):
                        # Analiz sekmesine yönlendir ve bu hisseyi analiz et
                        st.session_state.selected_stock_for_analysis = stock_symbol
                        st.info(f"{stock_symbol} analizi için Hisse Analizi sekmesine gidin.")
                with col2:
                    if st.button("❌", key=f"remove_{stock_symbol}", help="Favorilerden çıkar"):
                        if remove_from_favorites(stock_symbol):
                            st.success(f"{stock_symbol} favorilerden çıkarıldı.")
                            st.rerun()
                        else:
                            st.error("Hisse çıkarılırken bir hata oluştu.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Sidebar navigasyon menüsü
    st.sidebar.title("📊 Borsa Analiz Paneli")
    
    # Sayfa seçim menüsü
    selected_page = st.sidebar.selectbox(
        "Sayfa Seçin:",
        [
            "� Hisse Analizi",
            "�📊 BIST100 Genel Bakış", 
            "� ML Tarama"
        ],
        key="main_page_selector"
    )
    
    # Sidebar'da ek bilgiler
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Hızlı Erişim")
    
    # ML Tarama için hızlı erişim
    if st.sidebar.button("🔎 Hızlı ML Tarama", help="ML modelinizi hızlıca test edin"):
        selected_page = "🔎 ML Tarama"
        st.rerun()
    
    # Favoriler
    if st.session_state.favorite_stocks:
        st.sidebar.markdown("### ⭐ Favoriler")
        for stock in st.session_state.favorite_stocks[:5]:  # İlk 5 favoriyi göster
            if st.sidebar.button(f"📊 {stock}", key=f"sidebar_{stock}", help=f"{stock} hissesini analiz et"):
                st.session_state.selected_stock_for_analysis = stock
                selected_page = "🔍 Hisse Analizi"
                st.rerun()
    
    # Ana içerik
    col1, col2 = st.columns([7, 3])
    
    with col1:
        # Seçilen sayfayı render et
        if selected_page == "🔍 Hisse Analizi":
            render_stock_tab()
        elif selected_page == "📊 BIST100 Genel Bakış":
            render_bist100_tab()
        elif selected_page == "🔎 ML Tarama":
            render_ml_prediction_tab()
    
    with col2:
        # Piyasa Güncellemeleri kartı - Sade tasarım
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)  
        st.markdown('<div class="card-header">📊 Piyasa Güncellemeleri</div>', unsafe_allow_html=True)
        
        # BIST100 Verileri
        try:
            bist_data = market_data["bist100"]
            bist_status = bist_data["status"]
            bist_color = "green" if bist_status == "yükseliş" else ("red" if bist_status == "düşüş" else "gray")
            bist_arrow = "↑" if bist_status == "yükseliş" else ("↓" if bist_status == "düşüş" else "→")
            
            st.markdown(f"""
            <div style="padding: 0.8rem; background-color: #f5f5f5; border-radius: 5px; margin-bottom: 0.8rem; border: 1px solid #e0e0e0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                    <strong>BIST100</strong>
                    <span style="color: {bist_color}; font-weight: 600;">{bist_data["value"]:,.2f} {bist_arrow}</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.9rem; color: #666;">
                    <span>Değişim: {'+' if bist_data["change"] > 0 else ''}{bist_data["change_percent"]:.2f}%</span>
                    <span>Hacim: {bist_data["volume"]:.1f}B ₺</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.markdown("""
            <div style="padding: 0.8rem; background-color: #f5f5f5; border-radius: 5px; margin-bottom: 0.8rem; border: 1px solid #e0e0e0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                    <strong>BIST100</strong>
                    <span style="color: gray; font-weight: 600;">N/A</span>
                </div>
                <div style="font-size: 0.9rem; color: #666;">Veri alınamadı</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Son Analizler kartı - Sade tasarım
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)  
        st.markdown('<div class="card-header">🔄 Son Analizler</div>', unsafe_allow_html=True)
        
        # Veritabanından son analizleri getir
        try:
            # Ana modülleri kontrol et
            if MODULES_AVAILABLE:
                last_analyses = get_analysis_results(limit=3)  # Son 3 analizi al   
                
                if last_analyses:
                    for analysis in last_analyses:
                        stock = analysis.get("symbol", "N/A")
                        analysis_id = analysis.get("id", 0)
                        result_data = analysis.get("result_data", {})
                        if isinstance(result_data, dict):
                            recommendation = result_data.get("recommendation", "")
                        else:
                            recommendation = ""
                        recommendation_color = "green" if "AL" in recommendation else ("red" if "SAT" in recommendation else "gray")
                        
                        col_ana, col_but = st.columns([4, 1])
                        with col_ana:
                            st.markdown(f"""
                            <div style="padding: 0.5rem 0;">
                                <strong>{stock}</strong>
                                <div style="font-size: 0.8rem; color: {recommendation_color};">{recommendation}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col_but:
                            if st.button("🔄", key=f"reanalyze_{stock}_{analysis_id}", help="Yeniden analiz et"):
                                st.session_state.selected_stock_for_analysis = stock
                                st.info(f"{stock} için Hisse Analizi sekmesine gidin.")
                        
                        st.markdown("<hr style='margin: 0.3rem 0; border-color: #eee;'>", unsafe_allow_html=True)
                else:
                    st.info("Henüz analiz yapılmadı.")
            else:
                # Ana modüller yok, örnek veriler göster
                sample_analyses = [
                    {"symbol": "THYAO", "recommendation": "AL"},
                    {"symbol": "GARAN", "recommendation": "SAT"},
                    {"symbol": "ASELS", "recommendation": "AL"}
                ]
                
                for idx, analysis in enumerate(sample_analyses):
                    stock = analysis["symbol"]
                    recommendation = analysis["recommendation"]
                    recommendation_color = "green" if "AL" in recommendation else "red"
                    
                    col_ana, col_but = st.columns([4, 1])
                    with col_ana:
                        st.markdown(f"""
                        <div style="padding: 0.5rem 0;">
                            <strong>{stock}</strong>
                            <div style="font-size: 0.8rem; color: {recommendation_color};">{recommendation}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_but:
                        if st.button("🔄", key=f"sample_reanalyze_{stock}_{idx}", help="Yeniden analiz et"):
                            st.session_state.selected_stock_for_analysis = stock
                            st.info(f"{stock} için Hisse Analizi sekmesine gidin.")
                    
                    st.markdown("<hr style='margin: 0.3rem 0; border-color: #eee;'>", unsafe_allow_html=True)
        except Exception as e:
            st.info("Analiz geçmişi yüklenemedi.")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Popüler Hisseler kartı - Sade tasarım
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)  
        st.markdown('<div class="card-header">🔥 Popüler Hisseler</div>', unsafe_allow_html=True)
        
        if popular_stocks_data:
            for stock in popular_stocks_data[:5]:  # İlk 5 tanesini göster
                symbol = stock["symbol"]
                name = stock["name"]
                change_percent = stock["change_percent"]
                change_text = f"+{change_percent:.2f}%" if change_percent > 0 else f"{change_percent:.2f}%"
                change_color = "green" if change_percent > 0 else ("red" if change_percent < 0 else "gray")
                
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.8rem;
                     background-color: #f5f5f5; border-radius: 5px; margin-bottom: 0.5rem; border: 1px solid #e0e0e0;">
                    <div>
                        <strong>{symbol}</strong>
                        <div style="font-size: 0.85rem; color: #666;">{name[:15]}...</div>
                    </div>
                    <div>
                        <div style="color: {change_color}; font-weight: 600; text-align: right;">{change_text}</div>
                        <div style="font-size: 0.85rem; color: #666; text-align: right;">{stock["value"]:.2f} ₺</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Popüler hisse verileri yüklenemedi.")
        
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main() 