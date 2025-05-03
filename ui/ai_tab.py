import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

import traceback
import io
import logging

# Yapay zeka modüllerini ekle
from ai.api import initialize_gemini_api, ai_market_sentiment, ai_stock_analysis
from ai.api import ai_price_prediction, ai_sector_analysis, ai_portfolio_recommendation
from ai.api import ai_technical_interpretation

# Veri ve analiz modüllerinden gerekli fonksiyonları ekle
from data.stock_data import get_stock_data, get_company_info
from data.news_data import get_stock_news
from data.analysis_functions import calculate_indicators
from data.ai_functions import (
    ai_market_sentiment, 
    ai_stock_analysis,
    ai_price_prediction,
    ai_sector_analysis,
    ai_portfolio_recommendation,
    ai_technical_interpretation,
    load_gemini_pro
)
from data.visualization import create_stock_chart
from data.utils import save_analysis_result

# Loglama yapılandırması
logger = logging.getLogger(__name__)

def render_ai_tab():
    """
    Yapay Zeka sekmesini oluşturur
    """
    st.header("Yapay Zeka Analizleri", divider="rainbow")
    
    # İşlem günlüğü expander'ı - varsayılan olarak kapalı
    log_expander = st.expander("İşlem Günlüğü (Detaylar için tıklayın)", expanded=False)
    
    # Gemini API'yi başlat
    gemini_pro = initialize_gemini_api()
    
    # AI sekmeleri
    ai_tabs = st.tabs(["🔮 Piyasa Duyarlılığı", "🧠 Hisse Analizi", "📈 Fiyat Tahmini", 
                        "📊 Sektör Analizi", "💰 Portföy Önerileri", "📉 Teknik Analiz"])
    
    with ai_tabs[0]:
        st.subheader("Piyasa Genel Duyarlılığı")
        st.markdown("Bu bölümde yapay zeka, piyasanın genel durumunu analiz eder ve yatırımcı duyarlılığını değerlendirir.")
        
        if st.button("Piyasa Duyarlılığı Analizi", type="primary", key="market_sentiment"):
            try:
                # Log mesajlarını expander'a yönlendir
                with log_expander:
                    st.info("Piyasa Duyarlılığı Analizi başlatılıyor...")
                
                # Spinner yerine container kullan, böylece tüm içerik daha düzenli görünür
                result_container = st.container()
                
                # Minimal spinner
                with st.spinner(""):
                    # AI analizi - log mesajlarını expander'a yönlendir
                    sentiment_text, sentiment_data = ai_market_sentiment(gemini_pro, log_container=log_expander)
                    
                    # Temiz bir format içinde metni göster
                    with result_container:
                        st.markdown(f"""
                        <div style="background-color:#f0f7ff; padding:15px; border-radius:10px; border-left:5px solid #0066ff; margin-top:10px;">
                        {sentiment_text}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Eğer fallback sonuçlar varsa (API bağlantısı yoksa), görselleri göster
                        if sentiment_data:
                            st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)  # Daha az boşluk
                            
                            # Sütunları tanımla
                            col1, col2, col3, col4, col5 = st.columns(5)
                            
                            # Piyasa Duyarlılığı
                            with col1:
                                try:
                                    mood = sentiment_data.get('market_mood', 'Nötr')  # Varsayılan değer 'Nötr'
                                    mood_color = "green" if mood == "Olumlu" else ("red" if mood == "Olumsuz" else "orange")
                                    st.markdown(f"<h4 style='text-align:center; color:{mood_color}; font-size:1.1em;'>{mood}</h4>", unsafe_allow_html=True)
                                    st.markdown("<p style='text-align:center; font-size:0.9em;'>Piyasa Duyarlılığı</p>", unsafe_allow_html=True)
                                except Exception as e:
                                    st.markdown("<h4 style='text-align:center; color:orange; font-size:1.1em;'>Nötr</h4>", unsafe_allow_html=True)
                                    st.markdown("<p style='text-align:center; font-size:0.9em;'>Piyasa Duyarlılığı</p>", unsafe_allow_html=True)
                                    with log_expander:
                                        st.error(f"Piyasa duyarlılığı değerinde hata: {str(e)}")
                            
                            # Güven Oranı
                            with col2:
                                try:
                                    confidence = sentiment_data.get('confidence', 75)  # Varsayılan değer 75
                                    confidence_color = "green" if confidence > 75 else ("orange" if confidence > 50 else "red")
                                    st.markdown(f"<h4 style='text-align:center; color:{confidence_color}; font-size:1.1em;'>%{confidence}</h4>", unsafe_allow_html=True)
                                    st.markdown("<p style='text-align:center; font-size:0.9em;'>Güven Oranı</p>", unsafe_allow_html=True)
                                except Exception as e:
                                    st.markdown("<h4 style='text-align:center; color:orange; font-size:1.1em;'>%75</h4>", unsafe_allow_html=True)
                                    st.markdown("<p style='text-align:center; font-size:0.9em;'>Güven Oranı</p>", unsafe_allow_html=True)
                                    with log_expander:
                                        st.error(f"Güven oranı değerinde hata: {str(e)}")
                            
                            # Trend Gücü
                            with col3:
                                try:
                                    strength = sentiment_data.get('trend_strength', 50)  # Varsayılan değer 50
                                    strength_color = "green" if strength > 70 else ("orange" if strength > 40 else "red")
                                    st.markdown(f"<h4 style='text-align:center; color:{strength_color}; font-size:1.1em;'>%{strength}</h4>", unsafe_allow_html=True)
                                    st.markdown("<p style='text-align:center; font-size:0.9em;'>Trend Gücü</p>", unsafe_allow_html=True)
                                except Exception as e:
                                    st.markdown("<h4 style='text-align:center; color:orange; font-size:1.1em;'>%50</h4>", unsafe_allow_html=True)
                                    st.markdown("<p style='text-align:center; font-size:0.9em;'>Trend Gücü</p>", unsafe_allow_html=True)
                                    with log_expander:
                                        st.error(f"Trend gücü değerinde hata: {str(e)}")
                            
                            # Beklenen Volatilite
                            with col4:
                                try:
                                    volatility = sentiment_data.get('volatility_expectation', 'Orta')  # Varsayılan değer 'Orta'
                                    volatility_color = "green" if volatility == "Düşük" else ("red" if volatility == "Yüksek" else "orange")
                                    st.markdown(f"<h4 style='text-align:center; color:{volatility_color}; font-size:1.1em;'>{volatility}</h4>", unsafe_allow_html=True)
                                    st.markdown("<p style='text-align:center; font-size:0.9em;'>Beklenen Volatilite</p>", unsafe_allow_html=True)
                                except Exception as e:
                                    st.markdown("<h4 style='text-align:center; color:orange; font-size:1.1em;'>Orta</h4>", unsafe_allow_html=True)
                                    st.markdown("<p style='text-align:center; font-size:0.9em;'>Beklenen Volatilite</p>", unsafe_allow_html=True)
                                    with log_expander:
                                        st.error(f"Volatilite değerinde hata: {str(e)}")
                            
                            # Tavsiye
                            with col5:
                                try:
                                    recommendation = sentiment_data.get('overall_recommendation', 'Tut')  # Varsayılan değer 'Tut'
                                    rec_color = "green" if recommendation == "Al" else ("red" if recommendation == "Sat" else "orange")
                                    st.markdown(f"<h4 style='text-align:center; color:{rec_color}; font-size:1.1em;'>{recommendation}</h4>", unsafe_allow_html=True)
                                    st.markdown("<p style='text-align:center; font-size:0.9em;'>Tavsiye</p>", unsafe_allow_html=True)
                                except Exception as e:
                                    st.markdown("<h4 style='text-align:center; color:orange; font-size:1.1em;'>Tut</h4>", unsafe_allow_html=True)
                                    st.markdown("<p style='text-align:center; font-size:0.9em;'>Tavsiye</p>", unsafe_allow_html=True)
                                    with log_expander:
                                        st.error(f"Tavsiye değerinde hata: {str(e)}")
            except Exception as e:
                st.error(f"Piyasa duyarlılığı analizi sırasında bir hata oluştu: {str(e)}")
                with log_expander:
                    st.error(f"Hata detayı: {str(e)}")
    
    with ai_tabs[1]:
        st.subheader("Hisse Senedi Analizi")
        st.markdown("Seçtiğiniz hisse senedi için yapay zeka detaylı analiz yapar.")
        
        stock_symbol = st.text_input("Hisse Senedi Sembolü", value="THYAO.IS", key="ai_stock_symbol")
        stock_symbol = stock_symbol.upper()
        if not stock_symbol.endswith('.IS') and not stock_symbol == "":
            stock_symbol += '.IS'
        
        if st.button("Hisse Analizi", type="primary", key="stock_analysis"):
            results_container = st.container()
            
            # Log mesajlarını expander'a yönlendir
            with log_expander:
                st.info(f"{stock_symbol} için yapay zeka analizi yapılıyor...")
            
            with st.spinner(""):
                try:
                    # Hisse verilerini al
                    stock_data = get_stock_data(stock_symbol, period="6mo")
                    
                    with log_expander:
                        st.info(f"Hisse verileri alındı, analiz yapılıyor...")
                    
                    if stock_data is not None and not stock_data.empty:
                        # Göstergeleri hesapla
                        stock_data_with_indicators = calculate_indicators(stock_data)
                        
                        # Analizi çalıştır
                        with log_expander:
                            st.info("Göstergeler hesaplandı, YZ analizi başlatılıyor...")
                        
                        analysis_result = ai_stock_analysis(gemini_pro, stock_symbol, stock_data_with_indicators)
                        
                        # Sonuçları göster - sonuçları results_container içinde göster
                        with results_container:
                            st.markdown(f"""
                            <div style="background-color:#f0f7ff; padding:15px; border-radius:10px; border-left:5px solid #0066ff; margin-top:10px;">
                            {analysis_result}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Grafiği göster
                            fig = create_stock_chart(stock_data_with_indicators, stock_symbol)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Analiz sonuçlarını kaydet
                            company_info = get_company_info(stock_symbol)
                            last_price = stock_data['Close'].iloc[-1]
                            price_change = (stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]) / stock_data['Close'].iloc[-2] * 100
                            
                            # Basit trend tespiti
                            if stock_data['Close'].iloc[-1] > stock_data['SMA20'].iloc[-1]:
                                trend_direction = "Yükseliş"
                            else:
                                trend_direction = "Düşüş"
                                
                            # Risk seviyesi tespiti (basit)
                            volatility = stock_data['Close'].pct_change().std() * 100
                            if volatility > 3:
                                risk_level = "Yüksek"
                            elif volatility > 1.5:
                                risk_level = "Orta"
                            else:
                                risk_level = "Düşük"
                                
                            # Analiz sonuçlarını kaydet
                            ai_analysis_result = {
                                "symbol": stock_symbol,
                                "company_name": company_info.get("name", ""),
                                "last_price": last_price,
                                "price_change": price_change,
                                "recommendation": "AI Analiz",
                                "trend": trend_direction,
                                "risk_level": risk_level,
                                "analysis_summary": analysis_result,
                                "analysis_type": "ai"
                            }
                            
                            # Analiz sonuçlarını kaydet
                            save_analysis_result(stock_symbol, ai_analysis_result)
                            
                    else:
                        with results_container:
                            st.error(f"{stock_symbol} için veri alınamadı.")
                
                except Exception as e:
                    with results_container:
                        st.error(f"Hisse analizi sırasında bir hata oluştu: {str(e)}")
                    with log_expander:
                        st.error(f"Hata detayı: {str(e)}")
    
    with ai_tabs[2]:
        st.subheader("Fiyat Tahmini")
        st.markdown("Yapay zeka, seçtiğiniz hisse senedi için kısa ve orta vadeli fiyat tahminleri yapar.")
        
        price_symbol = st.text_input("Hisse Senedi Sembolü", value="THYAO.IS", key="ai_price_symbol")
        price_symbol = price_symbol.upper()
        if not price_symbol.endswith('.IS') and not price_symbol == "":
            price_symbol += '.IS'
        
        if st.button("Fiyat Tahmini", type="primary", key="price_prediction"):
            with st.spinner(f"{price_symbol} için fiyat tahmini yapılıyor..."):
                try:
                    # Hisse verilerini al
                    price_data = get_stock_data(price_symbol, period="6mo")
                    
                    if price_data is not None and not price_data.empty:
                        # Göstergeleri hesapla
                        price_data_with_indicators = calculate_indicators(price_data)
                        
                        # Tahmini çalıştır
                        prediction_result, prediction_data = ai_price_prediction(gemini_pro, price_symbol, price_data_with_indicators)
                        
                        # Sonuçları göster
                        st.markdown(f"""
                        <div style="background-color:#f0f7ff; padding:15px; border-radius:10px; border-left:5px solid #0066ff;">
                        {prediction_result}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Eğer fallback sonuçlar varsa (API bağlantısı yoksa), tahmin grafiği göster
                        if prediction_data:
                            st.subheader("Tahmin Grafiği")
                            
                            # Tahmin verilerini hazırla
                            current_price = prediction_data['current_price']
                            future_dates = []
                            future_prices = []
                            
                            # Gelecek 30 gün için tahmin
                            days = 30
                            target_price = prediction_data['predicted_price_30d']
                            
                            # Gelecek tarihleri oluştur
                            last_date = price_data.index[-1]
                            for i in range(1, days + 1):
                                if isinstance(last_date, pd.Timestamp):
                                    future_date = last_date + pd.Timedelta(days=i)
                                else:
                                    future_date = datetime.now() + timedelta(days=i)
                                future_dates.append(future_date)
                            
                            # Fiyat tahmini yap
                            for i in range(days):
                                progress = i / (days - 1)  # 0 to 1
                                # Basit doğrusal enterpolasyon
                                day_price = current_price + (target_price - current_price) * progress
                                
                                # Rastgele dalgalanmalar ekle
                                random_factor = np.random.uniform(-1, 1) * prediction_data['confidence'] / 500
                                day_price = day_price * (1 + random_factor)
                                
                                future_prices.append(day_price)
                            
                            # Tahmin grafiğini oluştur
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Geçmiş veri
                            ax.plot(price_data.index[-30:], price_data['Close'].iloc[-30:], label='Geçmiş Veri', color='blue')
                            
                            # Gelecek tahmin
                            ax.plot(future_dates, future_prices, label='YZ Tahmini', 
                                   color='green' if target_price > current_price else 'red', 
                                   linestyle='--')
                            
                            # Destek ve direnç çizgileri
                            ax.axhline(y=prediction_data['support_level'], color='green', linestyle=':', 
                                      label=f"Destek: {prediction_data['support_level']:.2f}")
                            ax.axhline(y=prediction_data['resistance_level'], color='red', linestyle=':', 
                                      label=f"Direnç: {prediction_data['resistance_level']:.2f}")
                            
                            ax.set_title(f"{price_symbol} Yapay Zeka Fiyat Tahmini")
                            ax.set_xlabel('Tarih')
                            ax.set_ylabel('Fiyat (TL)')
                            ax.grid(True, alpha=0.3)
                            ax.legend()
                            
                            # Grafiği göster
                            st.pyplot(fig)
                        
                        # Grafiği göster
                        fig = create_stock_chart(price_data_with_indicators, price_symbol)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"{price_symbol} için veri alınamadı.")
                
                except Exception as e:
                    st.error(f"Fiyat tahmini sırasında bir hata oluştu: {str(e)}")
    
    with ai_tabs[3]:
        st.subheader("Sektör Analizi")
        st.markdown("Seçtiğiniz hisse senedinin bulunduğu sektör için yapay zeka analizi yapar.")
        
        sector_symbol = st.text_input("Hisse Senedi Sembolü", value="THYAO.IS", key="ai_sector_symbol")
        sector_symbol = sector_symbol.upper()
        if not sector_symbol.endswith('.IS') and not sector_symbol == "":
            sector_symbol += '.IS'
        
        if st.button("Sektör Analizi", type="primary", key="sector_analysis"):
            with st.spinner(f"{sector_symbol} için sektör analizi yapılıyor..."):
                try:
                    # Analizi çalıştır
                    sector_result, sector_data = ai_sector_analysis(gemini_pro, sector_symbol)
                    
                    # Sonuçları göster
                    st.markdown(f"""
                    <div style="background-color:#f0f7ff; padding:15px; border-radius:10px; border-left:5px solid #0066ff;">
                    {sector_result}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Eğer fallback sonuçlar varsa (API bağlantısı yoksa), görselleri göster
                    if sector_data:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            outlook = sector_data['outlook']
                            outlook_color = "green" if outlook == "Olumlu" else ("red" if outlook == "Olumsuz" else "orange")
                            st.markdown(f"<h4 style='text-align: center; color: {outlook_color};'>{outlook}</h4>", unsafe_allow_html=True)
                            st.markdown("<p style='text-align: center;'>Sektör Görünümü</p>", unsafe_allow_html=True)
                        
                        with col2:
                            strength = sector_data['strength']
                            strength_color = "green" if strength > 70 else ("orange" if strength > 40 else "red")
                            st.markdown(f"<h4 style='text-align: center; color: {strength_color};'>%{strength}</h4>", unsafe_allow_html=True)
                            st.markdown("<p style='text-align: center;'>Sektör Gücü</p>", unsafe_allow_html=True)
                        
                        with col3:
                            trend = sector_data['trend']
                            trend_color = "green" if trend == "Yükseliş" else ("red" if trend == "Düşüş" else "orange")
                            st.markdown(f"<h4 style='text-align: center; color: {trend_color};'>{trend}</h4>", unsafe_allow_html=True)
                            st.markdown("<p style='text-align: center;'>Sektör Trendi</p>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Sektör analizi sırasında bir hata oluştu: {str(e)}")
    
    with ai_tabs[4]:
        st.subheader("Portföy Önerileri")
        st.markdown("Yapay zeka, yatırım bütçenize ve risk profilinize göre portföy önerisi oluşturur.")
        
        budget = st.number_input("Yatırım Bütçesi (TL)", min_value=1000, max_value=10000000, value=10000, step=1000)
        
        if st.button("Portföy Önerisi", type="primary", key="portfolio_recommendation"):
            with st.spinner("Portföy önerisi oluşturuluyor..."):
                try:
                    # Öneriyi çalıştır
                    portfolio_result = ai_portfolio_recommendation(gemini_pro, budget)
                    
                    # Sonuçları göster
                    st.markdown(f"""
                    <div style="background-color:#f0f7ff; padding:15px; border-radius:10px; border-left:5px solid #0066ff;">
                    {portfolio_result}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.warning("Bu öneri bir yatırım tavsiyesi niteliği taşımaz, sadece eğitim amaçlıdır.")
                    
                except Exception as e:
                    st.error(f"Portföy önerisi oluşturulurken bir hata oluştu: {str(e)}")
    
    with ai_tabs[5]:
        st.subheader("Teknik Analiz Yorumlama")
        st.markdown("Yapay zeka, teknik göstergeleri yorumlayarak alım/satım sinyallerini değerlendirir.")
        
        ta_symbol = st.text_input("Hisse Senedi Sembolü", value="THYAO.IS", key="ai_ta_symbol")
        ta_symbol = ta_symbol.upper()
        if not ta_symbol.endswith('.IS') and not ta_symbol == "":
            ta_symbol += '.IS'
        
        if st.button("Teknik Analiz", type="primary", key="technical_analysis"):
            with st.spinner(f"{ta_symbol} için teknik analiz yapılıyor..."):
                try:
                    # Hisse verilerini al
                    ta_data = get_stock_data(ta_symbol, period="6mo")
                    
                    if ta_data is not None and not ta_data.empty:
                        # Göstergeleri hesapla
                        ta_data_with_indicators = calculate_indicators(ta_data)
                        
                        # Analizi çalıştır
                        interpretation = ai_technical_interpretation(gemini_pro, ta_symbol, ta_data_with_indicators)
                        
                        # Sonuçları göster
                        st.markdown(f"""
                        <div style="background-color:#f0f7ff; padding:15px; border-radius:10px; border-left:5px solid #0066ff;">
                        {interpretation}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Grafiği göster
                        fig = create_stock_chart(ta_data_with_indicators, ta_symbol)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"{ta_symbol} için veri alınamadı.")
                
                except Exception as e:
                    st.error(f"Teknik analiz sırasında bir hata oluştu: {str(e)}")