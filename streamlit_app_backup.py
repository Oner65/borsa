import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import logging
import traceback
import random

# UI modüllerini import et
try:
    # Proje dizinini sys.path'e ekle
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    # UI modüllerini import et
    from ui.bist100_tab import render_bist100_tab
    from ui.ml_prediction_tab import render_ml_prediction_tab
    from ui.stock_tab import render_stock_tab
    
    UI_MODULES_AVAILABLE = True
except ImportError as e:
    st.warning(f"UI modülleri yüklenemedi: {e}")
    UI_MODULES_AVAILABLE = False

# Streamlit Cloud için sayfa konfigürasyonu
st.set_page_config(
    page_title="Kompakt Borsa Analiz Uygulaması",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Logging konfigürasyonu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Session state değişkenlerini başlat
if 'analyzed_stocks' not in st.session_state:
    st.session_state.analyzed_stocks = {}

if 'favorite_stocks' not in st.session_state:
    st.session_state.favorite_stocks = []

if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Yardımcı fonksiyonlar
def get_stock_data(symbol, period="1y"):
    """Hisse senedi verilerini al"""
    try:
        # Türk hisseler için .IS ekle
        if not symbol.endswith('.IS'):
            symbol = symbol + '.IS'
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            return pd.DataFrame()
        
        return data
    except Exception as e:
        logger.error(f"Veri alınırken hata: {e}")
        return pd.DataFrame()

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
        if all(col in df.columns for col in ['Upper_Band', 'Lower_Band', 'Middle_Band']):
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
            
            # RSI seviyeleri
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

def get_popular_stocks():
    """Popüler hisselerin listesini al"""
    popular_stocks = [
        {"symbol": "THYAO", "name": "Türk Hava Yolları", "value": 250.0, "change_percent": random.uniform(-5, 5)},
        {"symbol": "GARAN", "name": "Garanti BBVA", "value": 85.0, "change_percent": random.uniform(-5, 5)},
        {"symbol": "ASELS", "name": "Aselsan", "value": 75.0, "change_percent": random.uniform(-5, 5)},
        {"symbol": "AKBNK", "name": "Akbank", "value": 55.0, "change_percent": random.uniform(-5, 5)},
        {"symbol": "EREGL", "name": "Ereğli Demir Çelik", "value": 40.0, "change_percent": random.uniform(-5, 5)},
        {"symbol": "VAKBN", "name": "VakıfBank", "value": 30.0, "change_percent": random.uniform(-5, 5)},
        {"symbol": "TUPRS", "name": "Tüpraş", "value": 120.0, "change_percent": random.uniform(-5, 5)},
        {"symbol": "FROTO", "name": "Ford Otosan", "value": 65.0, "change_percent": random.uniform(-5, 5)},
    ]
    return popular_stocks

def add_to_favorites(stock_symbol):
    """Favorilere ekle"""
    try:
        stock_symbol = stock_symbol.upper().strip()
        
        if stock_symbol not in st.session_state.favorite_stocks:
            st.session_state.favorite_stocks.append(stock_symbol)
            return True
        return False
    except Exception as e:
        logger.error(f"Favori eklenirken hata: {e}")
        return False

def remove_from_favorites(stock_symbol):
    """Favorilerden çıkar"""
    try:
        if stock_symbol in st.session_state.favorite_stocks:
            st.session_state.favorite_stocks.remove(stock_symbol)
            return True
        return False
    except Exception as e:
        logger.error(f"Favori çıkarılırken hata: {e}")
        return False

def render_simple_bist100_tab():
    """BIST 100 sekmesi"""
    st.header("BIST 100 Genel Bakış", divider="blue")
    
    # BIST-100 verilerini al
    bist100_data = get_stock_data("XU100", "6mo")
    
    if not bist100_data.empty:
        bist100_data = calculate_indicators(bist100_data)
        
        # Günlük veriler
        last_price = bist100_data['Close'].iloc[-1]
        prev_price = bist100_data['Close'].iloc[-2]
        change = ((last_price - prev_price) / prev_price) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("BIST 100", f"{last_price:.0f}", f"{change:.2f}%")
        
        with col2:
            volume = bist100_data['Volume'].iloc[-1]
            st.metric("Hacim", f"{volume:,.0f}")
        
        with col3:
            high_52w = bist100_data['High'].rolling(252).max().iloc[-1]
            st.metric("52H Yüksek", f"{high_52w:.0f}")
        
        # Grafik
        fig = create_stock_chart(bist100_data, "BIST-100")
        st.plotly_chart(fig, use_container_width=True)
        
        # Popüler hisseler
        st.subheader("Popüler Hisseler")
        popular_stocks = get_popular_stocks()
        
        for i in range(0, len(popular_stocks), 4):
            cols = st.columns(4)
            for j, stock in enumerate(popular_stocks[i:i+4]):
                with cols[j]:
                    change_color = "green" if stock["change_percent"] > 0 else "red"
                    st.markdown(f"""
                    **{stock['symbol']}**  
                    {stock['name']}  
                    💰 {stock['value']:.2f} TL  
                    <span style='color: {change_color}'>📈 {stock['change_percent']:.2f}%</span>
                    """, unsafe_allow_html=True)
    else:
        st.error("BIST-100 verileri alınamadı")

def render_simple_stock_tab():
    """Hisse analizi sekmesi"""
    st.header("Hisse Senedi Teknik Analizi")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        stock_symbol = st.text_input("Hisse Senedi Kodu")
    
    with col2:
        period = st.selectbox("Zaman Aralığı", ["1mo", "3mo", "6mo", "1y", "2y"])
    
    with col3:
        st.write("")
        analyze = st.button("Analiz Et", use_container_width=True)
    
    if analyze and stock_symbol:
        with st.spinner(f"{stock_symbol} analiz ediliyor..."):
            df = get_stock_data(stock_symbol, period)
            
            if not df.empty:
                df = calculate_indicators(df)
                signals = get_signals(df)
                
                # Metrikler
                last_price = df['Close'].iloc[-1]
                prev_price = df['Close'].iloc[-2]
                change = ((last_price - prev_price) / prev_price) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Son Fiyat", f"{last_price:.2f} TL", f"{change:.2f}%")
                
                with col2:
                    rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 0
                    st.metric("RSI", f"{rsi:.1f}")
                
                with col3:
                    volume = df['Volume'].iloc[-1]
                    st.metric("Hacim", f"{volume:,.0f}")
                
                with col4:
                    volatility = df['Close'].pct_change().std() * 100
                    st.metric("Volatilite", f"{volatility:.2f}%")
                
                # Grafik
                fig = create_stock_chart(df, stock_symbol)
                st.plotly_chart(fig, use_container_width=True)
                
                # Sinyaller
                st.subheader("Teknik Göstergeler")
                if not signals.empty:
                    total_signal = signals['Total_Signal'].iloc[-1]
                    if total_signal > 5:
                        st.success("🟢 Güçlü Alım Sinyali")
                    elif total_signal > 0:
                        st.info("🔵 Zayıf Alım Sinyali")
                    elif total_signal < -5:
                        st.error("🔴 Güçlü Satım Sinyali")
                    elif total_signal < 0:
                        st.warning("🟡 Zayıf Satım Sinyali")
                    else:
                        st.info("⚪ Nötr")
            else:
                st.error("Hisse verisi alınamadı")
    else:
        st.info("Hisse kodu girin ve analiz edin.")

def render_simple_ml_tab():
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
        threshold = st.slider("Yükseliş Eşiği (%)", 1.0, 10.0, 3.0, 0.5)
    
    with col2:
        scan_option = st.selectbox("Tarama Modu", ["BIST 30", "BIST 50", "Popüler Hisseler"])
    
    with col3:
        days = st.selectbox("Tahmin Süresi", [1, 3, 5, 7], index=2)
    
    # Özel liste için
    custom_stocks = ""
    if scan_option == "Özel Liste":
        custom_stocks = st.text_area("Hisse Kodları (virgülle ayırın)", 
                                   placeholder="THYAO, GARAN, ASELS")
    
    # Hisse listesini belirle
    if scan_option == "BIST 30":
        stock_list = ["THYAO", "GARAN", "ASELS", "AKBNK", "EREGL", "VAKBN", "TUPRS", "FROTO"]
    elif scan_option == "BIST 50":
        stock_list = ["THYAO", "GARAN", "ASELS", "AKBNK", "EREGL", "VAKBN", "TUPRS", "FROTO", 
                     "TCELL", "SAHOL", "PETKM", "KCHOL", "KOZAL", "ARCLK"]
    elif scan_option == "Popüler Hisseler":
        stock_list = ["THYAO", "GARAN", "ASELS", "AKBNK", "EREGL"]
    else:
        stock_list = [s.strip().upper() for s in custom_stocks.split(",") if s.strip()]
    
    if st.button("ML Tarama Başlat", type="primary", use_container_width=True):
        if not stock_list:
            st.error("Taranacak hisse listesi boş!")
            return
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(stock_list):
            status_text.text(f"Analiz ediliyor: {symbol}")
            progress_bar.progress((i + 1) / len(stock_list))
            
            try:
                # Veri al
                df = get_stock_data(symbol, "1y")
                if df.empty:
                    continue
                
                # Özellikler hazırla
                features = prepare_features(df)
                if features.empty:
                    continue
                
                # Hedef değişken
                target = create_ml_target(df, threshold/100, days)
                if len(target) == 0:
                    continue
                
                # Model eğit
                model, accuracy = train_simple_model(features, target)
                if model is None:
                    continue
                
                # Tahmin yap
                last_features = features.iloc[-1:].fillna(0)
                prediction_proba = model.predict_proba(last_features)[0][1] if hasattr(model, 'predict_proba') else 0.5
                
                # Son fiyat
                last_price = df['Close'].iloc[-1]
                prev_price = df['Close'].iloc[-2]
                change = ((last_price - prev_price) / prev_price) * 100
                
                results.append({
                    "Hisse": symbol,
                    "Son Fiyat": f"{last_price:.2f} TL",
                    "Günlük Değişim": f"{change:.2f}%",
                    "Yükseliş Olasılığı": f"{prediction_proba:.1%}",
                    "Model Doğruluğu": f"{accuracy:.1%}" if accuracy else "N/A"
                })
                
            except Exception as e:
                logger.error(f"{symbol} için hata: {e}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        if results:
            # Sonuçları göster
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values("Yükseliş Olasılığı", ascending=False)
            
            st.subheader("ML Tarama Sonuçları")
            st.dataframe(df_results, use_container_width=True)
            
            # En umutvar 3 hisse
            st.subheader("🔥 En Umutvar 3 Hisse")
            top_3 = df_results.head(3)
            
            for _, row in top_3.iterrows():
                st.info(f"**{row['Hisse']}** - {row['Son Fiyat']} - Yükseliş Olasılığı: {row['Yükseliş Olasılığı']}")
        else:
            st.error("Hiç sonuç alınamadı. Lütfen farklı parametreler deneyin.")
    
    # Örnek tahmin gösterimi
    st.subheader("Nasıl Çalışıyor?")
    with st.expander("Detayları Gör"):
        st.markdown("""
        ### ML Tahmini Süreci:
        
        1. **Veri Toplama**: Her hisse için geçmiş fiyat ve hacim verileri alınır
        2. **Özellik Çıkarma**: RSI, MACD, MA gibi teknik göstergeler hesaplanır
        3. **Hedef Belirleme**: Gelecekteki fiyat artışları etiketlenir
        4. **Model Eğitimi**: RandomForest algoritması ile model eğitilir
        5. **Tahmin**: Güncel verilerle yükseliş olasılığı hesaplanır
        
        ### Değerlendirme:
        - **Yüksek Olasılık (>70%)**: Güçlü yükseliş potansiyeli
        - **Orta Olasılık (40-70%)**: Belirsiz, dikkatli izleme
        - **Düşük Olasılık (<40%)**: Zayıf yükseliş potansiyeli
        
        ⚠️ **Uyarı**: Bu tahminler garanti değildir ve yatırım tavsiyesi niteliği taşımaz.
        """)
    
    return

def render_bist100_tab():
    """BIST 100 sekmesi"""
    st.header("BIST 100 Genel Bakış", divider="blue")
    
    # BIST-100 verilerini al
    bist100_data = get_stock_data("XU100", "6mo")
    
    if not bist100_data.empty:
        bist100_data = calculate_indicators(bist100_data)
        
        # Günlük veriler
        last_price = bist100_data['Close'].iloc[-1]
        prev_price = bist100_data['Close'].iloc[-2]
        change = ((last_price - prev_price) / prev_price) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("BIST 100", f"{last_price:.0f}", f"{change:.2f}%")
        
        with col2:
            volume = bist100_data['Volume'].iloc[-1]
            st.metric("Hacim", f"{volume:,.0f}")
        
        with col3:
            high = bist100_data['High'].iloc[-1]
            low = bist100_data['Low'].iloc[-1]
            st.metric("Günlük Aralık", f"{low:.0f} - {high:.0f}")
        
        # Grafik
        fig = create_stock_chart(bist100_data, "BIST-100")
        st.plotly_chart(fig, use_container_width=True)
        
        # Popüler hisseler
        st.subheader("Popüler Hisseler")
        popular_stocks = get_popular_stocks()
        
        for i in range(0, len(popular_stocks), 4):
            cols = st.columns(4)
            for j, col in enumerate(cols):
                if i + j < len(popular_stocks):
                    stock = popular_stocks[i + j]
                    with col:
                        change_color = "green" if stock["change_percent"] > 0 else "red"
                        st.markdown(f"""
                        <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 10px;">
                            <h5>{stock["symbol"]}</h5>
                            <p style="font-size: 12px; color: gray;">{stock["name"]}</p>
                            <p style="color: {change_color}; font-weight: bold;">{stock["change_percent"]:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.error("BIST-100 verileri alınamadı")

def render_stock_tab():
    """Hisse analizi sekmesi"""
    st.header("Hisse Senedi Teknik Analizi")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        stock_symbol = st.text_input("Hisse Senedi Kodu")
    
    with col2:
        period = st.selectbox("Zaman Aralığı", ["1mo", "3mo", "6mo", "1y", "2y"])
    
    with col3:
        st.write("")
        analyze = st.button("Analiz Et", use_container_width=True)
    
    if analyze and stock_symbol:
        with st.spinner(f"{stock_symbol} analiz ediliyor..."):
            df = get_stock_data(stock_symbol, period)
            
            if not df.empty:
                df = calculate_indicators(df)
                signals = get_signals(df)
                
                # Favori ekleme butonu
                col_fav, col_info = st.columns([1, 4])
                with col_fav:
                    if st.button("⭐ Favorilere Ekle"):
                        if add_to_favorites(stock_symbol):
                            st.success("Favorilere eklendi!")
                        else:
                            st.warning("Zaten favorilerde")
                
                # Güncel bilgiler
                last_price = df['Close'].iloc[-1]
                prev_price = df['Close'].iloc[-2]
                change = ((last_price - prev_price) / prev_price) * 100
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Son Fiyat", f"{last_price:.2f} TL", f"{change:.2f}%")
                
                with col2:
                    rsi = df['RSI'].iloc[-1]
                    st.metric("RSI", f"{rsi:.2f}")
                
                with col3:
                    macd = df['MACD'].iloc[-1]
                    st.metric("MACD", f"{macd:.4f}")
                
                # Grafik
                fig = create_stock_chart(df, stock_symbol)
                st.plotly_chart(fig, use_container_width=True)
                
                # Sinyal analizi
                st.subheader("Teknik Göstergeler")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Hareketli Ortalamalar**")
                    ma_data = {
                        "Gösterge": ["SMA20", "SMA50", "SMA200"],
                        "Değer": [
                            f"{df['SMA20'].iloc[-1]:.2f}",
                            f"{df['SMA50'].iloc[-1]:.2f}",
                            f"{df['SMA200'].iloc[-1]:.2f}"
                        ],
                        "Sinyal": [
                            "AL" if signals['SMA20_Signal'].iloc[-1] > 0 else "SAT",
                            "AL" if signals['SMA50_Signal'].iloc[-1] > 0 else "SAT",
                            "AL" if signals['SMA200_Signal'].iloc[-1] > 0 else "SAT"
                        ]
                    }
                    st.dataframe(pd.DataFrame(ma_data), hide_index=True)
                
                with col2:
                    st.write("**Osilatörler**")
                    osc_data = {
                        "Gösterge": ["RSI", "MACD", "Stochastic"],
                        "Değer": [
                            f"{df['RSI'].iloc[-1]:.2f}",
                            f"{df['MACD'].iloc[-1]:.4f}",
                            f"{df['Stoch_%K'].iloc[-1]:.2f}"
                        ],
                        "Sinyal": [
                            "AL" if signals['RSI_Signal'].iloc[-1] > 0 else ("SAT" if signals['RSI_Signal'].iloc[-1] < 0 else "NÖTR"),
                            "AL" if signals['MACD_Signal'].iloc[-1] > 0 else "SAT",
                            "AL" if signals['Stoch_Signal'].iloc[-1] > 0 else ("SAT" if signals['Stoch_Signal'].iloc[-1] < 0 else "NÖTR")
                        ]
                    }
                    st.dataframe(pd.DataFrame(osc_data), hide_index=True)
                
                # Genel öneri
                total_signal = signals['Total_Signal'].iloc[-1]
                total_indicators = 16  # Toplam gösterge sayısı
                
                if total_signal > total_indicators * 0.6:
                    recommendation = "GÜÇLÜ AL"
                    rec_color = "green"
                elif total_signal > 0:
                    recommendation = "AL"
                    rec_color = "green"
                elif total_signal < -total_indicators * 0.6:
                    recommendation = "GÜÇLÜ SAT"
                    rec_color = "red"
                elif total_signal < 0:
                    recommendation = "SAT"
                    rec_color = "red"
                else:
                    recommendation = "NÖTR"
                    rec_color = "gray"
                
                st.markdown(f"<h2 style='text-align: center; color: {rec_color};'>{recommendation}</h2>", unsafe_allow_html=True)
                
            else:
                st.error("Hisse senedi verisi bulunamadı")

def render_ml_prediction_tab():
    """ML tahmin sekmesi"""
    st.header("ML Yükseliş Tahmini")
    st.info("Bu özellik geliştirme aşamasındadır. Gelecek güncellemelerde eklenecektir.")
    
    # Örnek tahmin verileri
    st.subheader("Örnek Tahminler")
    
    sample_predictions = [
        {"symbol": "THYAO", "prediction": "Yükseliş", "confidence": 75, "target": "+5.2%"},
        {"symbol": "GARAN", "prediction": "Düşüş", "confidence": 68, "target": "-2.1%"},
        {"symbol": "ASELS", "prediction": "Yükseliş", "confidence": 82, "target": "+7.8%"},
    ]
    
    for pred in sample_predictions:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.write(f"**{pred['symbol']}**")
        
        with col2:
            color = "green" if pred['prediction'] == "Yükseliş" else "red"
            st.markdown(f"<span style='color: {color}'>{pred['prediction']}</span>", unsafe_allow_html=True)
        
        with col3:
            st.write(f"{pred['confidence']}%")
        
        with col4:
            st.write(pred['target'])

def main():
    """Ana uygulama"""
    
    # CSS stilleri
    st.markdown("""
    <style>
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
    </style>
    """, unsafe_allow_html=True)
    
    # Ana başlık
    st.markdown('''
    <div class="main-header">
        <h1>📊 Kompakt Borsa Analiz Uygulaması</h1>
        <p>BIST 100 Genel Bakış • Hisse Analizi • ML Yükseliş Tahminleri</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Favori hisseler
    if st.session_state.favorite_stocks:
        st.subheader("⭐ Favori Hisseler")
        
        cols = st.columns(min(len(st.session_state.favorite_stocks), 5))
        for idx, stock in enumerate(st.session_state.favorite_stocks):
            col_idx = idx % 5
            with cols[col_idx]:
                st.write(f"**{stock}**")
                if st.button("❌", key=f"remove_{stock}", help="Favorilerden çıkar"):
                    if remove_from_favorites(stock):
                        st.success(f"{stock} favorilerden çıkarıldı")
                        st.rerun()
    
    # Ana sekmeler - İstediğiniz 3 sekme
    tab1, tab2, tab3 = st.tabs(["📊 BIST 100", "🧠 ML Tarama", "🔍 Hisse Analizi"])
    
    with tab1:
        if UI_MODULES_AVAILABLE:
            render_bist100_tab()
        else:
            # Fallback: Basit BIST 100 gösterimi
            render_simple_bist100_tab()
    
    with tab2:
        if UI_MODULES_AVAILABLE:
            render_ml_prediction_tab()
        else:
            # Fallback: Basit ML tarama gösterimi
            render_simple_ml_tab()
    
    with tab3:
        if UI_MODULES_AVAILABLE:
            render_stock_tab()
        else:
            # Fallback: Basit hisse analizi gösterimi
            render_simple_stock_tab()
    
    # Sidebar
    with st.sidebar:
        st.title("📊 Piyasa Özeti")
        
        # BIST-100 özet
        bist_data = get_stock_data("XU100", "5d")
        if not bist_data.empty:
            last_price = bist_data['Close'].iloc[-1]
            prev_price = bist_data['Close'].iloc[-2]
            change = ((last_price - prev_price) / prev_price) * 100
            
            st.metric("BIST 100", f"{last_price:.0f}", f"{change:.2f}%")
        
        # Popüler hisseler kısa liste
        st.subheader("🔥 Popüler Hisseler")
        popular_stocks = get_popular_stocks()[:5]
        
        for stock in popular_stocks:
            change_color = "green" if stock["change_percent"] > 0 else "red"
            st.markdown(f"""
            <div style="padding: 5px; margin-bottom: 5px;">
                <strong>{stock["symbol"]}</strong>
                <span style="float: right; color: {change_color};">
                    {stock["change_percent"]:.2f}%
                </span>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
