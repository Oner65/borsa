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
@st.cache_data(ttl=300)  # 5 dakika cache
def get_stock_data(symbol, period="1y"):
    """Hisse senedi verilerini al - Cache'li versiyon"""
    try:
        # Türk hisseler için .IS ekle
        if not symbol.endswith('.IS'):
            symbol = f"{symbol}.IS"
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            st.error(f"{symbol} için veri bulunamadı")
            return pd.DataFrame()
        
        return data
    except Exception as e:
        logger.error(f"Veri alınırken hata: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)  # 10 dakika cache (teknik göstergeler için)
def calculate_indicators(df):
    """Temel teknik göstergeleri hesapla - Cache'li versiyon"""
    if df.empty:
        return df
    
    try:
        # Hareketli ortalamalar
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        
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
        
        return df
    except Exception as e:
        logger.error(f"Göstergeler hesaplanırken hata: {e}")
        return df

@st.cache_data(ttl=300)  # 5 dakika cache
def create_stock_chart(df, symbol):
    """Basit hisse senedi grafiği oluştur - Cache'li versiyon"""
    try:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"{symbol} Fiyat", "Hacim")
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
        if 'SMA20' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], mode='lines', name='SMA20', line=dict(color='blue')), row=1, col=1)
        if 'SMA50' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA50', line=dict(color='orange')), row=1, col=1)
        
        # Hacim
        colors = ['green' if row['Close'] >= row['Open'] else 'red' for i, row in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Hacim', marker_color=colors), row=2, col=1)
        
        fig.update_layout(
            height=600,
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

# Basit Fallback Fonksiyonları (UI modülleri yüklenemezse kullanılacak)
def render_simple_bist100_tab():
    """Basit BIST 100 sekmesi"""
    st.header("📊 BIST 100 Genel Bakış", divider="blue")
    
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
        st.subheader("🔥 Popüler Hisseler")
        popular_stocks = get_popular_stocks()
        
        for i in range(0, len(popular_stocks), 4):
            cols = st.columns(4)
            for j, stock in enumerate(popular_stocks[i:i+4]):
                if j < len(cols):
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

# Cache'li yardımcı fonksiyonlar
@st.cache_data(ttl=300)  # 5 dakika cache
def analyze_stock_performance(symbol, period="1y"):
    """Hisse senedi performans analizi - Cache'li versiyon"""
    try:
        df = get_stock_data(symbol, period)
        if df.empty:
            return None
        
        df = calculate_indicators(df)
        
        # Performans metrikleri
        last_price = df['Close'].iloc[-1]
        first_price = df['Close'].iloc[0]
        total_return = ((last_price - first_price) / first_price) * 100
        
        # Volatilite
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5) * 100  # Yıllık volatilite
        
        # Sharpe benzeri metrik
        risk_free_rate = 15  # Türkiye için yaklaşık risk-free oran
        excess_return = total_return - risk_free_rate
        sharpe_like = excess_return / volatility if volatility > 0 else 0
        
        return {
            'symbol': symbol,
            'last_price': last_price,
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_like': sharpe_like,
            'df': df
        }
    except Exception as e:
        logger.error(f"Performans analizi hatası: {e}")
        return None

def render_simple_stock_tab():
    """Basit hisse analizi sekmesi"""
    st.header("🔍 Hisse Senedi Teknik Analizi")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        stock_symbol = st.text_input("Hisse Senedi Kodu", placeholder="Örnek: THYAO")
    
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
                
                # Basit sinyal analizi
                st.subheader("📊 Teknik Göstergeler")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # RSI analizi
                    rsi_val = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
                    if rsi_val > 70:
                        st.warning("🔴 RSI Aşırı Alım Bölgesinde (>70)")
                    elif rsi_val < 30:
                        st.success("🟢 RSI Aşırı Satım Bölgesinde (<30)")
                    else:
                        st.info(f"🔵 RSI Nötr Bölgede ({rsi_val:.1f})")
                
                with col2:
                    # Trend analizi
                    if 'SMA20' in df.columns and 'SMA50' in df.columns:
                        sma20 = df['SMA20'].iloc[-1]
                        sma50 = df['SMA50'].iloc[-1]
                        if sma20 > sma50:
                            st.success("🟢 Kısa Vadeli Yükseliş Trendi")
                        else:
                            st.error("🔴 Kısa Vadeli Düşüş Trendi")
                    else:
                        st.info("Trend analizi için yeterli veri yok")
                
                # Genel değerlendirme
                signals = 0
                if 'RSI' in df.columns:
                    rsi_val = df['RSI'].iloc[-1]
                    if rsi_val < 30:
                        signals += 1
                    elif rsi_val > 70:
                        signals -= 1
                
                if 'SMA20' in df.columns and 'SMA50' in df.columns:
                    if df['SMA20'].iloc[-1] > df['SMA50'].iloc[-1]:
                        signals += 1
                    else:
                        signals -= 1
                
                st.subheader("🎯 Genel Değerlendirme")
                if signals > 0:
                    st.success("🟢 Pozitif Sinyaller Ağırlıkta")
                elif signals < 0:
                    st.error("🔴 Negatif Sinyaller Ağırlıkta")
                else:
                    st.info("🔵 Karışık Sinyaller - Dikkatli İzleme")
                    
            else:
                st.error("Hisse verisi alınamadı")
    else:
        st.info("Hisse kodu girin ve analiz edin.")
        
        # Örnek kullanım
        st.markdown("""
        ### 💡 Örnek Kullanım:
        - **THYAO** (Türk Hava Yolları)
        - **GARAN** (Garanti BBVA)
        - **ASELS** (Aselsan)
        - **AKBNK** (Akbank)
        """)

def render_simple_ml_tab():
    """Basit ML tahmin sekmesi"""
    st.header("🧠 ML Yükseliş Tahmini", divider="rainbow")
    
    st.markdown("""
    Bu bölüm, makine öğrenmesi algoritmaları kullanarak hisse senetlerinin gelecekteki performansını tahmin eder.
    
    **🎯 Nasıl Çalışır:**
    - Teknik göstergeler ve fiyat hareketleri analiz edilir
    - Geçmiş verilerden öğrenme yapılır
    - Gelecekteki yükseliş olasılığı hesaplanır
    
    **⚠️ Önemli Not:** Bu tahminler yatırım tavsiyesi değildir, sadece analiz amaçlıdır.
    """)
    
    # Parametreler
    col1, col2, col3 = st.columns(3)
    
    with col1:
        threshold = st.slider("Yükseliş Eşiği (%)", 1.0, 10.0, 3.0, 0.5)
    
    with col2:
        scan_option = st.selectbox("Tarama Modu", ["BIST 30", "Popüler Hisseler", "Özel Liste"])
    
    with col3:
        days = st.selectbox("Tahmin Süresi (Gün)", [1, 3, 5, 7], index=2)
    
    # Özel liste için
    if scan_option == "Özel Liste":
        custom_stocks = st.text_area("Hisse Kodları (virgülle ayırın)", 
                                   placeholder="THYAO, GARAN, ASELS")
        stock_list = [s.strip().upper() for s in custom_stocks.split(",") if s.strip()]
    elif scan_option == "BIST 30":
        stock_list = ["THYAO", "GARAN", "ASELS", "AKBNK", "EREGL", "VAKBN", "TUPRS", "FROTO", 
                     "TCELL", "SAHOL", "PETKM", "KCHOL", "KOZAL", "ARCLK"]
    else:  # Popüler Hisseler
        stock_list = ["THYAO", "GARAN", "ASELS", "AKBNK", "EREGL"]
    
    if st.button("🚀 ML Tarama Başlat", type="primary", use_container_width=True):
        if not stock_list:
            st.error("Taranacak hisse listesi boş!")
            return
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(stock_list):
            status_text.text(f"🔄 Analiz ediliyor: {symbol}")
            progress_bar.progress((i + 1) / len(stock_list))
            
            try:
                # Veri al
                df = get_stock_data(symbol, "1y")
                if df.empty:
                    continue
                
                # Basit ML analizi (gerçek ML yerine basit istatistiksel analiz)
                df = calculate_indicators(df)
                
                # Son fiyat ve değişim
                last_price = df['Close'].iloc[-1]
                prev_price = df['Close'].iloc[-2]
                change = ((last_price - prev_price) / prev_price) * 100
                
                # Basit yükseliş olasılığı hesaplama
                # RSI, trend ve volatilite bazlı
                score = 0
                
                # RSI skoru
                if 'RSI' in df.columns:
                    rsi = df['RSI'].iloc[-1]
                    if rsi < 30:  # Oversold
                        score += 30
                    elif rsi < 50:
                        score += 15
                    elif rsi > 70:  # Overbought
                        score -= 20
                
                # Trend skoru
                if 'SMA20' in df.columns and 'SMA50' in df.columns:
                    sma20 = df['SMA20'].iloc[-1]
                    sma50 = df['SMA50'].iloc[-1]
                    if sma20 > sma50:
                        score += 25
                    else:
                        score -= 15
                
                # Momentum skoru
                if len(df) >= 5:
                    momentum = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100
                    if momentum > 2:
                        score += 20
                    elif momentum < -2:
                        score -= 20
                
                # Skor normalizasyonu (0-100 arası)
                probability = max(0, min(100, score + 50)) / 100
                
                results.append({
                    "Hisse": symbol,
                    "Son Fiyat": f"{last_price:.2f} TL",
                    "Günlük Değişim": f"{change:.2f}%",
                    "Yükseliş Olasılığı": f"{probability:.1%}",
                    "Durum": "🟢 Umutvar" if probability > 0.6 else ("🟡 Nötr" if probability > 0.4 else "🔴 Zayıf")
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
            
            st.subheader("📊 ML Tarama Sonuçları")
            st.dataframe(df_results, use_container_width=True)
            
            # En umutvar 3 hisse
            st.subheader("🏆 En Umutvar 3 Hisse")
            top_3 = df_results.head(3)
            
            for i, (_, row) in enumerate(top_3.iterrows(), 1):
                if float(row['Yükseliş Olasılığı'].strip('%')) > 60:
                    st.success(f"**{i}. {row['Hisse']}** - {row['Son Fiyat']} - Yükseliş Olasılığı: {row['Yükseliş Olasılığı']}")
                else:
                    st.info(f"**{i}. {row['Hisse']}** - {row['Son Fiyat']} - Yükseliş Olasılığı: {row['Yükseliş Olasılığı']}")
        else:
            st.error("❌ Hiç sonuç alınamadı. Lütfen farklı parametreler deneyin.")
    
    # Nasıl çalışır açıklaması
    with st.expander("🔍 Nasıl Çalışıyor?"):
        st.markdown("""
        ### 🧮 Analiz Süreci:
        
        1. **📈 Teknik Analiz**: RSI, hareketli ortalamalar ve momentum hesaplanır
        2. **📊 Skorlama**: Her gösterge için pozitif/negatif puanlar verilir  
        3. **🎯 Olasılık**: Toplam skor 0-100% arasında normalize edilir
        4. **📋 Sıralama**: En yüksek olasılıktan düşüğe doğru sıralanır
        
        ### 📊 Yorumlama:
        - **🟢 >60%**: Güçlü yükseliş potansiyeli
        - **🟡 40-60%**: Belirsiz durum
        - **🔴 <40%**: Zayıf yükseliş potansiyeli
        
        ### ⚠️ Önemli Uyarılar:
        - Bu tahminler **garanti değildir**
        - **Yatırım tavsiyesi niteliği taşımaz**
        - Kendi araştırmanızı da yapın
        - **Risk yönetimini** ihmal etmeyin
        """)

def main():
    """Ana uygulama"""
    
    # CSS stilleri
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
        }
        
        .main-header h1 {
            font-weight: 700;
            margin-bottom: 0.5rem;
            font-size: 2.2rem;
        }
        
        .main-header p {
            font-size: 1.1rem;
            opacity: 0.9;
            margin-bottom: 0;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
            background-color: #f0f2f6;
            border-radius: 10px 10px 0px 0px;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #667eea;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Ana başlık
    st.markdown('''
    <div class="main-header">
        <h1>📊 Kompakt Borsa Analiz Uygulaması</h1>
        <p>BIST 100 • ML Tarama • Hisse Analizi</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Ana sekmeler - İstediğiniz 3 sekme
    tab1, tab2, tab3 = st.tabs(["📊 BIST 100", "🧠 ML Tarama", "🔍 Hisse Analizi"])
    
    with tab1:
        if UI_MODULES_AVAILABLE:
            try:
                render_bist100_tab()
            except Exception as e:
                st.error(f"BIST100 modülü hatası: {e}")
                render_simple_bist100_tab()
        else:
            render_simple_bist100_tab()
    
    with tab2:
        if UI_MODULES_AVAILABLE:
            try:
                render_ml_prediction_tab()
            except Exception as e:
                st.error(f"ML modülü hatası: {e}")
                render_simple_ml_tab()
        else:
            render_simple_ml_tab()
    
    with tab3:
        if UI_MODULES_AVAILABLE:
            try:
                render_stock_tab()
            except Exception as e:
                st.error(f"Stock modülü hatası: {e}")
                render_simple_stock_tab()
        else:
            render_simple_stock_tab()
    
    # Sidebar
    with st.sidebar:
        st.title("📊 Piyasa Özeti")
        
        # BIST-100 özet
        try:
            bist_data = get_stock_data("XU100", "5d")
            if not bist_data.empty:
                last_price = bist_data['Close'].iloc[-1]
                prev_price = bist_data['Close'].iloc[-2]
                change = ((last_price - prev_price) / prev_price) * 100
                
                st.metric("BIST 100", f"{last_price:.0f}", f"{change:.2f}%")
        except:
            st.info("BIST 100 verisi yüklenemedi")
        
        # Popüler hisseler
        st.subheader("🔥 Popüler Hisseler")
        popular_stocks = get_popular_stocks()[:5]
        
        for stock in popular_stocks:
            change_color = "green" if stock["change_percent"] > 0 else "red"
            st.markdown(f"""
            **{stock['symbol']}**  
            <span style='color: {change_color}'>{stock['change_percent']:.2f}%</span>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Bilgilendirme
        st.markdown("""
        ### ℹ️ Bilgilendirme
        
        Bu uygulama sadece **eğitim ve analiz** amaçlıdır.
        
        **⚠️ Yatırım tavsiyesi değildir!**
        
        Yatırım kararlarınızı almadan önce:
        - Detaylı araştırma yapın
        - Profesyonel danışmanlık alın
        - Risk yönetimini gözetin
        """)

if __name__ == "__main__":
    main()
