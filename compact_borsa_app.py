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

def main():
    """
    Kompakt borsa uygulaması - BIST 100, Hisse Analizi ve ML Tahminleri
    """
    
    # CSS stil ekle - Sade tasarım
    st.markdown("""
    <style>
        /* Ana stil ayarları */
        .main-header {
            color: #2e86de;
            text-align: center;
            margin-bottom: 30px;
        }
        
        .stSelectbox > div > div > div {
            background-color: #f0f2f6;
        }
        
        .metric-card {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
        }
        
        /* Sidebar stil */
        .css-1d391kg {
            background-color: #f8f9fa;
        }
        
        /* Buton stil */
        .stButton > button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: bold;
        }
        
        .stButton > button:hover {
            background: linear-gradient(90deg, #5a6fd8 0%, #6a4190 100%);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Ana başlık
    st.markdown('<h1 class="main-header">📊 Kompakt Borsa Analiz Uygulaması</h1>', unsafe_allow_html=True)
    
    # Sidebar - Sadece sayfa seçimi
    with st.sidebar:
        st.markdown("### 🎯 Sayfa Seçimi")
        
        selected_page = st.selectbox(
            "Hangi sayfaya gitmek istiyorsunuz?",
            ["📊 BIST100 Genel Bakış", "🔍 Hisse Analizi", "🔎 ML Tarama"],
            key="page_selector"
        )
        
        st.divider()
        
        # Piyasa özeti
        st.markdown("### 📈 Piyasa Durumu")
        try:
            bist100 = yf.Ticker("XU100.IS")
            hist = bist100.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_close = bist100.info.get('previousClose', current_price)
                change = ((current_price - prev_close) / prev_close) * 100
                
                st.metric(
                    label="BIST 100",
                    value=f"{current_price:.0f}",
                    delta=f"{change:.2f}%"
                )
        except:
            st.warning("Piyasa verileri yüklenemedi")
        
        st.divider()
        
        # Bilgi
        st.markdown("### ℹ️ Bilgi")
        st.markdown("""
        **Kompakt Borsa Analizi**
        
        Bu uygulama:
        - ✅ BIST 100 takibi
        - ✅ Teknik analiz
        - ✅ ML tahminleri
        - ✅ Favori hisseler
        
        *Veriler yfinance'dan alınır*
        """)
    
    # İçerik alanı - Seçilen sayfaya göre
    if selected_page == "📊 BIST100 Genel Bakış":
        if MODULES_AVAILABLE:
            render_bist100_tab()
        else:
            st.error("BIST100 modülü yüklenemedi")
            
    elif selected_page == "🔍 Hisse Analizi":
        if MODULES_AVAILABLE:
            render_stock_tab()
        else:
            st.error("Hisse analizi modülü yüklenemedi")
            
    elif selected_page == "🔎 ML Tarama":
        if MODULES_AVAILABLE:
            render_ml_prediction_tab()
        else:
            st.error("ML tarama modülü yüklenemedi")

if __name__ == "__main__":
    main()
