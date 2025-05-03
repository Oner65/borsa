"""
Canlı piyasa verisi için entegrasyon modülü.
Borsa İstanbul ve diğer piyasalardan gerçek zamanlı veri alımı için fonksiyonlar içerir.
"""

import pandas as pd
import numpy as np
import requests
import yfinance as yf
import time
import logging
import json
import os
import streamlit as st
from datetime import datetime, timedelta
import traceback

# Interval için varsayılan değerler (tradingview_ta import edilemezse kullanılacak)
class IntervalFallback:
    INTERVAL_1_MINUTE = "1m"
    INTERVAL_5_MINUTES = "5m"
    INTERVAL_15_MINUTES = "15m"
    INTERVAL_30_MINUTES = "30m"
    INTERVAL_1_HOUR = "1h"
    INTERVAL_2_HOURS = "2h"
    INTERVAL_4_HOURS = "4h"
    INTERVAL_1_DAY = "1d"
    INTERVAL_1_WEEK = "1W"
    INTERVAL_1_MONTH = "1M"

# TradingView teknik analiz kütüphanesi
try:
    from tradingview_ta import TA_Handler, Interval, Exchange
    TRADINGVIEW_IMPORTED = True
except ImportError:
    TRADINGVIEW_IMPORTED = False
    logging.warning("tradingview-ta kütüphanesi yüklenemedi. TradingView analizi kullanılamayacak.")
    # Interval sınıfını kullanabilmek için yedek sınıfımızı atayalım
    Interval = IntervalFallback

# CCXT - Kripto ve diğer borsalar için
try:
    import ccxt
    CCXT_IMPORTED = True
except ImportError:
    CCXT_IMPORTED = False
    logging.warning("ccxt kütüphanesi yüklenemedi. Kripto borsaları ve bazı global piyasalar kullanılamayacak.")

# Investpy - Finansal veriler için
try:
    import investpy
    INVESTPY_IMPORTED = True
except ImportError:
    INVESTPY_IMPORTED = False
    logging.warning("investpy kütüphanesi yüklenemedi. Bazı finansal veriler kullanılamayacak.")

# Loglama yapılandırması
logger = logging.getLogger(__name__)

def get_live_stock_data(symbol, interval='1m', limit=100):
    """
    Belirtilen hisse senedi için canlıya yakın veri alır (yfinance API üzerinden).
    
    Args:
        symbol (str): Hisse senedi sembolü (örn: 'GARAN.IS')
        interval (str): '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
        limit (int): Kaç veri noktası alınacak
        
    Returns:
        pd.DataFrame: Canlı hisse senedi verileri
    """
    try:
        # YFinance ile veri çekme
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1d', interval=interval, actions=False)
        
        if data.empty:
            logger.warning(f"{symbol} için {interval} aralığında veri bulunamadı.")
            return pd.DataFrame()
        
        # Sütun isimlerini standartlaştır
        data.columns = [c.title() for c in data.columns]
        
        # Veriyi sınırla
        if len(data) > limit:
            data = data.tail(limit)
            
        # Son güncellenme zamanını ekle
        data['LastUpdated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"{symbol} için canlı veri başarıyla alındı. Son fiyat: {data['Close'].iloc[-1]:.2f}")
        return data
    
    except Exception as e:
        logger.error(f"{symbol} için canlı veri alınırken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def get_tradingview_analysis(symbol, exchange="BIST", screener="turkey", interval=Interval.INTERVAL_1_DAY):
    """
    TradingView teknik analiz özetini alır.
    
    Args:
        symbol (str): Hisse senedi sembolü (örn: 'GARAN')
        exchange (str): Borsa (örn: 'BIST')
        screener (str): Ülke/bölge (örn: 'turkey')
        interval (str): Zaman aralığı (Interval sınıfından)
        
    Returns:
        dict: TradingView analiz sonuçları
    """
    if not TRADINGVIEW_IMPORTED:
        logger.warning("TradingView analizi için gerekli kütüphane eksik.")
        return {"error": "TradingView kütüphanesi yüklü değil"}
    
    try:
        # Sembolü düzelt (gerekirse .IS ekini kaldır)
        clean_symbol = symbol.replace('.IS', '')
        
        # TradingView handler'ı başlat
        handler = TA_Handler(
            symbol=clean_symbol,
            exchange=exchange,
            screener=screener,
            interval=interval
        )
        
        # Analizi getir
        analysis = handler.get_analysis()
        
        # Sonuçları düzenle
        result = {
            "summary": {
                "recommendation": analysis.summary["RECOMMENDATION"],
                "buy_signals": analysis.summary["BUY"],
                "sell_signals": analysis.summary["SELL"],
                "neutral_signals": analysis.summary["NEUTRAL"]
            },
            "oscillators": {
                "recommendation": analysis.oscillators["RECOMMENDATION"],
                "buy_signals": analysis.oscillators["BUY"],
                "sell_signals": analysis.oscillators["SELL"],
                "neutral_signals": analysis.oscillators["NEUTRAL"]
            },
            "moving_averages": {
                "recommendation": analysis.moving_averages["RECOMMENDATION"],
                "buy_signals": analysis.moving_averages["BUY"],
                "sell_signals": analysis.moving_averages["SELL"],
                "neutral_signals": analysis.moving_averages["NEUTRAL"]
            },
            "indicators": analysis.indicators
        }
        
        logger.info(f"{symbol} için TradingView analizi başarıyla alındı. Öneri: {result['summary']['recommendation']}")
        return result
    
    except Exception as e:
        logger.error(f"{symbol} için TradingView analizi alınırken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def get_market_snapshot():
    """
    Türkiye ve global piyasaların anlık durumunu getirir.
    
    Returns:
        dict: Piyasa genel görünümü
    """
    try:
        # BIST endeksi
        bist100 = yf.Ticker("XU100.IS")
        bist100_data = bist100.history(period="1d")
        
        # USD/TRY
        usdtry = yf.Ticker("USDTRY=X")
        usdtry_data = usdtry.history(period="1d")
        
        # EUR/TRY
        eurtry = yf.Ticker("EURTRY=X")
        eurtry_data = eurtry.history(period="1d")
        
        # Altın (Ons)
        gold = yf.Ticker("GC=F")
        gold_data = gold.history(period="1d")
        
        # Petrol (Brent)
        oil = yf.Ticker("BZ=F")
        oil_data = oil.history(period="1d")
        
        # S&P 500
        sp500 = yf.Ticker("^GSPC")
        sp500_data = sp500.history(period="1d")
        
        # NASDAQ
        nasdaq = yf.Ticker("^IXIC")
        nasdaq_data = nasdaq.history(period="1d")
        
        # Sonuçları topla
        snapshot = {
            "BIST100": {
                "last": float(bist100_data["Close"].iloc[-1]) if not bist100_data.empty else None,
                "change": float(bist100_data["Close"].iloc[-1] - bist100_data["Open"].iloc[0]) if not bist100_data.empty else None,
                "change_pct": float((bist100_data["Close"].iloc[-1] / bist100_data["Open"].iloc[0] - 1) * 100) if not bist100_data.empty else None
            },
            "USD/TRY": {
                "last": float(usdtry_data["Close"].iloc[-1]) if not usdtry_data.empty else None,
                "change": float(usdtry_data["Close"].iloc[-1] - usdtry_data["Open"].iloc[0]) if not usdtry_data.empty else None,
                "change_pct": float((usdtry_data["Close"].iloc[-1] / usdtry_data["Open"].iloc[0] - 1) * 100) if not usdtry_data.empty else None
            },
            "EUR/TRY": {
                "last": float(eurtry_data["Close"].iloc[-1]) if not eurtry_data.empty else None,
                "change": float(eurtry_data["Close"].iloc[-1] - eurtry_data["Open"].iloc[0]) if not eurtry_data.empty else None,
                "change_pct": float((eurtry_data["Close"].iloc[-1] / eurtry_data["Open"].iloc[0] - 1) * 100) if not eurtry_data.empty else None
            },
            "Altın": {
                "last": float(gold_data["Close"].iloc[-1]) if not gold_data.empty else None,
                "change": float(gold_data["Close"].iloc[-1] - gold_data["Open"].iloc[0]) if not gold_data.empty else None,
                "change_pct": float((gold_data["Close"].iloc[-1] / gold_data["Open"].iloc[0] - 1) * 100) if not gold_data.empty else None
            },
            "Petrol": {
                "last": float(oil_data["Close"].iloc[-1]) if not oil_data.empty else None,
                "change": float(oil_data["Close"].iloc[-1] - oil_data["Open"].iloc[0]) if not oil_data.empty else None,
                "change_pct": float((oil_data["Close"].iloc[-1] / oil_data["Open"].iloc[0] - 1) * 100) if not oil_data.empty else None
            },
            "S&P500": {
                "last": float(sp500_data["Close"].iloc[-1]) if not sp500_data.empty else None,
                "change": float(sp500_data["Close"].iloc[-1] - sp500_data["Open"].iloc[0]) if not sp500_data.empty else None,
                "change_pct": float((sp500_data["Close"].iloc[-1] / sp500_data["Open"].iloc[0] - 1) * 100) if not sp500_data.empty else None
            },
            "NASDAQ": {
                "last": float(nasdaq_data["Close"].iloc[-1]) if not nasdaq_data.empty else None,
                "change": float(nasdaq_data["Close"].iloc[-1] - nasdaq_data["Open"].iloc[0]) if not nasdaq_data.empty else None,
                "change_pct": float((nasdaq_data["Close"].iloc[-1] / nasdaq_data["Open"].iloc[0] - 1) * 100) if not nasdaq_data.empty else None
            },
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info("Piyasa anlık görünümü başarıyla alındı.")
        return snapshot
    
    except Exception as e:
        logger.error(f"Piyasa anlık görünümü alınırken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def get_stock_alerts(symbol, price, condition, target_price):
    """
    Hisse senedi için fiyat alarmı oluşturur.
    
    Args:
        symbol (str): Hisse senedi sembolü
        price (float): Mevcut fiyat
        condition (str): 'above' veya 'below'
        target_price (float): Hedef fiyat
        
    Returns:
        bool: Alarm tetiklendi mi
    """
    try:
        if condition == 'above' and price > target_price:
            return True
        elif condition == 'below' and price < target_price:
            return False
        return False
    except Exception as e:
        logger.error(f"Fiyat alarmı kontrol edilirken hata: {str(e)}")
        return False 