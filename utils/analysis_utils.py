"""
Borsa Uygulaması için Analiz Yardımcı Modülü
Bu modül, risk seviyesi ve öneri hesaplamaları için parametrik fonksiyonlar içerir.
"""

import pandas as pd
import numpy as np
from config import RISK_THRESHOLDS, RECOMMENDATION_THRESHOLDS

def calculate_risk_level(volatility, thresholds=None):
    """
    Volatilite değerine göre risk seviyesini hesaplar.
    
    Args:
        volatility (float): Volatilite değeri (%)
        thresholds (dict, optional): Risk eşik değerleri. Varsayılan: RISK_THRESHOLDS
        
    Returns:
        tuple: (risk_level, risk_color) risk seviyesi ve renk kodu
    """
    if thresholds is None:
        thresholds = RISK_THRESHOLDS
        
    if volatility is None:
        return "BELİRSİZ", "gray"
        
    if volatility <= thresholds["low"]:
        return "DÜŞÜK", "green"
    elif volatility <= thresholds["medium"]:
        return "ORTA", "orange"
    else:
        return "YÜKSEK", "red"

def calculate_recommendation(total_signal, indicator_count, thresholds=None):
    """
    Sinyal skorlarına göre yatırım tavsiyesi hesaplar.
    
    Args:
        total_signal (float): Toplam sinyal skoru
        indicator_count (int): Toplam gösterge sayısı
        thresholds (dict, optional): Öneri eşik değerleri. Varsayılan: RECOMMENDATION_THRESHOLDS
        
    Returns:
        tuple: (recommendation_text, recommendation_color) tavsiye metni ve renk kodu
    """
    if thresholds is None:
        thresholds = RECOMMENDATION_THRESHOLDS
        
    ratio = total_signal / indicator_count if indicator_count > 0 else 0
    
    if ratio > thresholds["strong_buy"]:
        return "GÜÇLÜ AL", "darkgreen"
    elif ratio > 0:
        return "AL", "green"
    elif ratio < thresholds["strong_sell"]:
        return "GÜÇLÜ SAT", "darkred"
    elif ratio < 0:
        return "SAT", "red"
    else:
        return "NÖTR", "gray"

def determine_trend(df, sma_columns):
    """
    Fiyat hareketleri ve hareketli ortalamalardan trend yönünü belirler.
    
    Args:
        df (DataFrame): Veri seti (Close fiyatları ve hareketli ortalamalar içermeli)
        sma_columns (list): Hareketli ortalama sütun adları (örn. ['SMA20', 'SMA50', 'SMA200'])
        
    Returns:
        dict: Trend bilgileri
    """
    trend_info = {}
    close_price = df['Close'].iloc[-1]
    
    # Kısa-orta-uzun vadeli trend belirlemeleri
    trend_info['short_term'] = "Yükseliş" if close_price > df[sma_columns[0]].iloc[-1] else "Düşüş"
    trend_info['medium_term'] = "Yükseliş" if close_price > df[sma_columns[1]].iloc[-1] else "Düşüş"
    trend_info['long_term'] = "Yükseliş" if close_price > df[sma_columns[2]].iloc[-1] else "Düşüş"
    
    # Trendin gücü
    trend_info['short_term_strength'] = abs((close_price / df[sma_columns[0]].iloc[-1] - 1) * 100)
    trend_info['medium_term_strength'] = abs((close_price / df[sma_columns[1]].iloc[-1] - 1) * 100)
    trend_info['long_term_strength'] = abs((close_price / df[sma_columns[2]].iloc[-1] - 1) * 100)
    
    # Genel trend yönü
    if trend_info['short_term'] == "Yükseliş" and trend_info['medium_term'] == "Yükseliş":
        trend_info['direction'] = "yükseliş eğiliminde"
    elif trend_info['short_term'] == "Düşüş" and trend_info['medium_term'] == "Düşüş":
        trend_info['direction'] = "düşüş eğiliminde"
    elif trend_info['short_term'] == "Yükseliş" and trend_info['medium_term'] == "Düşüş":
        trend_info['direction'] = "kısa vadede yükseliş gösterse de genel düşüş eğiliminde"
    elif trend_info['short_term'] == "Düşüş" and trend_info['medium_term'] == "Yükseliş":
        trend_info['direction'] = "kısa vadede düşüş gösterse de genel yükseliş eğiliminde"
    else:
        trend_info['direction'] = "yatay seyretmekte"
        
    # Genel trend gücü ve rengi
    if trend_info['short_term'] == trend_info['medium_term'] == trend_info['long_term'] == "Yükseliş":
        trend_info['overall'] = "GÜÇLÜ YÜKSELİŞ TRENDİ"
        trend_info['color'] = "darkgreen"
    elif trend_info['short_term'] == trend_info['medium_term'] == trend_info['long_term'] == "Düşüş":
        trend_info['overall'] = "GÜÇLÜ DÜŞÜŞ TRENDİ"
        trend_info['color'] = "darkred"
    elif trend_info['long_term'] == "Yükseliş" and (trend_info['short_term'] == "Yükseliş" or trend_info['medium_term'] == "Yükseliş"):
        trend_info['overall'] = "YÜKSELİŞ TRENDİ"
        trend_info['color'] = "green"
    elif trend_info['long_term'] == "Düşüş" and (trend_info['short_term'] == "Düşüş" or trend_info['medium_term'] == "Düşüş"):
        trend_info['overall'] = "DÜŞÜŞ TRENDİ"
        trend_info['color'] = "red"
    elif trend_info['short_term'] != trend_info['medium_term'] or trend_info['medium_term'] != trend_info['long_term']:
        trend_info['overall'] = "TREND BELİRSİZ"
        trend_info['color'] = "gray"
    else:
        trend_info['overall'] = "YATAY TREND"
        trend_info['color'] = "blue"
        
    return trend_info

def generate_analysis_summary(stock_symbol, trend_info, risk_level, recommendation, price_changes, market_info="", news_info=""):
    """
    Analiz sonuçlarına göre özet metin oluşturur.
    
    Args:
        stock_symbol (str): Hisse senedi sembolü
        trend_info (dict): Trend bilgileri
        risk_level (str): Risk seviyesi
        recommendation (str): Tavsiye metni
        price_changes (dict): Fiyat değişim bilgileri
        market_info (str, optional): Piyasa bilgisi metni
        news_info (str, optional): Haber bilgisi metni
        
    Returns:
        str: Analiz özet metni
    """
    risk_desc = "düşük riskli" if risk_level == "DÜŞÜK" else ("orta riskli" if risk_level == "ORTA" else "yüksek riskli")
    
    # Öneriye göre aksiyon belirleme
    if "AL" in recommendation:
        if "GÜÇLÜ" in recommendation:
            action = "alım için uygun görünüyor"
        else:
            action = "dikkatli bir şekilde alım için değerlendirilebilir"
    elif "SAT" in recommendation:
        if "GÜÇLÜ" in recommendation:
            action = "satış için uygun görünüyor"
        else:
            action = "satış düşünülebilir"
    else:
        action = "bekleme pozisyonunda kalınması uygun olabilir"
    
    # Değerlendirmeyi birleştir
    last_week_trend = f"Son bir haftada %{price_changes.get('1w', 0):.2f} " if price_changes.get('1w') is not None else ""
    
    summary = f"{stock_symbol} hissesi şu anda {trend_info['direction']} ve {risk_desc} bir yatırım olarak değerlendirilmektedir. " + \
              f"{last_week_trend}değişim göstermiş olup, teknik analiz sonuçlarına göre {action}. " + \
              f"{market_info} {news_info} " + \
              f"Yatırım kararınızda bu analiz sonuçlarının yanı sıra şirketin temel verilerini ve piyasa koşullarını da dikkate almanız önerilir."
              
    return summary 