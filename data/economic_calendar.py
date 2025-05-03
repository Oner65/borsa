"""
Ekonomik takvim ve göstergeler için veri sağlayıcı modülü.
Makroekonomik veriler, merkez bankası açıklamaları, şirket bilançoları ve diğer finansal takvim olaylarını içerir.
"""

import pandas as pd
import numpy as np
import requests
import json
import os
import time
import logging
from datetime import datetime, timedelta
import streamlit as st
import traceback
from bs4 import BeautifulSoup

# Loglama yapılandırması
logger = logging.getLogger(__name__)

# Investpy kütüphanesi kontrolü
try:
    import investpy
    INVESTPY_AVAILABLE = True
except ImportError:
    INVESTPY_AVAILABLE = False
    logger.warning("investpy kütüphanesi yüklenemedi. Bazı ekonomik takvim verileri kullanılamayacak.")

def get_economic_calendar(start_date=None, end_date=None, country="turkey", importance="high"):
    """
    Belirtilen tarih aralığı için ekonomik takvim verilerini döndürür.
    
    Args:
        start_date (str): Başlangıç tarihi (YYYY/MM/DD formatında)
        end_date (str): Bitiş tarihi (YYYY/MM/DD formatında)
        country (str): Ülke kodu (varsayılan: 'turkey')
        importance (str): Önem seviyesi ('high', 'medium', 'low', 'all')
        
    Returns:
        pd.DataFrame: Ekonomik takvim verileri
    """
    if not INVESTPY_AVAILABLE:
        logger.error("Ekonomik takvim için gerekli investpy kütüphanesi yüklü değil.")
        return pd.DataFrame()
    
    try:
        # Tarih değerlerini ayarla
        if start_date is None:
            start_date = datetime.now().strftime('%d/%m/%Y')
        if end_date is None:
            end_date = (datetime.now() + timedelta(days=7)).strftime('%d/%m/%Y')
            
        # Tarihleri investpy formatına dönüştür
        start_date_formatted = datetime.strptime(start_date, '%Y/%m/%d').strftime('%d/%m/%Y') if '/' in start_date else start_date
        end_date_formatted = datetime.strptime(end_date, '%Y/%m/%d').strftime('%d/%m/%Y') if '/' in end_date else end_date
            
        # Veriyi getir
        calendar = investpy.economic_calendar(
            countries=[country],
            from_date=start_date_formatted,
            to_date=end_date_formatted,
            importance=importance
        )
        
        # Veriyi düzenle
        if not calendar.empty:
            # Tarih sütununu datetime tipine dönüştür
            calendar['date'] = pd.to_datetime(calendar['date'])
            
            # Önem seviyesini sayısal değere dönüştür (filtreleme için)
            importance_map = {'high': 3, 'medium': 2, 'low': 1}
            calendar['importance_level'] = calendar['importance'].map(
                lambda x: importance_map.get(x.lower(), 0) if isinstance(x, str) else 0
            )
            
            # Sütunları yeniden adlandır
            calendar = calendar.rename(columns={
                'id': 'event_id',
                'date': 'event_date',
                'time': 'event_time',
                'zone': 'timezone',
                'currency': 'currency',
                'importance': 'importance',
                'event': 'event_name',
                'actual': 'actual_value',
                'forecast': 'forecast_value',
                'previous': 'previous_value'
            })
            
            # Önem seviyesine göre sırala
            calendar = calendar.sort_values(['event_date', 'importance_level'], ascending=[True, False])
            
            logger.info(f"{len(calendar)} ekonomik takvim olayı başarıyla alındı.")
            return calendar
        
        logger.warning("Ekonomik takvim verisi bulunamadı.")
        return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"Ekonomik takvim alınırken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def get_central_bank_announcements(days=30):
    """
    Merkez bankası açıklamalarını ve toplantı tarihlerini getirir.
    
    Args:
        days (int): Kaç gün geriye/ileriye bakılacak
        
    Returns:
        pd.DataFrame: Merkez bankası açıklamaları
    """
    try:
        # Bugünü ve tarih aralığını belirle
        today = datetime.now()
        start_date = (today - timedelta(days=days//2)).strftime('%Y/%m/%d')
        end_date = (today + timedelta(days=days//2)).strftime('%Y/%m/%d')
        
        # Ekonomik takvimden merkez bankası olaylarını filtrele
        calendar = get_economic_calendar(start_date, end_date, country="turkey", importance="all")
        
        if calendar.empty:
            logger.warning("Merkez bankası açıklamaları için veri bulunamadı.")
            return pd.DataFrame()
        
        # Merkez bankası etkinliklerini filtrele (anahtar kelimelere göre)
        keywords = ['merkez', 'tcmb', 'faiz', 'para politikası', 'enflasyon raporu', 'ppr', 'central bank']
        
        # Filtreleme işlemi (büyük/küçük harf duyarsız)
        filtered_events = calendar[
            calendar['event_name'].str.lower().str.contains('|'.join(keywords), na=False)
        ]
        
        if filtered_events.empty:
            logger.warning("Merkez bankası açıklamaları için filtrelenmiş veri bulunamadı.")
            return pd.DataFrame()
        
        # Sonuçları formatla
        filtered_events = filtered_events.sort_values('event_date')
        
        logger.info(f"{len(filtered_events)} merkez bankası açıklaması başarıyla alındı.")
        return filtered_events
    
    except Exception as e:
        logger.error(f"Merkez bankası açıklamaları alınırken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def get_financial_statements_calendar(days=30):
    """
    Şirketlerin finansal tablo açıklama tarihlerini getirir.
    
    Args:
        days (int): Kaç gün geriye/ileriye bakılacak
        
    Returns:
        pd.DataFrame: Finansal tablo açıklama takvimi
    """
    try:
        # Web sitesinden veri çekme simülasyonu
        # Not: Gerçek uygulamada KAP veya başka bir veri kaynağından alınabilir
        
        # Örnek veri oluştur
        today = datetime.now()
        dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(-days//2, days//2)]
        
        companies = [
            'ASELS', 'AKBNK', 'ARCLK', 'BIMAS', 'EKGYO', 
            'EREGL', 'FROTO', 'GARAN', 'HEKTS', 'ISCTR', 
            'KCHOL', 'KOZAL', 'PETKM', 'PGSUS', 'SASA', 
            'SISE', 'THYAO', 'TOASO', 'TUPRS', 'YKBNK'
        ]
        
        # Rastgele şirket ve tarih eşleştirmeleri
        import random
        data = []
        
        for _ in range(40):  # 40 farklı finansal tablo açıklaması
            company = random.choice(companies)
            date = random.choice(dates)
            report_type = random.choice(['Q1', 'Q2', 'Q3', 'Yıllık'])
            
            data.append({
                'stock_code': company,
                'announcement_date': date,
                'report_type': report_type,
                'status': random.choice(['Bekleniyor', 'Açıklandı']),
                'financial_period': f'2024-{report_type}' if 'Q' in report_type else '2023-Yıllık'
            })
        
        # DataFrame oluştur
        df = pd.DataFrame(data)
        
        # Çift kayıtları temizle
        df = df.drop_duplicates(subset=['stock_code', 'report_type'])
        
        # Tarihe göre sırala
        df['announcement_date'] = pd.to_datetime(df['announcement_date'])
        df = df.sort_values('announcement_date')
        
        logger.info(f"{len(df)} finansal tablo açıklama tarihi başarıyla oluşturuldu.")
        return df
    
    except Exception as e:
        logger.error(f"Finansal tablo takvimi alınırken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def get_key_economic_indicators():
    """
    Temel ekonomik göstergelerin son değerlerini getirir (Enflasyon, Büyüme, İşsizlik, Faiz)
    
    Returns:
        dict: Ekonomik göstergeler
    """
    try:
        # Enflasyon verisini simüle et
        inflation_data = {
            'TÜFE (Yıllık)': {
                'value': 51.32,
                'previous': 52.10,
                'change': -0.78,
                'date': '2024-05-01'
            },
            'ÜFE (Yıllık)': {
                'value': 43.78,
                'previous': 45.52,
                'change': -1.74,
                'date': '2024-05-01'
            },
            'Çekirdek Enflasyon': {
                'value': 49.21,
                'previous': 48.68,
                'change': 0.53,
                'date': '2024-05-01'
            }
        }
        
        # Büyüme verisini simüle et
        growth_data = {
            'GSYH Büyüme (Yıllık)': {
                'value': 4.5,
                'previous': 3.8,
                'change': 0.7,
                'date': '2024-03-01'
            },
            'Sanayi Üretimi (Yıllık)': {
                'value': 2.8,
                'previous': 1.5,
                'change': 1.3,
                'date': '2024-04-01'
            }
        }
        
        # İşsizlik verisini simüle et
        unemployment_data = {
            'İşsizlik Oranı': {
                'value': 9.1,
                'previous': 9.4,
                'change': -0.3,
                'date': '2024-04-01'
            }
        }
        
        # Faiz verisini simüle et
        interest_rate_data = {
            'TCMB Politika Faizi': {
                'value': 42.50,
                'previous': 45.00,
                'change': -2.50,
                'date': '2024-05-23'
            },
            '10 Yıllık Tahvil Faizi': {
                'value': 25.82,
                'previous': 26.54,
                'change': -0.72,
                'date': '2024-05-29'
            }
        }
        
        # Dış ticaret verisini simüle et
        trade_data = {
            'İhracat (Milyar $)': {
                'value': 24.5,
                'previous': 23.1,
                'change': 1.4,
                'date': '2024-04-01'
            },
            'İthalat (Milyar $)': {
                'value': 30.2,
                'previous': 29.7,
                'change': 0.5,
                'date': '2024-04-01'
            },
            'Dış Ticaret Dengesi (Milyar $)': {
                'value': -5.7,
                'previous': -6.6,
                'change': 0.9,
                'date': '2024-04-01'
            }
        }
        
        # Tüm veriyi birleştir
        all_indicators = {
            'Enflasyon': inflation_data,
            'Büyüme': growth_data,
            'İşsizlik': unemployment_data,
            'Faiz': interest_rate_data,
            'Dış Ticaret': trade_data,
            'son_güncelleme': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info("Ekonomik göstergeler başarıyla alındı.")
        return all_indicators
    
    except Exception as e:
        logger.error(f"Ekonomik göstergeler alınırken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def search_economic_events(query, days=30):
    """
    Ekonomik takvimde belirli bir arama sorgusu için sonuçları döndürür.
    
    Args:
        query (str): Arama sorgusu
        days (int): Kaç günlük veri alınacak
        
    Returns:
        pd.DataFrame: Arama sonuçları
    """
    try:
        # Tüm takvimi al
        today = datetime.now()
        start_date = (today - timedelta(days=days//2)).strftime('%Y/%m/%d')
        end_date = (today + timedelta(days=days//2)).strftime('%Y/%m/%d')
        
        calendar = get_economic_calendar(start_date, end_date, country="all", importance="all")
        
        if calendar.empty:
            logger.warning("Ekonomik takvim verisi bulunamadı.")
            return pd.DataFrame()
        
        # Arama sorgusu ile filtrele (büyük/küçük harf duyarsız)
        filtered_events = calendar[
            calendar['event_name'].str.lower().str.contains(query.lower(), na=False)
        ]
        
        if filtered_events.empty:
            logger.warning(f"'{query}' sorgusu için ekonomik takvimde sonuç bulunamadı.")
            return pd.DataFrame()
        
        # Sonuçları formatla ve tarihe göre sırala
        filtered_events = filtered_events.sort_values('event_date')
        
        logger.info(f"'{query}' sorgusu için {len(filtered_events)} ekonomik takvim olayı bulundu.")
        return filtered_events
    
    except Exception as e:
        logger.error(f"Ekonomik takvimde arama yapılırken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame() 