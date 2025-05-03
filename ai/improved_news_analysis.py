import os
import sys
import re
import requests
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import logging
import numpy as np
import time
import random
import concurrent.futures
from urllib.parse import quote_plus
import traceback
from typing import Dict, List, Tuple, Any, Optional, Union

# Proje dizinini path'e ekle
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)
    print(f"{datetime.now().strftime('%m/%d/%Y %I:%M:%S %p')} - Proje kök dizini sys.path'e eklendi: {project_dir}\n")

# Veritabanı oluşturma işlemi
try:
    from data.create_database import create_db_if_not_exists
    # SQLite veritabanını oluştur veya bağlan
    db_file = create_db_if_not_exists()
    print(f"{datetime.now().strftime('%m/%d/%Y %I:%M:%S %p')} - Veritabanı başarıyla oluşturuldu")
except Exception as e:
    print(f"Veritabanı oluşturma hatası: {e}")

# Gelişmiş duyarlılık analizini içe aktar
try:
    from ai.sentiment_analysis_improved import ImprovedSentimentAnalyzer
    improved_analyzer = ImprovedSentimentAnalyzer(use_bert=True)
    IMPROVED_ANALYZER_AVAILABLE = True
    print("Gelişmiş duyarlılık analizi modülü başarıyla yüklendi.")
except ImportError as e:
    print(f"Gelişmiş duyarlılık analizi modülü yüklenemedi: {e}")
    IMPROVED_ANALYZER_AVAILABLE = False
    improved_analyzer = None

# Yedek olarak mevcut duyarlılık analizini içe aktar
try:
    from ai.sentiment_analysis import SentimentAnalyzer
    backup_analyzer = SentimentAnalyzer()
    BACKUP_ANALYZER_AVAILABLE = True
    print(f"Yedek analiz modeli başarıyla yüklendi: {project_dir}/ai/sentiment_model.pkl")
except ImportError:
    BACKUP_ANALYZER_AVAILABLE = False
    backup_analyzer = None
    
# Gelişmiş terim sözlüğü modülü
try:
    # Gelişmiş finansal terimler sözlüğü
    from ai.sentiment_terms import analyze_text_with_terms, positive_terms, negative_terms
    # Gelişmiş terim analizi aktif
    term_analysis_available = True
except Exception as e:
    print(f"Gelişmiş terim analizi yüklenemedi: {e}")
    term_analysis_available = False

# Veri kaynakları
NEWS_SOURCES = {
    'Google News': 'https://news.google.com/rss/search?q={query}&hl=tr&gl=TR&ceid=TR:tr',
    'Finans Gündem': 'https://www.finansgundem.com/arama/{query}',
    'Bloomberght': 'https://www.bloomberght.com/arama?query={query}',
    'Ekonomim': 'https://ekonomim.com/arama/{query}',
    'Dünya': 'https://www.dunya.com/arama/{query}',
}

# Sektör bazlı anahtar kelimeler
SECTOR_KEYWORDS = {
    'BANKA': ['faiz', 'kredi', 'mevduat', 'bankacılık', 'finans', 'tl', 'dolar', 'euro', 'merkez bankası', 'tcmb'],
    'TEKNOLOJİ': ['yazılım', 'teknoloji', 'bilişim', 'ar-ge', 'yapay zeka', 'dijital', 'inovasyon'],
    'OTOMOTİV': ['otomotiv', 'araç', 'otomobil', 'ihracat', 'ithalat', 'üretim', 'satış'],
    'ENERJİ': ['enerji', 'elektrik', 'doğalgaz', 'petrol', 'yenilenebilir', 'rüzgar', 'güneş'],
    'PERAKENDE': ['perakende', 'mağaza', 'satış', 'e-ticaret', 'tüketici', 'alışveriş'],
    'MADENCİLİK': ['maden', 'altın', 'gümüş', 'bakır', 'çinko', 'demir', 'çelik', 'metal'],
    'TELEKOMÜNİKASYON': ['telekom', 'iletişim', 'cep telefonu', 'internet', 'fiber', 'altyapı'],
    'GAYRİMENKUL': ['gayrimenkul', 'inşaat', 'konut', 'emlak', 'kira', 'fiyat'],
    'GIDA': ['gıda', 'tarım', 'süt', 'et', 'içecek', 'ihracat', 'üretim'],
    'SAĞLIK': ['sağlık', 'ilaç', 'hastane', 'medikal', 'tıbbi cihaz'],
}

# Haber kaynaklarına dair veri tipleri
DATA_SOURCES = {
    'AKBNK.IS': {
        'search_terms': ['"Akbank" OR "AKBNK" hisse OR borsa OR yatırım OR ekonomi'],
        'sector': 'bankacılık',
        'related_companies': ['GARAN.IS', 'YKBNK.IS', 'ISCTR.IS', 'HALKB.IS'],
    },
    'GARAN.IS': {
        'search_terms': ['"Garanti Bankası" OR "GARAN" hisse OR borsa OR yatırım OR ekonomi'],
        'sector': 'bankacılık',
        'related_companies': ['AKBNK.IS', 'YKBNK.IS', 'ISCTR.IS', 'HALKB.IS'],
    },
    'YKBNK.IS': {
        'search_terms': ['"Yapı Kredi" OR "YKBNK" hisse OR borsa OR yatırım OR ekonomi'],
        'sector': 'bankacılık',
        'related_companies': ['AKBNK.IS', 'GARAN.IS', 'ISCTR.IS', 'HALKB.IS'],
    },
    'ISCTR.IS': {
        'search_terms': ['"İş Bankası" OR "ISCTR" hisse OR borsa OR yatırım OR ekonomi'],
        'sector': 'bankacılık',
        'related_companies': ['AKBNK.IS', 'GARAN.IS', 'YKBNK.IS', 'HALKB.IS'],
    },
    'HALKB.IS': {
        'search_terms': ['"Halkbank" OR "HALKB" hisse OR borsa OR yatırım OR ekonomi'],
        'sector': 'bankacılık',
        'related_companies': ['AKBNK.IS', 'GARAN.IS', 'YKBNK.IS', 'ISCTR.IS'],
    },
    'THYAO.IS': {
        'search_terms': ['"Türk Hava Yolları" OR "THY" OR "Turkish Airlines" hisse OR borsa OR yatırım OR ekonomi'],
        'sector': 'havacılık',
        'related_companies': ['PGSUS.IS'],
    },
    'PGSUS.IS': {
        'search_terms': ['"Pegasus" OR "PGSUS" hisse OR borsa OR yatırım OR ekonomi'],
        'sector': 'havacılık',
        'related_companies': ['THYAO.IS'],
    },
    'KCHOL.IS': {
        'search_terms': ['"Koç Holding" OR "KCHOL" hisse OR borsa OR yatırım OR ekonomi'],
        'sector': 'holding',
        'related_companies': ['SAHOL.IS', 'FROTO.IS', 'TOASO.IS', 'ARCLK.IS'],
    },
    'SAHOL.IS': {
        'search_terms': ['"Sabancı Holding" OR "SAHOL" hisse OR borsa OR yatırım OR ekonomi'],
        'sector': 'holding',
        'related_companies': ['KCHOL.IS'],
    },
    'ARCLK.IS': {
        'search_terms': ['"Arçelik" OR "ARCLK" hisse OR borsa OR yatırım OR ekonomi'],
        'sector': 'beyaz eşya',
        'related_companies': ['KCHOL.IS'],
    },
    'ASELS.IS': {
        'search_terms': ['"Aselsan" OR "ASELS" hisse OR borsa OR yatırım OR ekonomi'],
        'sector': 'savunma',
        'related_companies': ['KCAER.IS'],
    },
    'KCAER.IS': {
        'search_terms': ['"Kale Havacılık" OR "KCAER" hisse OR borsa OR yatırım OR ekonomi'],
        'sector': 'savunma',
        'related_companies': ['ASELS.IS'],
    },
    'FROTO.IS': {
        'search_terms': ['"Ford Otosan" OR "FROTO" hisse OR borsa OR yatırım OR ekonomi'],
        'sector': 'otomotiv',
        'related_companies': ['TOASO.IS', 'KCHOL.IS'],
    },
    'TOASO.IS': {
        'search_terms': ['"Tofaş" OR "TOASO" hisse OR borsa OR yatırım OR ekonomi'],
        'sector': 'otomotiv',
        'related_companies': ['FROTO.IS'],
    },
    'EKGYO.IS': {
        'search_terms': ['"Emlak Konut" OR "EKGYO" hisse OR borsa OR yatırım OR ekonomi'],
        'sector': 'gayrimenkul',
        'related_companies': [],
    },
    'SASA.IS': {
        'search_terms': ['"Sasa Polyester" OR "SASA" hisse OR borsa OR yatırım OR ekonomi'],
        'sector': 'kimya',
        'related_companies': [],
    },
    'TUPRS.IS': {
        'search_terms': ['"Tüpraş" OR "TUPRS" hisse OR borsa OR yatırım OR ekonomi'],
        'sector': 'enerji',
        'related_companies': ['KCHOL.IS'],
    },
    'BIMAS.IS': {
        'search_terms': ['"BİM" OR "BIMAS" hisse OR borsa OR yatırım OR ekonomi'],
        'sector': 'perakende',
        'related_companies': ['MGROS.IS'],
    },
    'MGROS.IS': {
        'search_terms': ['"Migros" OR "MGROS" hisse OR borsa OR yatırım OR ekonomi'],
        'sector': 'perakende',
        'related_companies': ['BIMAS.IS'],
    },
    'TAVHL.IS': {
        'search_terms': ['"TAV" OR "TAVHL" hisse OR borsa OR yatırım OR ekonomi'],
        'sector': 'havacılık',
        'related_companies': ['THYAO.IS'],
    },
    'EREGL.IS': {
        'search_terms': ['"Ereğli Demir Çelik" OR "EREGL" hisse OR borsa OR yatırım OR ekonomi'],
        'sector': 'demir-çelik',
        'related_companies': [],
    },
    # Varsayılan arama terimi
    'DEFAULT': {
        'search_terms': ['hisse fiyat yükseliş düşüş borsa'],
        'sector': 'genel',
        'related_companies': [],
    }
}

# Gelişmiş haber analizi için Google News sınıfı
class GoogleNews:
    def __init__(self, lang='tr', country='TR'):
        """
        Türkçe Google News'den haber aramak için sınıf.
        
        Args:
            lang (str, optional): Dil kodu. Varsayılan 'tr'.
            country (str, optional): Ülke kodu. Varsayılan 'TR'.
        """
        self.lang = lang 
        self.country = country
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

    def search(self, query, max_results=10):
        """
        Google News'de arama yapar ve sonuçları döndürür.
        
        Args:
            query (str): Arama sorgusu
            max_results (int, optional): Maksimum sonuç sayısı. Varsayılan 10.
            
        Returns:
            list: Bulunan haberlerin listesi
        """
        try:
            # RSS sorgusu oluştur
            query_encoded = quote_plus(query)
            rss_url = f'https://news.google.com/rss/search?q={query_encoded}&hl={self.lang}-{self.country}&gl={self.country}&ceid={self.country}:{self.lang.lower()}'
            
            print(f"RSS URL deneniyor: {rss_url}")
            
            # İstek gönder
            response = requests.get(rss_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # XML/HTML'i parse et
            try:
                from bs4 import XMLParsedAsHTMLWarning
                import warnings
                warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
            except:
                pass
                
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Bu noktada içeriğin XML olup olmadığını kontrol et
            if 'xml' in response.headers.get('Content-Type', '').lower() or '<rss' in response.text:
                # XML formatı
                items = soup.find_all('item')
                print(f"XML Parser ile {len(items)} haber bulundu.")
            else:
                # HTML formatı (alternatif)
                items = soup.find_all('article')
                if not items:
                    # Eğer article etiketleri bulunamazsa, div.dbsr gibi etiketleri dene
                    items = soup.select('div.dbsr')
                print(f"HTML/XML Parser ile {len(items)} haber bulundu.")
                
            results = []
            counter = 0
            
            for item in items:
                if counter >= max_results:
                    break
                    
                try:
                    # XML formatı için
                    if item.name == 'item':
                        title = item.title.text if item.title else "Başlık bilgisi yok"
                        link = item.link.text if item.link else item.find('link', href=True)
                        if not link:
                            continue
                            
                        pub_date = item.pubDate.text if item.pubDate else ""
                        description = item.description.text if item.description else ""
                        
                        # Kaynak bilgisini extract et
                        source = "Google News"
                        source_tag = item.find('source')
                        if source_tag and source_tag.text:
                            source = source_tag.text
                        
                    # HTML formatı için (daha güvenilir)    
                    else:
                        title_elem = item.find("h3") or item.find("a", class_="titlelnk")
                        title = title_elem.text.strip() if title_elem else "Başlık bilgisi yok"
                        
                        link_elem = item.find("a", href=True)
                        link = link_elem['href'] if link_elem else ""
                        
                        # Görecel linkler
                        if link and link.startswith('./'):
                            link = f"https://news.google.com{link[1:]}"
                        
                        # Kaynak bilgisi
                        source_elem = item.find("span", class_="source") or item.find("cite", class_="lihb")
                        source = source_elem.text.strip() if source_elem else "Google News"
                        
                        # Yayın tarihi
                        date_elem = item.find("time") or item.find("span", class_="date")
                        pub_date = date_elem.text.strip() if date_elem else ""
                        
                        # Özet
                        desc_elem = item.find("div", class_="st") or item.find("p")
                        description = desc_elem.text.strip() if desc_elem else ""
                        
                    # Sonuçlara ekle    
                    results.append({
                        'title': title,
                        'link': link,
                        'pub_date': pub_date,
                        'source': source,
                        'description': description
                    })
                    
                    counter += 1
                    
                except Exception as e:
                    print(f"Haber parse edilirken hata: {e}")
                    continue
            
            return results
        
        except Exception as e:
            print(f"Google News aramasında hata: {e}")
            return []

def get_news_for_stock(stock_code, max_news=5):
    """
    Bir hisse için en güncel haberleri getirir.
    
    Args:
        stock_code (str): Hisse kodu
        max_news (int, optional): Maksimum haber sayısı. Varsayılan 5.
        
    Returns:
        list: Haberler listesi
    """
    print(f"[{stock_code}] için haber analizi başlıyor...")
    
    try:
        # Veri kaynağını kontrol et
        stock_data = DATA_SOURCES.get(stock_code, DATA_SOURCES['DEFAULT'])
        search_terms = stock_data['search_terms']
        sector = stock_data['sector']
        
        # Haberleri getir
        news_data = []
        google_news = GoogleNews()
        
        for search_term in search_terms:
            # Sektör anahtar kelimelerini ekle
            sector_keywords = SECTOR_KEYWORDS.get(sector, [])
            if sector_keywords:
                sector_term = " OR ".join([f'"{keyword}"' for keyword in sector_keywords[:3]])  # İlk 3 kelimeyi ekle
                enhanced_term = f'{search_term} OR ({sector_term})'
            else:
                enhanced_term = search_term
                
            print(f"[{stock_code}] için haber araması yapılıyor... Sorgu: {enhanced_term}")
            
            results = google_news.search(enhanced_term, max_results=max_news*2)  # Filtre sonrası yeterli haber kalması için
            
            if results:
                # Sonuçları filtrele
                filtered_results = []
                for news in results:
                    # Kopya kontrol
                    if any(existing['link'] == news['link'] for existing in news_data):
                        continue
                        
                    # Stok kodu ve sektör kelimesi kontrolü yaparak ilgili haberleri ön plana çıkar
                    relevance_score = 0
                    
                    # Başlıkta hisse kodu geçiyorsa
                    if stock_code.replace('.IS', '') in news['title']:
                        relevance_score += 3
                        
                    # Başlıkta veya açıklamada sektör kelimesi geçiyorsa  
                    for keyword in sector_keywords:
                        if keyword.lower() in news['title'].lower() or \
                           (news['description'] and keyword.lower() in news['description'].lower()):
                            relevance_score += 1
                            break
                    
                    news['relevance'] = relevance_score
                    filtered_results.append(news)
                
                # Önce ilgili haberleri göster
                filtered_results.sort(key=lambda x: x['relevance'], reverse=True)
                
                # En ilgili haberleri ekle
                news_data.extend(filtered_results[:max_news])
                
                # Yeterli haber varsa döngüden çık
                if len(news_data) >= max_news:
                    news_data = news_data[:max_news]
                    break
        
        # Haber bulunamadıysa 
        if not news_data:
            print(f"[{stock_code}] için haber bulunamadı.")
            return []
            
        print(f"[{stock_code}] için {len(news_data)} adet haber bulundu.")
        
        # Haberleri listele
        for i, news in enumerate(news_data[:3], 1):
            print(f"{i}. {news['title']} <a href=\"{news['link']}...\"")
            
        return news_data
        
    except Exception as e:
        print(f"Haber getirme hatası: {e}")
        traceback.print_exc()
        return []

def analyze_news_sentiment(news_list, stock_code=None):
    """
    Haberleri analiz ederek duyarlılık skorları hesaplar.
    
    Args:
        news_list (list): Analiz edilecek haberler listesi
        stock_code (str, optional): Hisse kodu
        
    Returns:
        dict: Analiz sonuçları
    """
    if not news_list:
        print(f"[{stock_code}] Haber bulunmadığı için nötr skor (0) kullanılıyor.")
        return {
            'score': 0.5,  # Normalize edilmiş nötr skor
            'normalized_score': 0.0,  # -1 ile 1 arasında normalize skor
            'sentiment': 'Nötr',
            'analyzed_count': 0,
            'failed_count': 0,
            'news_sentiments': []
        }
    
    print(f"[{stock_code}] Haberler analiz ediliyor...")
    
    sentiment_scores = []
    analyzed_news = []
    failed_count = 0
    
    for news in news_list:
        try:
            title = news.get('title', '')
            link = news.get('link', '')
            description = news.get('description', '')
            
            # Link kontrolü
            if not link:
                continue
                
            # Bu haber daha önce analiz edilmiş mi?
            if any(n['link'] == link for n in analyzed_news):
                continue
                
            print(f"-----> Haber analiz ediliyor: {title}")
            
            # İçerik kontrolü
            content = description if description else title
            
            # Hisse kodunu veya sembolünü içeriyorsa bunu loglayalım
            if stock_code and stock_code.replace('.IS', '') in content:
                print(f"-----> Hisse kodu ({stock_code}) içerikte geçiyor. Faydalı bir haber olabilir.")
            
            # Duyarlılık analizi yap
            sentiment_score = 0.5  # Varsayılan nötr değer
            sentiment_label = "Nötr"
            
            # 1. Önce gelişmiş kelime-tabanlı analiz dene
            if term_analysis_available:
                try:
                    term_result = analyze_text_with_terms(content)
                    
                    # Önemli terimleri logla
                    if term_result['positive_terms']:
                        print(f"-----> Olumlu kelime sayısı: {len(term_result['positive_terms'])}")
                    
                    if term_result['negative_terms']:
                        print(f"-----> Olumsuz kelime sayısı: {len(term_result['negative_terms'])}")
                    
                    # Özel haberleri tespit et
                    if term_result['special_announcements']:
                        print(f"-----> Önemli duyuru tespit edildi: {term_result['special_announcements']}")
                    
                    sentiment_score = term_result['score']
                    
                    # 0-1 arasındaki skoru etiketle
                    if sentiment_score >= 0.6:
                        sentiment_label = "Olumlu"
                    elif sentiment_score <= 0.4:
                        sentiment_label = "Olumsuz"
                    else:
                        sentiment_label = "Nötr"
                    
                    print(f"-----> Gelişmiş kelime analizi sonucu: {sentiment_label} ({sentiment_score:.2f})")
                    
                except Exception as e:
                    print(f"-----> Gelişmiş kelime analizi hatası: {e}")
                    # Yedek yönteme devam et
            
            # 2. Gelişmiş analiz başarısız olursa ve ImprovedSentimentAnalyzer varsa
            if (sentiment_score == 0.5 or sentiment_label == "Nötr") and improved_analyzer:
                try:
                    improved_result = improved_analyzer.analyze(content)
                    if improved_result and 'score' in improved_result:
                        sentiment_score = improved_result['score']
                        
                        # 0-1 arasındaki skoru etiketle
                        if sentiment_score >= 0.6:
                            sentiment_label = "Olumlu"
                        elif sentiment_score <= 0.4:
                            sentiment_label = "Olumsuz"
                        else:
                            sentiment_label = "Nötr"
                        
                        print(f"-----> Gelişmiş analiz sonucu: {sentiment_label} ({sentiment_score:.2f})")
                except Exception as e:
                    print(f"-----> Gelişmiş analiz hatası: {e}")
                    # Yedek yönteme devam et
            
            # 3. Son çare olarak yedek analizör kullan
            if (sentiment_score == 0.5 or sentiment_label == "Nötr") and backup_analyzer:
                try:
                    backup_result = backup_analyzer.analyze_text(content)
                    if backup_result and 'label' in backup_result:
                        label = backup_result['label']
                        backup_score = backup_result.get('score', 0.5)
                        
                        # SentimentAnalyzer'dan gelen POSITIVE/NEGATIVE/NEUTRAL etiketlerini dönüştür
                        if label == "POSITIVE":
                            sentiment_score = 0.5 + (backup_score * 0.5)  # 0.5-1.0 arası
                            sentiment_label = "Olumlu"
                        elif label == "NEGATIVE":
                            sentiment_score = 0.5 - (backup_score * 0.5)  # 0.0-0.5 arası
                            sentiment_label = "Olumsuz"
                        else:
                            sentiment_score = 0.5
                            sentiment_label = "Nötr"
                        
                        print(f"-----> Yedek analiz sonucu: {sentiment_label} ({sentiment_score:.2f})")
                except Exception as e:
                    print(f"-----> Yedek analiz hatası: {e}")
            
            # Analiz sonuçlarını kaydet
            sentiment_scores.append(sentiment_score)
            analyzed_news.append({
                'title': title,
                'link': link,
                'score': sentiment_score,
                'sentiment': sentiment_label
            })
            
            print(f"-----> Haber analiz edildi: {title}")
            
        except Exception as e:
            print(f"-----> Haber analiz edilirken hata: {e}")
            failed_count += 1
    
    # Özet sonuçları hesapla
    analyzed_count = len(sentiment_scores)
    
    if analyzed_count > 0:
        avg_score = sum(sentiment_scores) / analyzed_count
        
        # -1 ile 1 arasında normalize edilmiş skor
        normalized_score = (avg_score - 0.5) * 2
        
        # Genel duyarlılık etiketi
        if avg_score >= 0.6:
            overall_sentiment = "Olumlu"
        elif avg_score <= 0.4:
            overall_sentiment = "Olumsuz"
        else:
            overall_sentiment = "Nötr"
            
        print(f"[{stock_code}] Toplam {len(news_list)} haberden {analyzed_count} adedi başarıyla analiz edildi, {failed_count} adedi başarısız.")
        print(f"[{stock_code}] Ortalama Duyarlılık: {overall_sentiment} ({avg_score:.2f})")
        print(f"[{stock_code}] Normalize Edilmiş Skor: {normalized_score:.2f}")
        
        return {
            'score': avg_score,
            'normalized_score': normalized_score,
            'sentiment': overall_sentiment,
            'analyzed_count': analyzed_count,
            'failed_count': failed_count,
            'news_sentiments': analyzed_news
        }
    else:
        print(f"[{stock_code}] Haberler analiz edilemedi.")
        return {
            'score': 0.5,
            'normalized_score': 0.0,
            'sentiment': 'Nötr',
            'analyzed_count': 0,
            'failed_count': failed_count,
            'news_sentiments': []
        }

def analyze_stock_news(stock_code, max_news=5):
    """
    Bir hisse senedi için haberleri alıp analiz eder.
    
    Args:
        stock_code (str): Hisse kodu
        max_news (int, optional): Alınacak maksimum haber sayısı. Varsayılan 5.
        
    Returns:
        dict: Analiz sonuçları
    """
    # Haberleri getir
    news_list = get_news_for_stock(stock_code, max_news=max_news)
    
    # Haberleri analiz et
    result = analyze_news_sentiment(news_list, stock_code)
    
    return result

# Örnek kullanım ve test fonksiyonu
def test_stock_sentiment_analysis():
    """
    Duyarlılık analizi modülünü test eder.
    """
    print("Gelişmiş Haber Analizi Testi")
    print("==================================================")
    
    # Test için birkaç hisse analiz et
    test_stocks = ['GARAN.IS', 'THYAO.IS', 'ASELS.IS', 'SASA.IS']
    
    for stock in test_stocks:
        print(f"\n[{stock}] için haber analizi başlıyor...")
        result = analyze_stock_news(stock, max_news=3)
        
        # Analiz sonuçlarını yazdır
        if result['analyzed_count'] > 0:
            print(f"\nSentiment Analiz Sonucu: {result['sentiment']} (Skor: {result['score']:.2f}, Normalize: {result['normalized_score']:.2f})")
        else:
            print("\nYeterli haber analiz edilemedi.")
            
        print("==================================================")
    
# Bu dosya doğrudan çalıştırıldığında test fonksiyonunu çalıştır
if __name__ == "__main__":
    test_stock_sentiment_analysis() 