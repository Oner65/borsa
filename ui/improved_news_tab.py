import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time
import requests
from bs4 import BeautifulSoup
import os
import json
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("Transformers kütüphanesi bulunamadı. Duyarlılık analizi basit metotlar kullanılarak yapılacak.")
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
    # İlk çalıştırmada NLTK veri paketini indirme
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
except ImportError:
    NLTK_AVAILABLE = False
import traceback

# Kendi eğittiğimiz duyarlılık analizi modelini kullan
try:
    import sys
    import os
    # Proje kök dizinine eriş
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from ai.sentiment_analysis import SentimentAnalyzer
    SENTIMENT_ANALYZER_AVAILABLE = True
    # Sentiment analyzer örneği oluştur
    sentiment_analyzer = SentimentAnalyzer()
except Exception as e:
    SENTIMENT_ANALYZER_AVAILABLE = False
    st.error(f"Duyarlılık analizi modeli yüklenirken hata: {str(e)}")

# API anahtarı kontrolü
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")

# Log mesajları için yardımcı fonksiyon
def display_log_message(message, log_container=None, type="info"):
    import logging
    logger = logging.getLogger(__name__)
    
    if type == "error":
        logger.error(message)
        if log_container and getattr(log_container, "_expanded", False):
            log_container.error(message)
    elif type == "warning":
        logger.warning(message)
        if log_container and getattr(log_container, "_expanded", False):
            log_container.warning(message)
    else:
        logger.info(message)
        if log_container and getattr(log_container, "_expanded", False):
            log_container.info(message)

# Sentiment analiz modeli - global tanımlama ve lazy loading
@st.cache_resource
def load_sentiment_model():
    if not TRANSFORMERS_AVAILABLE:
        return lambda text: [{"label": "POSITIVE", "score": 0.5}]
        
    try:
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
        model = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased")
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {str(e)}")
        # Fallback olarak basit bir fonksiyon döndür
        return lambda text: [{"label": "POSITIVE", "score": 0.5}]

# Basit duyarlılık analizi - transformers olmadığında kullanılır
def simple_sentiment_analysis(text):
    """Basit kelime tabanlı duyarlılık analizi"""
    if not text:
        return {"label": "POSITIVE", "score": 0.5}
        
    # Türkçe olumlu ve olumsuz kelimeler
    positive_words = {
        'artış', 'yükseliş', 'kazanç', 'kâr', 'rekor', 'başarı', 'pozitif', 'olumlu', 'güçlü', 'büyüme',
        'iyileşme', 'yükseldi', 'arttı', 'çıktı', 'güven', 'istikrar', 'avantaj', 'fırsat', 'yatırım',
        'imzalandı', 'anlaşma', 'destek', 'teşvik', 'ivme', 'fayda', 'gelişti', 'yeni', 'ileri'
    }
    
    negative_words = {
        'düşüş', 'kayıp', 'zarar', 'risk', 'gerileme', 'olumsuz', 'negatif', 'zayıf', 'belirsizlik',
        'endişe', 'azaldı', 'düştü', 'kaybetti', 'gecikme', 'borç', 'iflas', 'kriz', 'tehdit', 'sorun',
        'başarısız', 'yaptırım', 'ceza', 'iptal', 'durgunluk'
    }
    
    # Metin içindeki kelimeleri kontrol et
    text = text.lower()
    words = text.split()
    
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    total = positive_count + negative_count
    if total == 0:
        return {"label": "POSITIVE", "score": 0.5}
    
    score = positive_count / total if total > 0 else 0.5
    label = "POSITIVE" if score >= 0.5 else "NEGATIVE"
    
    return {"label": label, "score": score if label == "POSITIVE" else 1 - score}

# Haber içeriği çekme - geliştirilmiş versiyon
def fetch_news_content(url, log_container=None):
    if not url or url == "#":
        display_log_message("Geçersiz URL", log_container, "warning")
        return None
        
    try:
        # Farklı User-Agent'lar deneyelim
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        display_log_message(f"İçerik çekiliyor: {url}", log_container)
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()  # HTTP hatalarını yakala
        except requests.exceptions.RequestException as e:
            display_log_message(f"İstek hatası: {str(e)}", log_container, "error")
            # Alternatif User-Agent ile tekrar dene
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15"
            }
            try:
                display_log_message("Alternatif User-Agent ile tekrar deneniyor...", log_container)
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                display_log_message(f"İkinci deneme başarısız: {str(e)}", log_container, "error")
                return None
            
        # İçeriği parse et
        try:
            soup = BeautifulSoup(response.content, "html.parser")
        except Exception as e:
            display_log_message(f"HTML ayrıştırma hatası: {str(e)}", log_container, "error")
            return None
        
        # Makale içeriğini bulmaya çalış - farklı yöntemler dene
        content = ""
        
        # Başlığı almaya çalış
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text().strip()
            display_log_message(f"Başlık bulundu: {title[:50]}...", log_container)
        
        # Yaygın makale içerik alanlarını kontrol et (genişletilmiş)
        article_selectors = [
            "article", "div.content", "div.article-body", "div.news-detail", "div.news-content",
            "div.post-content", "div.entry-content", "div.story-body", "div.article", 
            ".article__content", ".news-item", ".news-text", "#content-area", "#article-body",
            ".content-detail", ".article-text", ".news__content", ".article-container"
        ]
        
        for selector in article_selectors:
            try:
                article = soup.select_one(selector)
                if article:
                    # İçerik paragraflarını bul
                    paragraphs = article.find_all("p")
                    if paragraphs:
                        content = " ".join(p.get_text().strip() for p in paragraphs)
                        display_log_message(f"İçerik bulundu ('{selector}' seçicisiyle): {len(content)} karakter", log_container)
                        break
            except Exception as e:
                display_log_message(f"Seçici '{selector}' işlenirken hata: {str(e)}", log_container, "warning")
                continue
        
        # Hala içerik bulunamadıysa, tüm paragrafları dene
        if not content:
            display_log_message("Standart seçicilerle içerik bulunamadı, tüm paragraflar deneniyor...", log_container)
            paragraphs = soup.find_all("p")
            if paragraphs:
                # En az 30 karakterlik paragrafları al
                valid_paragraphs = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 30]
                if valid_paragraphs:
                    content = " ".join(valid_paragraphs)
                    display_log_message(f"Tüm paragraflardan içerik bulundu: {len(content)} karakter", log_container)
            
        # İçerik yoksa ve haber sitesi bilinen bir siteyse özel selectorlar dene
        if not content and "bloomberght.com" in url:
            display_log_message("BloombergHT için özel seçiciler deneniyor...", log_container)
            paragraphs = soup.select(".news-content p, .paywall-content p")
            if paragraphs:
                content = " ".join(p.get_text().strip() for p in paragraphs)
        
        if not content and "bigpara.com" in url:
            display_log_message("BigPara için özel seçiciler deneniyor...", log_container)
            paragraphs = soup.select(".detailText p")
            if paragraphs:
                content = " ".join(p.get_text().strip() for p in paragraphs)
        
        # İçerik yoksa
        if not content:
            display_log_message("İçerik bulunamadı, meta açıklaması deneniyor...", log_container, "warning")
            # Son çare: Meta açıklamasını al
            meta_desc = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
            if meta_desc:
                content = meta_desc.get("content", "")
                display_log_message(f"Meta açıklamasından içerik bulundu: {len(content)} karakter", log_container)
        
        # Hala içerik yoksa
        if not content:
            display_log_message("İçerik bulunamadı!", log_container, "error")
            return None
        
        # Tarih elementi
        publish_date = ""
        date_selectors = [
            'meta[property="article:published_time"]',
            'time', 
            'span.date', 
            'div.date',
            '.publish-date',
            '.news-date',
            '.article-date',
            'meta[property="og:published_time"]'
        ]
        
        for selector in date_selectors:
            try:
                date_element = soup.select_one(selector)
                if date_element:
                    if selector.startswith('meta'):
                        publish_date = date_element.get('content', '')
                    else:
                        publish_date = date_element.get_text().strip()
                    if publish_date:
                        display_log_message(f"Yayın tarihi bulundu: {publish_date}", log_container)
                        break
            except Exception:
                continue
        
        return {
            "title": title,
            "content": content,
            "publish_date": publish_date,
            "authors": "Belirtilmemiş"  # Yazarlar varsayılan olarak belirtilmedi
        }
                
    except Exception as e:
        display_log_message(f"İçerik çekerken hata: {str(e)}", log_container, "error")
        import traceback
        display_log_message(f"Hata detayı: {traceback.format_exc()}", log_container, "error")
        return None

# Metni özetleme fonksiyonu
def summarize_text(text, max_length=150):
    if not text:
        return "İçerik bulunamadı."
        
    # Metin çok kısaysa direkt döndür
    if len(text) <= max_length:
        return text
    
    if NLTK_AVAILABLE:
        try:
            # Metni cümlelere ayır
            sentences = sent_tokenize(text)
            
            # Metin çok uzunsa ilk birkaç cümleyi al
            summary = ""
            for sentence in sentences:
                if len(summary) + len(sentence) < max_length:
                    summary += sentence + " "
                else:
                    break
                    
            return summary.strip()
        except Exception:
            # NLTK ile özetleme başarısız olursa basit kesme kullan
            return text[:max_length] + "..."
    else:
        # NLTK yoksa basit kesme kullan
        return text[:max_length] + "..."

# Duyarlılık değerini yorumlama fonksiyonu
def get_sentiment_explanation(score):
    if score >= 0.7:
        return "Bu haber piyasa/şirket için oldukça olumlu içerik barındırıyor."
    elif score >= 0.55:
        return "Bu haber genel olarak olumlu bir ton taşıyor."
    elif score <= 0.3:
        return "Bu haber piyasa/şirket için olumsuz içerik barındırıyor."
    elif score <= 0.45:
        return "Bu haber hafif olumsuz bir tona sahip."
    else:
        return "Bu haber nötr bir tona sahip."

# Duyarlılık analizi fonksiyonu
def analyze_news(url, log_container=None):
    try:
        if not url or url == "#":
            display_log_message(f"Geçersiz URL: '{url}'", log_container, "warning")
            return {"success": False, "error": "Geçersiz URL"}
            
        # URL formatını kontrol et
        if not url.startswith(('http://', 'https://')):
            display_log_message(f"Geçersiz URL formatı: '{url}'", log_container, "warning")
            return {"success": False, "error": f"Geçersiz URL formatı: {url}"}
        
        display_log_message(f"Haber analiz ediliyor: {url}", log_container)
        
        # Haber içeriğini çek
        news_data = fetch_news_content(url, log_container)
        if not news_data:
            display_log_message(f"Haber içeriği çekilemedi: {url}", log_container, "warning")
            return {"success": False, "error": "Haber içeriği çekilemedi"}
            
        if not news_data.get("content"):
            display_log_message(f"Haber içeriği boş: {url}", log_container, "warning") 
            return {"success": False, "error": "Haber içeriği boş"}
        
        content = news_data.get("content")
        display_log_message(f"İçerik uzunluğu: {len(content)} karakter", log_container)
        
        # İçerik çok uzunsa modelin işleyebileceği boyuta getir
        if len(content) > 500:
            # Analiz için ilk 500 karakter
            analysis_content = content[:500]
            display_log_message(f"İçerik kısaltıldı: {len(analysis_content)} karakter", log_container)
        else:
            analysis_content = content
            
        # Özet oluştur
        summary = summarize_text(content)
        
        display_log_message("Duyarlılık analizi yapılıyor...", log_container)
            
        # Duyarlılık analizi yap - önce kendi modelimizi dene
        if SENTIMENT_ANALYZER_AVAILABLE:
            try:
                display_log_message("Özel eğitilmiş model kullanılıyor...", log_container)
                # scikit-learn modeli kullan
                prediction = sentiment_analyzer.predict([analysis_content])[0]
                probabilities = sentiment_analyzer.predict_proba([analysis_content])[0]
                
                # 0 (negatif) veya 1 (pozitif) - bunu "NEGATIVE" veya "POSITIVE" formatına dönüştür
                label = "POSITIVE" if prediction == 1 else "NEGATIVE"
                
                # -1 ile 1 arasında değer döndürür, bunu 0-1 arasına normalize et
                score = (probabilities + 1) / 2 if label == "POSITIVE" else 1 - ((probabilities + 1) / 2)
                
                result = {"label": label, "score": score}
                display_log_message(f"Duyarlılık analiz sonucu: {result}", log_container)
            except Exception as e:
                display_log_message(f"Özel model hatası: {str(e)}", log_container, "error")
                display_log_message(traceback.format_exc(), log_container, "error")
                # Hata durumunda transformers veya basit analiz yap
                if TRANSFORMERS_AVAILABLE:
                    try:
                        sentiment_model = load_sentiment_model()
                        display_log_message(f"TransformersAPI kullanılıyor...", log_container)
                        result = sentiment_model(analysis_content)[0]
                        display_log_message(f"TransformersAPI sonucu: {result}", log_container)
                    except Exception as e:
                        display_log_message(f"Transformers hatası: {str(e)}", log_container, "error")
                        # Hata durumunda basit analiz yap
                        display_log_message("Basit duyarlılık analizi kullanılıyor...", log_container)
                        result = simple_sentiment_analysis(analysis_content)
                        display_log_message(f"Basit analiz sonucu: {result}", log_container)
                else:
                    # Transformers yoksa basit analiz yap
                    display_log_message("Basit duyarlılık analizi kullanılıyor...", log_container)
                    result = simple_sentiment_analysis(analysis_content)
                    display_log_message(f"Basit analiz sonucu: {result}", log_container)
        elif TRANSFORMERS_AVAILABLE:
            try:
                sentiment_model = load_sentiment_model()
                display_log_message(f"TransformersAPI kullanılıyor...", log_container)
                result = sentiment_model(analysis_content)[0]
                display_log_message(f"TransformersAPI sonucu: {result}", log_container)
            except Exception as e:
                display_log_message(f"Transformers hatası: {str(e)}", log_container, "error")
                # Hata durumunda basit analiz yap
                display_log_message("Basit duyarlılık analizi kullanılıyor...", log_container)
                result = simple_sentiment_analysis(analysis_content)
                display_log_message(f"Basit analiz sonucu: {result}", log_container)
        else:
            # Transformers yoksa basit analiz yap
            display_log_message("Basit duyarlılık analizi kullanılıyor...", log_container)
            result = simple_sentiment_analysis(analysis_content)
            display_log_message(f"Basit analiz sonucu: {result}", log_container)
        
        # Sonucu formatla
        label = result.get("label", "")
        score = result.get("score", 0.5)
        
        # Açıklama oluştur
        explanation = get_sentiment_explanation(score)
        
        return {
            "success": True,
            "title": news_data.get("title", ""),
            "content": content,
            "summary": summary,
            "authors": news_data.get("authors", "Belirtilmemiş"),
            "sentiment": {
                "label": label,
                "score": score,
                "explanation": explanation
            },
            "publish_date": news_data.get("publish_date", "")
        }
        
    except Exception as e:
        display_log_message(f"Haber analiz edilirken hata: {str(e)}", log_container, "error")
        import traceback
        display_log_message(f"Hata detayı: {traceback.format_exc()}", log_container, "error")
        return {"success": False, "error": f"Analiz hatası: {str(e)}"}

# NewsAPI alternatifi - farklı kaynaklardan haber toplama
def get_stock_news(ticker, num_news=5, max_days=30, debug=False):
    """
    Belirli bir hisse senedi için haberleri getiren fonksiyon.
    
    Args:
        ticker (str): Hisse senedi sembolü (örn. 'THYAO')
        num_news (int): Getirilecek haber sayısı
        max_days (int): Haberlerin maksimum kaç gün öncesine ait olabileceği
        debug (bool): Hata ayıklama mesajlarının gösterilip gösterilmeyeceği
        
    Returns:
        dict: Aşağıdaki anahtarları içeren sözlük:
            - success (bool): İşlemin başarılı olup olmadığı
            - message (str): İşlem sonucu mesajı
            - data (list): Haber verileri
            - logs (list): İşlem sırasında oluşan loglar
    """
    import time
    from datetime import datetime, timedelta
    import random
    import re
    from bs4 import BeautifulSoup
    
    # Yardımcı işlevler
    def log_message(message, level="INFO"):
        """Log mesajı oluşturur"""
        logs.append(f"[{level}] {message}")
        if debug:
            print(f"[{level}] {message}")
            
    # Ticker-şirket ismi eşleştirmeleri
    ticker_to_company = {
        "THYAO": "Türk Hava Yolları",
        "TUPRS": "Tüpraş",
        "ASELS": "Aselsan",
        "GARAN": "Garanti Bankası",
        "EREGL": "Ereğli Demir Çelik",
        "AKBNK": "Akbank",
        "KCHOL": "Koç Holding",
        "SAHOL": "Sabancı Holding",
        "YKBNK": "Yapı Kredi Bankası",
        "BIMAS": "BİM"
    }
    
    # HTML temizleme işlevi
    def clean_html(html_text):
        """HTML içeriğini düz metne dönüştürür"""
        if not html_text:
            return ""
        try:
            # HTML etiketlerini kaldır
            soup = BeautifulSoup(html_text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            # Fazla boşlukları temizle
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except Exception as e:
            log_message(f"HTML temizleme hatası: {str(e)}", "ERROR")
            return html_text
    
    # Sonuç ve log değişkenleri
    result = {"success": False, "message": "", "data": [], "logs": []}
    logs = []
    news_data = []
    
    try:
        # Google News provider'ını başlat
        log_message(f"{ticker} için Google News başlatılıyor...")
        google_news = GoogleNews()
        
        # Şirket adını bul veya ticker'ı kullan
        company_name = ticker_to_company.get(ticker, ticker)
        search_term = f"{company_name} hisse"
        
        log_message(f"Arama terimi: {search_term}")
        
        # Google News ile haber ara
        news_items = google_news.search(search_term, max_results=15)
        
        # Yetersiz sonuç varsa alternatif arama terimleri dene
        if len(news_items) < 3:
            log_message(f"Yetersiz sonuç ({len(news_items)}), alternatif arama yapılıyor...")
            alt_terms = [f"{ticker} borsa", "ekonomi borsa", "piyasa analiz"]
            
            for term in alt_terms:
                log_message(f"Alternatif arama terimi: {term}")
                alt_items = google_news.search(term, max_results=10)
                if alt_items:
                    news_items.extend(alt_items)
                    if len(news_items) >= 10:
                        break
        
        log_message(f"Toplam {len(news_items)} haber bulundu, işleniyor...")
        
        # Haberleri işle
        for item in news_items:
            try:
                # Gerekli bilgileri çıkar
                title = item.get('title', '')
                link = item.get('link', '')
                desc = item.get('desc', '')
                site = item.get('site', '')
                date_str = item.get('datetime', '')
                
                # Boş alanları kontrol et
                if not title or not link:
                    continue
                
                # Açıklama boşsa başlığı kullan
                if not desc:
                    desc = title
                else:
                    desc = clean_html(desc)
                
                # Haber kaynağı kontrolü
                if not site:
                    # URL'den site adını çıkarmaya çalış
                    match = re.search(r'https?://(?:www\.)?([^/]+)', link)
                    if match:
                        site = match.group(1)
                
                # Haber öğesini oluştur
                news_item = {
                    "title": title,
                    "link": link,
                    "description": desc[:300] + ('...' if len(desc) > 300 else ''),
                    "source": site,
                    "date": date_str,
                    "sentiment_score": 0,
                    "sentiment_label": "Nötr"
                }
                
                # Duygu analizi yap
                try:
                    from ai.sentiment_analyzer import SentimentAnalyzer
                    
                    analyzer = SentimentAnalyzer()
                    text_to_analyze = f"{title} {desc}"
                    
                    # Model kullanılamadığında basit anahtar kelime tabanlı duygu analizi
                    text = f"{title} {desc}"
                    
                    # Olumlu ve olumsuz kelimeler
                    positive_words = [
                        # Genel olumlu terimler
                        "artış", "yükseldi", "yükseliş", "kazanç", "başarı", "olumlu", "güçlü", 
                        "kar", "büyüme", "yatırım", "fırsat", "rekor", "güven", "avantaj",
                        # Hisse senedi ve finans ile ilgili olumlu terimler
                        "alım", "geri alım", "pay geri alım", "hedef fiyat", "yukarı yönlü", "artırıldı", 
                        "yükseltildi", "ivme", "güçleniyor", "zirve", "tavan", "prim", "aşırı alım", 
                        "güçlü performans", "kârlılık", "temettü", "beklentilerin üzerinde", "kapasite artışı",
                        "lider", "pazar payı", "büyüdü", "arttı", "genişleme", "ihracat", "yeni anlaşma",
                        "ortaklık", "işbirliği", "strateji", "dijitalleşme", "teknoloji", "dev", "program",
                        "iş hacmi", "rağbet", "talep", "ihale", "kazandı", "başarıyla", "gelir artışı"
                    ]
                    
                    negative_words = [
                        # Genel olumsuz terimler
                        "düşüş", "geriledi", "azaldı", "zarar", "kayıp", "olumsuz", "zayıf", 
                        "risk", "endişe", "kriz", "tehlike", "yavaşlama", "dezavantaj",
                        # Hisse senedi ve finans ile ilgili olumsuz terimler
                        "satış baskısı", "değer kaybı", "daraldı", "daralma", "borç", "iflas", "konkordato",
                        "aşağı yönlü", "indirildi", "düşürüldü", "aşırı satım", "volatilite", "zayıf performans",
                        "beklentilerin altında", "ertelendi", "iptal", "durgunluk", "negatif", "dibe", "dip",
                        "indirim", "faiz artışı", "vergi", "ceza", "yaptırım", "manipülasyon", "soruşturma",
                        "dava", "para cezası", "şikayet", "protesto", "grev", "maliyet artışı", "fire"
                    ]
                    
                    # Metin özel durumları ele al - finans haberlerinde bazı ifadeler özel anlam taşır
                    special_cases = [
                        {"phrase": "pay geri alım", "score": 1.0},
                        {"phrase": "hisse geri alım", "score": 1.0},
                        {"phrase": "hedef fiyat", "score": 0.8},
                        {"phrase": "tavan fiyat", "score": 0.7},
                        {"phrase": "ek sefer", "score": 0.7},
                        {"phrase": "yatırım tavsiyesi", "score": 0.6},
                        {"phrase": "al tavsiyesi", "score": 0.9},
                        {"phrase": "tut tavsiyesi", "score": 0.6},
                        {"phrase": "sat tavsiyesi", "score": 0.2}
                    ]
                    
                    # Özel durumları kontrol et
                    special_score = None
                    for case in special_cases:
                        if case["phrase"].lower() in text.lower():
                            special_score = case["score"]
                            log_message(f"Özel durum tespit edildi: '{case['phrase']}', skor: {special_score}", "INFO")
                            break
                    
                    # Eğer özel durum varsa, doğrudan skoru ata
                    if special_score is not None:
                        score = special_score
                        label = "Olumlu" if score > 0.5 else ("Nötr" if score == 0.5 else "Olumsuz")
                    else:
                        # Kelime sayaçları
                        positive_count = sum(1 for word in positive_words if word.lower() in text.lower())
                        negative_count = sum(1 for word in negative_words if word.lower() in text.lower())
                        
                        # Duygu skoru hesapla (0 ile 1 arasında)
                        total = positive_count + negative_count
                        if total > 0:
                            # Pozitif sayısı ağırlıklıysa skoru yükselt
                            score = positive_count / (positive_count + negative_count)
                        else:
                            score = 0.5  # Nötr değer
                        
                        # Etiket belirle
                        if score > 0.6:
                            label = "Olumlu"
                        elif score < 0.4:
                            label = "Olumsuz"
                        else:
                            label = "Nötr"
                    
                    news_item["sentiment_score"] = score
                    news_item["sentiment_label"] = label
                
                except Exception as e:
                    log_message(f"Model tabanlı duygu analizi başarısız: {str(e)}", "WARNING")
                    
                    # Model kullanılamadığında basit anahtar kelime tabanlı duygu analizi
                    text = f"{title} {desc}"
                    
                    # Olumlu ve olumsuz kelimeler
                    positive_words = [
                        # Genel olumlu terimler
                        "artış", "yükseldi", "yükseliş", "kazanç", "başarı", "olumlu", "güçlü", 
                        "kar", "büyüme", "yatırım", "fırsat", "rekor", "güven", "avantaj",
                        # Hisse senedi ve finans ile ilgili olumlu terimler
                        "alım", "geri alım", "pay geri alım", "hedef fiyat", "yukarı yönlü", "artırıldı", 
                        "yükseltildi", "ivme", "güçleniyor", "zirve", "tavan", "prim", "aşırı alım", 
                        "güçlü performans", "kârlılık", "temettü", "beklentilerin üzerinde", "kapasite artışı",
                        "lider", "pazar payı", "büyüdü", "arttı", "genişleme", "ihracat", "yeni anlaşma",
                        "ortaklık", "işbirliği", "strateji", "dijitalleşme", "teknoloji", "dev", "program",
                        "iş hacmi", "rağbet", "talep", "ihale", "kazandı", "başarıyla", "gelir artışı"
                    ]
                    
                    negative_words = [
                        # Genel olumsuz terimler
                        "düşüş", "geriledi", "azaldı", "zarar", "kayıp", "olumsuz", "zayıf", 
                        "risk", "endişe", "kriz", "tehlike", "yavaşlama", "dezavantaj",
                        # Hisse senedi ve finans ile ilgili olumsuz terimler
                        "satış baskısı", "değer kaybı", "daraldı", "daralma", "borç", "iflas", "konkordato",
                        "aşağı yönlü", "indirildi", "düşürüldü", "aşırı satım", "volatilite", "zayıf performans",
                        "beklentilerin altında", "ertelendi", "iptal", "durgunluk", "negatif", "dibe", "dip",
                        "indirim", "faiz artışı", "vergi", "ceza", "yaptırım", "manipülasyon", "soruşturma",
                        "dava", "para cezası", "şikayet", "protesto", "grev", "maliyet artışı", "fire"
                    ]
                    
                    # Metin özel durumları ele al - finans haberlerinde bazı ifadeler özel anlam taşır
                    special_cases = [
                        {"phrase": "pay geri alım", "score": 1.0},
                        {"phrase": "hisse geri alım", "score": 1.0},
                        {"phrase": "hedef fiyat", "score": 0.8},
                        {"phrase": "tavan fiyat", "score": 0.7},
                        {"phrase": "ek sefer", "score": 0.7},
                        {"phrase": "yatırım tavsiyesi", "score": 0.6},
                        {"phrase": "al tavsiyesi", "score": 0.9},
                        {"phrase": "tut tavsiyesi", "score": 0.6},
                        {"phrase": "sat tavsiyesi", "score": 0.2}
                    ]
                    
                    # Özel durumları kontrol et
                    special_score = None
                    for case in special_cases:
                        if case["phrase"].lower() in text.lower():
                            special_score = case["score"]
                            log_message(f"Özel durum tespit edildi: '{case['phrase']}', skor: {special_score}", "INFO")
                            break
                    
                    # Eğer özel durum varsa, doğrudan skoru ata
                    if special_score is not None:
                        score = special_score
                        label = "Olumlu" if score > 0.5 else ("Nötr" if score == 0.5 else "Olumsuz")
                    else:
                        # Kelime sayaçları
                        positive_count = sum(1 for word in positive_words if word.lower() in text.lower())
                        negative_count = sum(1 for word in negative_words if word.lower() in text.lower())
                        
                        # Duygu skoru hesapla (0 ile 1 arasında)
                        total = positive_count + negative_count
                        if total > 0:
                            # Pozitif sayısı ağırlıklıysa skoru yükselt
                            score = positive_count / (positive_count + negative_count)
                        else:
                            score = 0.5  # Nötr değer
                        
                        # Etiket belirle
                        if score > 0.6:
                            label = "Olumlu"
                        elif score < 0.4:
                            label = "Olumsuz"
                        else:
                            label = "Nötr"
                    
                    news_item["sentiment_score"] = score
                    news_item["sentiment_label"] = label
                
                # Haber verisini listeye ekle
                news_data.append(news_item)
            
            except Exception as e:
                log_message(f"Haber öğesi işleme hatası: {str(e)}", "ERROR")
                continue
        
        # Haberleri duygu skoruna göre sırala (olumlu haberler önce)
        news_data.sort(key=lambda x: x["sentiment_score"], reverse=True)
        
        # Belirtilen sayıda haberi al
        if len(news_data) > num_news:
            news_data = news_data[:num_news]
        
        # Sonuç oluştur
        result = {
            "success": True,
            "message": f"{ticker} için {len(news_data)} haber bulundu.",
            "data": news_data,
            "logs": logs
        }
        
    except Exception as e:
        log_message(f"Genel hata: {str(e)}", "ERROR")
        result = {
            "success": False,
            "message": f"Haber getirme hatası: {str(e)}",
            "data": [],
            "logs": logs
        }
    
    return result

# Streamlit arayüzü
def render_stock_news_tab():
    st.title("Hisse Senedi Haberleri 📰")
    
    # Session state yönetimi
    if 'show_news_analysis' not in st.session_state:
        st.session_state.show_news_analysis = False
        st.session_state.news_url = ""
        st.session_state.news_analysis_results = None
    
    if 'analyzed_news_ids' not in st.session_state:
        st.session_state.analyzed_news_ids = []
    
    # İşlem günlüğü
    log_expander = st.expander("İşlem Günlüğü (Detaylar için tıklayın)", expanded=False)
    
    # Eğer analiz gösterilmesi gerekiyorsa
    if st.session_state.show_news_analysis and st.session_state.news_url:
        with st.spinner("Haber analiz ediliyor..."):
            if st.session_state.news_analysis_results is None:
                display_log_message("Haber analizi başlatılıyor...", log_expander)
                analysis_results = analyze_news(st.session_state.news_url, log_expander)
                st.session_state.news_analysis_results = analysis_results
            else:
                analysis_results = st.session_state.news_analysis_results
            
            # Analiz sonuçlarını göster
            if analysis_results.get("success", False):
                # Haber içeriği için container
                with st.expander("Haber İçeriği", expanded=True):
                    st.markdown(f"## {analysis_results['title']}")
                    st.markdown(f"**Yazar:** {analysis_results['authors']} | **Tarih:** {analysis_results['publish_date']}")
                    st.markdown("---")
                    st.markdown(analysis_results['content'])
                
                # Analiz sonuçları
                st.subheader("Yapay Zeka Analizi")
                
                # Duyarlılık analizi
                sentiment = analysis_results['sentiment']['label']
                sentiment_score = analysis_results['sentiment']['score']
                sentiment_color = "green" if sentiment == "POSITIVE" else ("red" if sentiment == "NEGATIVE" else "gray")
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Duyarlılık", sentiment, f"{sentiment_score:.2f}")
                
                with col2:
                    st.markdown(f"""
                    <div style="border-left:5px solid {sentiment_color}; padding-left:15px; margin-top:10px;">
                    <h4>Haber Özeti</h4>
                    <p>{analysis_results['summary']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Duyarlılık açıklaması
                    st.markdown(f"""
                    <div style="margin-top:10px; padding:10px; background-color:#f8f9fa; border-radius:5px;">
                    <p><strong>Yorum:</strong> {analysis_results['sentiment']['explanation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Kapat butonu
                if st.button("Analizi Kapat", key="close_analysis"):
                    st.session_state.show_news_analysis = False
                    st.session_state.news_url = ""
                    st.session_state.news_analysis_results = None
                    st.experimental_rerun()
            else:
                st.error(f"Haber analizi yapılamadı: {analysis_results.get('error', 'Bilinmeyen hata')}")
                if st.button("Geri Dön"):
                    st.session_state.show_news_analysis = False
                    st.session_state.news_url = ""
                    st.session_state.news_analysis_results = None
                    st.experimental_rerun()
        
        # Analiz modu aktifse, diğer içerikleri gösterme
        return
    
    # Normal haber arama arayüzü
    col1, col2, col3 = st.columns([3, 1.5, 1])
    
    with col1:
        stock_symbol = st.text_input("Hisse Senedi Kodu (örn: THYAO)", "THYAO", key="news_stock_symbol")
    
    with col2:
        # Haber kaynakları seçimi
        available_providers = ["Google News", "Yahoo Finance"]
        selected_providers = st.multiselect(
            "Haber Kaynakları", 
            available_providers,
            default=available_providers,
            key="news_providers"
        )
    
    with col3:
        max_results = st.selectbox("Maksimum Haber", [5, 10, 15, 20], index=1)
        search_btn = st.button("Haberleri Getir")
    
    # Sonuçlar için container
    results_container = st.container()
    
    if search_btn or ('news_last_symbol' in st.session_state and st.session_state.news_last_symbol == stock_symbol):
        stock_symbol = stock_symbol.upper().strip()
        st.session_state.news_last_symbol = stock_symbol
        
        display_log_message(f"{stock_symbol} ile ilgili haberler aranıyor...", log_expander)
        
        with results_container:
            try:
                # En az bir kaynak seçilmemişse uyarı ver
                if not selected_providers:
                    st.warning("Lütfen en az bir haber kaynağı seçin.")
                    return
                
                # Haberleri getir
                display_log_message("Haberler getiriliyor...", log_expander)
                result = get_stock_news(stock_symbol, num_news=max_results)
                
                # Haber sonuçlarını göster
                if isinstance(result, dict) and "data" in result:
                    news_list = result["data"]
                    display_log_message(f"get_stock_news fonksiyonu tarafından döndürülen haber sayısı: {len(news_list)}", log_expander)
                    
                    if news_list and len(news_list) > 0:
                        # Gerekli alanları standardize et
                        standardized_news_list = []
                        for news_item in news_list:
                            # Eksik alanları varsayılan değerlerle doldur
                            standardized_item = {
                                "title": news_item.get("title", "Başlık Bulunamadı"),
                                "url": news_item.get("link", "#"),  # "link" anahtarı kullanılıyor
                                "source": news_item.get("source", "Bilinmeyen Kaynak"),
                                "pub_date": news_item.get("date", ""),
                                "summary": news_item.get("description", "Bu haber için özet bulunmuyor."),  # "description" anahtarı kullanılıyor
                                "sentiment": news_item.get("sentiment_score", 0.5),  # "sentiment_score" anahtarı kullanılıyor
                                "image_url": news_item.get("image_url", "")
                            }
                            
                            standardized_news_list.append(standardized_item)
                            
                            # Log için haber bilgilerini göster
                            display_log_message(f"Haber: {standardized_item['title']}, Kaynak: {standardized_item['source']}", log_expander)
                        
                        # Liste olarak dönen haberleri DataFrame'e dönüştür
                        news_df = pd.DataFrame(standardized_news_list)
                        
                        # Tekrarlı haberleri kaldır
                        news_df = news_df.drop_duplicates(subset=['title'], keep='first')
                        
                        display_log_message(f"{len(news_df)} haber bulundu", log_expander)
                        
                        # DataFrame'in sütunlarını kontrol et ve görüntüle
                        display_log_message(f"DataFrame sütunları: {list(news_df.columns)}", log_expander)
                        
                        # Haberleri göster
                        st.subheader(f"{stock_symbol} ile İlgili Haberler")
                        
                        # Her haberi kart olarak göster
                        for idx, news in news_df.iterrows():
                            # Duyarlılık rengi ve etiketi
                            sentiment_value = float(news["sentiment"])
                            if sentiment_value > 0.65:
                                sentiment_color = "#4CAF50"  # green
                                sentiment_label = "Olumlu"
                            elif sentiment_value < 0.35:
                                sentiment_color = "#F44336"  # red
                                sentiment_label = "Olumsuz"
                            else:
                                sentiment_color = "#FF9800"  # amber
                                sentiment_label = "Nötr"
                            
                            # Tarih formatı
                            try:
                                pub_date = pd.to_datetime(news["pub_date"]).strftime("%d.%m.%Y %H:%M") 
                            except:
                                pub_date = "Tarih bilinmiyor"
                            
                            # Özet
                            summary = news["summary"] if not pd.isna(news["summary"]) and news["summary"] != "Özet alınamadı." else "Bu haber için özet bulunmuyor."
                            
                            # Haber kartı
                            cols = st.columns([3, 1])
                            with cols[0]:
                                st.markdown(f"### {news['title']}")
                                st.markdown(f"""
                                <div style="font-size:0.85rem; color:#666; margin-bottom:8px;">
                                    <span style="background-color:#f0f0f0; padding:3px 6px; border-radius:3px; margin-right:8px;">
                                        <strong>{news['source']}</strong>
                                    </span>
                                    <span style="color:#888;">
                                        {pub_date}
                                    </span>
                                </div>
                                """, unsafe_allow_html=True)
                                st.markdown(f"""
                                <div style="padding: 10px; border-left: 3px solid #2196F3; background-color: #f8f9fa; margin-bottom: 10px;">
                                    {summary}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Haberi oku butonu
                                st.markdown(f"""
                                <a href="{news['url']}" target="_blank" style="text-decoration:none;">
                                    <div style="display:inline-block; padding:5px 15px; background-color:#2196F3; color:white; 
                                                border-radius:4px; font-weight:bold; text-align:center; margin-top:5px;">
                                        Haberi Oku
                                    </div>
                                </a>
                                """, unsafe_allow_html=True)
                                
                                # Analiz et butonu
                                if st.button(f"Haberi Analiz Et", key=f"analyze_{idx}"):
                                    st.session_state.show_news_analysis = True
                                    st.session_state.news_url = news['url']
                                    st.session_state.news_analysis_results = None
                                    st.experimental_rerun()
                            
                            with cols[1]:
                                # Duyarlılık skoru
                                st.markdown(f"""
                                <div style="background-color:{sentiment_color}; color:white; padding:10px; border-radius:8px; 
                                          text-align:center; box-shadow:0 2px 5px rgba(0,0,0,0.1);">
                                    <h4 style="margin:0; font-size:1.2rem;">{sentiment_label}</h4>
                                    <p style="margin:0; font-size:1.5rem; font-weight:bold;">{sentiment_value:.2f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Resim varsa göster
                                if not pd.isna(news.get("image_url")) and news["image_url"] != "":
                                    st.image(news["image_url"], use_column_width=True)
                            
                            st.markdown("---")
                    else:
                        display_log_message("news_list boş veya None dönüyor", log_expander, "warning")
                        st.warning("Haber bulunamadı. Farklı bir hisse kodu deneyin veya daha sonra tekrar deneyin.")
            
            except Exception as e:
                st.error(f"Haber arama sırasında bir hata oluştu: {str(e)}")
                display_log_message(f"Hata: {str(e)}", log_expander, "error")

# Basit GoogleNews sınıfı
class GoogleNews:
    """
    Google Haberler'den RSS beslemesi aracılığıyla haber toplama sınıfı.
    
    Türkçe Google Haberler için optimize edilmiştir ve hisse senedi haberleri
    aramak için kullanılır.
    """
    
    def __init__(self, language='tr', region='TR'):
        """
        GoogleNews nesnesini başlatır.
        
        Args:
            language (str): Haber dili (varsayılan: 'tr' - Türkçe)
            region (str): Bölge kodu (varsayılan: 'TR' - Türkiye)
        """
        import warnings
        # RSS parse ederken çıkan bazı uyarıları bastır
        warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
        
        self.language = language.lower()
        self.region = region.upper()
        self.base_url = f"https://news.google.com/rss/search"
        
    def search(self, query, max_results=15):
        """
        Google Haberler'de arama yapar ve sonuçları listeler.
        
        Args:
            query (str): Arama sorgusu
            max_results (int): Maksimum sonuç sayısı (varsayılan: 15)
            
        Returns:
            list: Aşağıdaki anahtarları içeren sözlük listesi:
                - title (str): Haber başlığı
                - link (str): Haber bağlantısı
                - desc (str): Açıklama veya özet
                - site (str): Haber kaynağı
                - datetime (str): Yayın tarihi
        """
        import requests
        from bs4 import BeautifulSoup
        from datetime import datetime
        import re
        import time
        import xml.etree.ElementTree as ET
        
        # Sonuçları saklamak için liste
        results = []
        
        try:
            # Arama sorgusunu URL'ye ekle
            encoded_query = requests.utils.quote(query)
            url = f"{self.base_url}?q={encoded_query}&hl={self.language}&gl={self.region}&ceid={self.region}:{self.language}"
            
            # RSS beslemesini al
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept-Language': f'{self.language},{self.language}-{self.region};q=0.9'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            
            # Yanıt başarılı değilse hata fırlat
            if response.status_code != 200:
                raise Exception(f"HTTP Hatası: {response.status_code}")
            
            items = []
            content = response.content
            
            # Parsing denemelerini sırayla gerçekleştir
            # 1. XML parser ile dene (lxml)
            try:
                soup = BeautifulSoup(content, 'lxml-xml')
                items = soup.find_all('item')
            except Exception:
                pass
                
            # 2. HTML parser ile dene
            if not items:
                try:
                    soup = BeautifulSoup(content, 'lxml')
                    items = soup.find_all('item')
                except Exception:
                    pass
                    
            # 3. Fallback: ElementTree ile dene
            if not items:
                try:
                    root = ET.fromstring(content)
                    # ElementTree ile item elementlerini bul
                    items_et = root.findall('.//item')
                    
                    # ElementTree elementlerini BeautifulSoup nesnelerine dönüştür
                    if items_et:
                        items = []
                        for item_et in items_et:
                            item_str = ET.tostring(item_et, encoding='unicode')
                            items.append(BeautifulSoup(item_str, 'lxml-xml').find('item'))
                except Exception:
                    pass
                    
            # 4. Son çare: regex ile dene
            if not items:
                try:
                    pattern = r'<item>(.*?)<\/item>'
                    items_raw = re.findall(pattern, response.text, re.DOTALL)
                    items = []
                    for item_raw in items_raw:
                        try:
                            bs_item = BeautifulSoup(f"<item>{item_raw}</item>", 'lxml-xml')
                            if bs_item and bs_item.find('item'):
                                items.append(bs_item.find('item'))
                        except Exception:
                            continue
                except Exception:
                    pass
            
            # İşlenmiş URL'leri takip et (çift kayıtları önlemek için)
            processed_urls = set()
            
            # Her öğeyi işle
            for item in items[:max_results]:
                try:
                    # Temel bilgileri çıkar
                    title = ""
                    site = ""
                    
                    title_elem = item.find('title')
                    if title_elem and title_elem.text:
                        title_text = title_elem.text.strip()
                        
                        # Google News başlık formatı: "Başlık - Kaynak"
                        if " - " in title_text:
                            title_parts = title_text.rsplit(" - ", 1)
                            if len(title_parts) == 2:
                                title = title_parts[0].strip()
                                site = title_parts[1].strip()
                        else:
                            title = title_text
                            # Kaynak bilgisini <source> etiketinden almayı dene
                            source_elem = item.find('source')
                            if source_elem and source_elem.text:
                                site = source_elem.text.strip()
                    
                    # Bağlantıyı al
                    link = ""
                    
                    # Önce <link> etiketi içeriğine bak
                    link_elem = item.find('link')
                    if link_elem:
                        if link_elem.text and link_elem.text.strip():
                            link = link_elem.text.strip()
                        elif link_elem.get('href'):
                            link = link_elem.get('href').strip()
                    
                    # <link> bulunamadıysa <guid> etiketine bak
                    if not link:
                        guid_elem = item.find('guid')
                        if guid_elem and guid_elem.text:
                            link = guid_elem.text.strip()
                    
                    # <guid> bulunamadıysa, <enclosure> etiketine bak
                    if not link:
                        enclosure_elem = item.find('enclosure')
                        if enclosure_elem and enclosure_elem.get('url'):
                            link = enclosure_elem.get('url').strip()
                    
                    # URL'yi temizle
                    if link and link.startswith("https://news.google.com/articles/"):
                        # Google News yönlendirme URL'sini temizle
                        link = re.sub(r'\?.*$', '', link)
                    
                    # URL tekrarını kontrol et veya geçerli bir URL yoksa atla
                    if not link or link in processed_urls:
                        continue
                    processed_urls.add(link)
                    
                    # Açıklamayı al (description veya summary)
                    desc = ""
                    desc_elem = item.find('description') or item.find('summary') or item.find('content')
                    if desc_elem and desc_elem.text:
                        desc = desc_elem.text.strip()
                        # HTML etiketlerini temizle
                        desc = re.sub(r'<[^>]*>', '', desc)
                    
                    # Yayın tarihini al
                    pub_date_str = ""
                    pub_date_elem = item.find('pubDate') or item.find('published') or item.find('updated')
                    if pub_date_elem and pub_date_elem.text:
                        pub_date_str = pub_date_elem.text.strip()
                    
                    # Tarih formatını düzenle
                    datetime_str = ""
                    if pub_date_str:
                        try:
                            # Tarihi standartlaştır
                            pub_date_str = pub_date_str.replace('GMT', '+0000')
                            dt = None
                            
                            # Farklı tarih formatlarını dene
                            date_formats = [
                                "%a, %d %b %Y %H:%M:%S %z",  # RFC 2822
                                "%Y-%m-%dT%H:%M:%S%z",       # ISO 8601
                                "%Y-%m-%dT%H:%M:%SZ",        # ISO 8601 (Z)
                                "%Y-%m-%d %H:%M:%S",         # Basic format
                                "%a, %d %b %Y %H:%M:%S"      # RFC 2822 without timezone
                            ]
                            
                            for fmt in date_formats:
                                try:
                                    dt = datetime.strptime(pub_date_str, fmt)
                                    break
                                except ValueError:
                                    continue
                            
                            if dt:
                                # Kullanıcı dostu biçime çevir
                                datetime_str = dt.strftime("%d.%m.%Y %H:%M")
                            else:
                                datetime_str = pub_date_str
                        except Exception:
                            datetime_str = pub_date_str
                    
                    # Sonuçlara ekle
                    if title and link:  # Hem başlık hem bağlantı varsa ekle
                        results.append({
                            'title': title,
                            'link': link,
                            'desc': desc,
                            'site': site,
                            'datetime': datetime_str
                        })
                    
                except Exception as e:
                    print(f"Haber öğesi işlenirken hata: {str(e)}")
                    continue
            
            # Sonuç yoksa alternatif URL dene
            if not results:
                try:
                    # Alternatif URL formatını dene
                    alt_url = f"https://news.google.com/rss?q={encoded_query}&hl={self.language}&gl={self.region}&ceid={self.region}:{self.language}"
                    alt_response = requests.get(alt_url, headers=headers, timeout=15)
                    
                    if alt_response.status_code == 200:
                        # Aynı ayrıştırma adımlarını alternatif URL için tekrarla
                        items = []
                        content = alt_response.content
                        
                        # 1. XML parser
                        try:
                            soup = BeautifulSoup(content, 'lxml-xml')
                            items = soup.find_all('item')
                        except Exception:
                            pass
                        
                        # 2. HTML parser
                        if not items:
                            try:
                                soup = BeautifulSoup(content, 'lxml')
                                items = soup.find_all('item')
                            except Exception:
                                pass
                        
                        # Alternatif URL'den gelen sonuçları da aynı şekilde işle
                        for item in items[:max_results]:
                            try:
                                # Yukarıdaki işlemleri tekrarla
                                title = ""
                                site = ""
                                
                                title_elem = item.find('title')
                                if title_elem and title_elem.text:
                                    title_text = title_elem.text.strip()
                                    
                                    if " - " in title_text:
                                        title_parts = title_text.rsplit(" - ", 1)
                                        if len(title_parts) == 2:
                                            title = title_parts[0].strip()
                                            site = title_parts[1].strip()
                                    else:
                                        title = title_text
                                        source_elem = item.find('source')
                                        if source_elem and source_elem.text:
                                            site = source_elem.text.strip()
                                
                                # Bağlantı
                                link = ""
                                link_elem = item.find('link')
                                if link_elem:
                                    if link_elem.text and link_elem.text.strip():
                                        link = link_elem.text.strip()
                                    elif link_elem.get('href'):
                                        link = link_elem.get('href').strip()
                                
                                if not link:
                                    guid_elem = item.find('guid')
                                    if guid_elem and guid_elem.text:
                                        link = guid_elem.text.strip()
                                
                                # URL tekrarını kontrol et
                                if not link or link in processed_urls:
                                    continue
                                processed_urls.add(link)
                                
                                # Açıklama
                                desc = ""
                                desc_elem = item.find('description') or item.find('summary') or item.find('content')
                                if desc_elem and desc_elem.text:
                                    desc = desc_elem.text.strip()
                                    # HTML etiketlerini temizle
                                    desc = re.sub(r'<[^>]*>', '', desc)
                                
                                # Yayın tarihi
                                pub_date_str = ""
                                pub_date_elem = item.find('pubDate') or item.find('published') or item.find('updated')
                                if pub_date_elem and pub_date_elem.text:
                                    pub_date_str = pub_date_elem.text.strip()
                                
                                # Tarih formatı
                                datetime_str = ""
                                if pub_date_str:
                                    try:
                                        pub_date_str = pub_date_str.replace('GMT', '+0000')
                                        dt = None
                                        
                                        date_formats = [
                                            "%a, %d %b %Y %H:%M:%S %z",
                                            "%Y-%m-%dT%H:%M:%S%z",
                                            "%Y-%m-%dT%H:%M:%SZ",
                                            "%Y-%m-%d %H:%M:%S",
                                            "%a, %d %b %Y %H:%M:%S"
                                        ]
                                        
                                        for fmt in date_formats:
                                            try:
                                                dt = datetime.strptime(pub_date_str, fmt)
                                                break
                                            except ValueError:
                                                continue
                                        
                                        if dt:
                                            datetime_str = dt.strftime("%d.%m.%Y %H:%M")
                                        else:
                                            datetime_str = pub_date_str
                                    except Exception:
                                        datetime_str = pub_date_str
                                
                                # Sonuçları ekle
                                if title and link:
                                    results.append({
                                        'title': title,
                                        'link': link,
                                        'desc': desc,
                                        'site': site,
                                        'datetime': datetime_str
                                    })
                                
                            except Exception as e:
                                print(f"Alternatif haber öğesi işlenirken hata: {str(e)}")
                                continue
                except Exception as alt_e:
                    print(f"Alternatif URL ile arama hatası: {str(alt_e)}")
                    pass
            
        except Exception as e:
            print(f"GoogleNews arama hatası: {str(e)}")
            
        return results

# Haber kaynakları (provider'lar)
NEWS_PROVIDERS = {
    'Google News': {
        'name': 'Google News',
        'icon': '📰',
        'enabled': True
    }
}

# Render the news tab
if __name__ == "__main__":
    render_stock_news_tab() 