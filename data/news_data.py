import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta
import pandas as pd
import traceback
import sys
import logging
import os
import importlib.util
from bs4 import BeautifulSoup
import random
import time
import concurrent.futures
import re
import json
from dateutil import parser
import ssl
from gnews import GNews

# Google arama için alternatif çözüm
def google_search(query, num_results=10, lang='tr'):
    """
    Google araması yapmak için basit bir fonksiyon.
    Eğer google-search-python çalışmazsa, bu basit aramaları kullanacağız.
    """
    try:
        # Basit query formatı
        formatted_query = query.replace(' ', '+')
        search_results = []
        
        # Sadece log olarak bildir
        logging.info(f"Arama yapılıyor: {query}")
        
        # Burada gerçek bir arama yapmıyoruz, ancak yapabileceğimiz alternatif kaynakları gösteriyoruz:
        # 1. Doğrudan requests ile Google'a istek yapabilirdik (ancak bu IP engellemesine yol açabilir)
        # 2. Daha güvenli bir API hizmeti kullanabiliriz (SerpAPI, Google Custom Search API vb.)
        # 3. Aşağıdaki gibi dummy sonuçlar döndürebiliriz:
        
        # Örnek sonuçlar - gerçek sonuçlar değil
        dummy_results = [
            f"https://tr.investing.com/news/economy/{formatted_query}-ekonomi-haberleri-1",
            f"https://www.bloomberght.com/haberler/{formatted_query}-2",
            f"https://www.paraanaliz.com/piyasa/{formatted_query}-3",
            f"https://bigpara.hurriyet.com.tr/borsa/{formatted_query}-4",
            f"https://www.sabah.com.tr/ekonomi/{formatted_query}-5"
        ]
        
        return dummy_results[:num_results]
    except Exception as e:
        logging.error(f"Google arama hatası: {str(e)}")
        return []

# YENİ: Haber içeriğini çekme fonksiyonu (ui/improved_news_tab.py modülünü çağırır)
def fetch_news_content(url, log_container=None):
    """
    Haber içeriğini çeken fonksiyon. Bu fonksiyon ui/improved_news_tab.py içerisindeki 
    aynı isimli fonksiyonu çağırır ve farklı modüllerden erişim sağlar.
    
    Args:
        url (str): Haber URL'si
        log_container: Loglama için container (opsiyonel)
    
    Returns:
        dict: İçerik bilgilerini içeren sözlük ya da None
    """
    try:
        # ui/improved_news_tab.py içindeki fetch_news_content fonksiyonunu dinamik olarak import et
        import os
        import sys
        import importlib.util
        
        # Kök dizini belirle
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        # Modül yolu
        module_path = os.path.join(parent_dir, 'ui', 'improved_news_tab.py')
        
        # Log mesajı göster
        logger.info(f"improved_news_tab.py modülü import ediliyor: {module_path}")
        
        # Modülü dinamik olarak import et
        if not os.path.exists(module_path):
            logger.error(f"Modül dosyası bulunamadı: {module_path}")
            return None
            
        # Modülü spec ile import et
        spec = importlib.util.spec_from_file_location("improved_news_tab", module_path)
        news_tab = importlib.util.module_from_spec(spec)
        
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
            
        spec.loader.exec_module(news_tab)
        
        # Fonksiyonu çağır
        if hasattr(news_tab, 'fetch_news_content'):
            logger.info(f"fetch_news_content fonksiyonu başarıyla import edildi ve çağrılıyor: {url}")
            return news_tab.fetch_news_content(url, log_container)
        else:
            logger.error("fetch_news_content fonksiyonu improved_news_tab modülünde bulunamadı!")
            return None
            
    except Exception as e:
        logger.error(f"fetch_news_content fonksiyonu çağrılırken hata: {e}")
        logger.error(traceback.format_exc())
        return None

# timezone işlemleri için pytz ekleyelim
import pytz

# newspaper3k importunu try-except içine al
try:
    from newspaper import Article, Config, ArticleException
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    ArticleException = Exception # Hata yakalama için tanımla
    logger = logging.getLogger(__name__)
    logger.warning("newspaper3k kütüphanesi bulunamadı. Haber içeriği çekme özelliği kısıtlı olacak.")

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# utils klasörünü sys.path'e ekle (borsa.py ile aynı seviyede olduğunu varsayıyoruz)
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir) # /data'nın üstü, yani proje kök dizini
    utils_dir = os.path.join(parent_dir, 'utils')
    if parent_dir not in sys.path:
        sys.path.append(parent_dir) # Kök dizini ekleyelim ki utils'i bulabilsin
    logger.info(f"Proje kök dizini sys.path'e eklendi: {parent_dir}")
except Exception as path_e:
    logger.error(f"sys.path ayarlanırken hata: {path_e}")
    st.warning(f"sys.path ayarlanırken hata: {path_e}. Modül importları başarısız olabilir.")

# İlerleme ve durum bildirimleri için yardımcı fonksiyonlar
def log_info(message, visible=False):
    """Bilgi mesajı göster, visible=False durumunda sadece debug modunda göster"""
    if visible:
        st.info(message)
    else:
        st.write(f"<div style='display:none'>{message}</div>", unsafe_allow_html=True)
    logger.info(message)

def log_success(message, visible=False):
    """Başarı mesajı göster"""
    if visible:
        st.success(message)
    logger.info(f"BAŞARILI: {message}")

def log_warning(message, visible=True):
    """Uyarı mesajı göster"""
    if visible:
        st.warning(message)
    logger.warning(message)

def log_error(message, exception=None, visible=False):
    """Hata mesajı göster"""
    if visible:
        st.error(message)
    logger.error(message)
    if exception:
        logger.error(f"HATA DETAY: {str(exception)}")
        logger.error(traceback.format_exc())

def log_progress(message, is_warning=False, is_error=False, icon=None, visible=False, progress_container=None):
    """Logging fonksiyonu - container'a veya doğrudan UI'ye gönderilebilir"""
    logger.info(message)  # Her durumda log dosyasına yaz
    
    if not visible:
        return None
        
    if progress_container is not None:
        # Log mesajını container içinde göster
        if is_error:
            progress_container.error(message, icon=icon)
        elif is_warning:
            progress_container.warning(message, icon=icon)
        else:
            progress_container.info(message)
    else:
        # Log mesajını doğrudan UI'de göster
        progress_placeholder = st.empty()
        if is_error:
            progress_placeholder.error(message, icon=icon)
        elif is_warning:
            progress_placeholder.warning(message, icon=icon)
        else:
            progress_placeholder.info(message)
        return progress_placeholder

# Kütüphane kontrol fonksiyonu
def check_required_libraries():
    """Gerekli kütüphanelerin yüklü olup olmadığını kontrol eder"""
    required_packages = {
        'newspaper3k': 'newspaper',
        'lxml_html_clean': 'lxml_html_clean',
        'bs4': 'bs4',
        'requests': 'requests'
    }
    
    missing_packages = []
    
    for package, module_name in required_packages.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

# NewsAPI entegrasyonu için yardımcı fonksiyon
def get_news_from_newsapi(search_term, max_results=10, api_key="ac20645b55e94f07b8c69a830d17a473"):
    """NewsAPI kullanarak haber arama"""
    try:
        # NewsAPI kütüphanesini yükleme kontrolü
        try:
            from newsapi import NewsApiClient
            newsapi = NewsApiClient(api_key=api_key)
        except ImportError:
            logger.warning("newsapi-python kütüphanesi yüklü değil, direkt API isteği kullanılacak")
            return fetch_news_direct_api(search_term, max_results, api_key)
        
        # API'ye istek gönder
        articles = newsapi.get_everything(
            q=f"{search_term}",
            sort_by="publishedAt",
            page_size=max_results
        )
        
        if not articles or "articles" not in articles:
            logger.warning(f"NewsAPI'den haber getirilemedi: {articles.get('message', 'Bilinmeyen hata')}")
            return []
        
        # Sonuçları formatlayıp döndür
        news_list = []
        for article in articles["articles"]:
            # Duyarlılık hesaplama
            sentiment_data = {"sentiment": "Nötr", "score": 0.5}
            if article.get("description"):
                sentiment_data = analyze_sentiment(article["description"])
            
            news_list.append({
                "title": article.get("title", "Başlık Yok"),
                "source": article.get("source", {}).get("name", "Bilinmeyen Kaynak"),
                "summary": article.get("description", "Özet alınamadı."),
                "url": article.get("url", "#"),
                "link": article.get("url", "#"),
                "pub_date": article.get("publishedAt", datetime.now().isoformat()),
                "published_date": article.get("publishedAt", datetime.now().isoformat()),
                "image_url": article.get("urlToImage", ""),
                "sentiment": sentiment_data["score"],
                "provider": "NewsAPI"
            })
        
        return news_list
    
    except Exception as e:
        logger.error(f"NewsAPI haberleri alınırken hata: {str(e)}")
        return []

# Doğrudan NewsAPI'ye istek gönderme (kütüphane yoksa)
def fetch_news_direct_api(search_term, max_results=10, api_key="ac20645b55e94f07b8c69a830d17a473"):
    """NewsAPI kütüphanesi olmadan direkt API çağrısı yapar"""
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": search_term,
            "sortBy": "publishedAt",
            "pageSize": max_results,
            "apiKey": api_key
        }
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            logger.warning(f"NewsAPI HTTP hatası: {response.status_code}")
            return []
        
        try:
            data = response.json()
        except ValueError:
            logger.error("NewsAPI'den geçersiz JSON yanıtı alındı")
            return []
            
        if not data or "articles" not in data:
            logger.warning(f"NewsAPI'den geçersiz yanıt: {data.get('message', 'Bilinmeyen hata')}")
            return []
        
        # Sonuçları formatlayıp döndür
        news_list = []
        for article in data["articles"]:
            # Duyarlılık hesaplama
            sentiment_data = {"sentiment": "Nötr", "score": 0.5}
            if article.get("description"):
                sentiment_data = analyze_sentiment(article["description"])
            
            news_list.append({
                "title": article.get("title", "Başlık Yok"),
                "source": article.get("source", {}).get("name", "Bilinmeyen Kaynak"),
                "summary": article.get("description", "Özet alınamadı."),
                "url": article.get("url", "#"),
                "link": article.get("url", "#"),
                "pub_date": article.get("publishedAt", datetime.now().isoformat()),
                "published_date": article.get("publishedAt", datetime.now().isoformat()),
                "image_url": article.get("urlToImage", ""),
                "sentiment": sentiment_data["score"],
                "provider": "NewsAPI"
            })
        
        return news_list
    
    except Exception as e:
        logger.error(f"Doğrudan NewsAPI isteği sırasında hata: {str(e)}")
        return []

# Duyarlılık analizi fonksiyonu
def analyze_sentiment(text):
    """Metin içeriğine göre duyarlılık analizi yapar"""
    if not text or len(text) < 20:
        return {"sentiment": "Nötr", "score": 0.5}
    
    try:
        # Önce belirli negatif anahtar kelimeler için kontrol et
        negative_indicators = [
            "düşüş", "düştü", "kayb", "geriledi", "düşerek", "azald", "eksiye", 
            "eksi", "değer kaybetti", "düşüşle", "kapatmıştı", "gerileme"
        ]
        
        # Belirli pozitif anahtar kelimeler için kontrol et
        positive_indicators = [
            "yükseliş", "yükseldi", "artış", "arttı", "kazanç", "rekor", "başarı", 
            "yükselerek", "değer kazandı", "olumlu"
        ]
        
        text_lower = text.lower()
        
        # Eğer hem negatif hem de pozitif göstergeler varsa, daha detaylı analiz yap
        neg_matches = sum(1 for word in negative_indicators if word in text_lower)
        pos_matches = sum(1 for word in positive_indicators if word in text_lower)
        
        # Borsa ve negatif kelime kombinasyonları için özel kontrol
        borsa_negative_patterns = [
            "borsa düş", "borsa geril", "endeks düş", "endeks geril", 
            "borsa eksi", "endeks eksi", "bist düş", "bist geril"
        ]
        
        for pattern in borsa_negative_patterns:
            if pattern in text_lower:
                # Borsa düşüşü açıkça belirtilmiş, kesinlikle negatif dön
                return {"sentiment": "Olumsuz", "score": 0.2}
        
        # Eğer kesin bir belirteç bulunamazsa, AI modele sor
        from ai.sentiment_analysis import SentimentAnalyzer
        import re
        
        analyzer = SentimentAnalyzer()
        
        if not analyzer.model:
            raise ValueError("Duyarlılık analizi modeli yüklenemedi")
        
        # Metinde kelime sayısını sınırla (çok uzun metinler için)
        max_words = 200
        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[:max_words])
            
        # Metni temizle - sentiment_training.py'daki clean_text ile benzer olmalı
        def clean_text(txt):
            # Küçük harfe çevir
            txt = txt.lower()
            # Özel karakterleri temizle (noktalama işaretleri hariç)
            txt = re.sub(r'[^\w\s.,!?%]', ' ', txt)
            # Fazla boşlukları temizle
            txt = re.sub(r'\s+', ' ', txt).strip()
            return txt
            
        # Temizlenmiş metin
        cleaned_text = clean_text(text)
        
        # Duyarlılık tahminini yap
        prediction = analyzer.predict([cleaned_text])[0]
        score = analyzer.predict_proba([cleaned_text])[0]
        
        # Pozitif/Negatif belirleme
        sentiment = "Olumlu" if prediction == 1 else "Olumsuz"
        
        # Özel durum kontrolü: Borsa düşüşünü içeren metinler için
        if sentiment == "Olumlu" and neg_matches > pos_matches and any(p in text_lower for p in ["borsa", "endeks", "bist"]):
            # Model yanlış tahmin yapmış olabilir, skoru tersine çevir
            sentiment = "Olumsuz"
            score = -abs(score)  # Skoru negatif yap
            
        return {"sentiment": sentiment, "score": score}
            
    except ImportError:
        # AI modeli kullanılamıyorsa basit kelime analizi yap
        pass
    
    # Basit kelime bazlı analiz (ImportError için fallback)
    import re
    
    # Türkçe olumlu ve olumsuz kelimelerin listesi
    positive_words = {
        'artış', 'yükseliş', 'kazanç', 'kâr', 'rekor', 'başarı', 'pozitif', 'olumlu', 'güçlü', 'büyüme', 
        'iyileşme', 'yükseldi', 'arttı', 'çıktı', 'güven', 'istikrar', 'avantaj', 'fırsat', 'yatırım',
        'imzalandı', 'anlaşma', 'destek', 'teşvik', 'ivme', 'fayda', 'artırdı', 'kazandı', 'genişleme',
        'ihracat', 'ciro', 'teşvik', 'ödül', 'toparlanma', 'umut', 'iyi', 'memnuniyet', 'ralli',
        'yüksek', 'çözüm', 'artacak', 'başarılı', 'kazanım', 'gelişme', 'ilerleme', 'potansiyel'
    }
    
    negative_words = {
        'düşüş', 'kayıp', 'zarar', 'risk', 'gerileme', 'olumsuz', 'negatif', 'zayıf', 'belirsizlik', 
        'endişe', 'azaldı', 'düştü', 'kaybetti', 'gecikme', 'borç', 'iflas', 'kriz', 'tehdit', 'sorun',
        'başarısız', 'yaptırım', 'ceza', 'iptal', 'durgunluk', 'darbe', 'kötü', 'daralma', 'kesinti',
        'baskı', 'paniği', 'çöküş', 'alarm', 'tedirgin', 'zor', 'şok', 'dava', 'soruşturma', 'satış',
        'düşük', 'ağır', 'kötüleşme', 'panik', 'küçülme', 'yavaşlama', 'kapatma', 'haciz', 'çöktü'
    }
    
    # Metin içindeki kelimeleri küçük harfe çevir ve temizle
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)  # Sadece kelimeler
    
    # Olumlu ve olumsuz kelime sayısı
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    total_count = positive_count + negative_count
    if total_count == 0:
        # Kelime listesinde hiçbir kelime bulunamadıysa nötr
        return {"sentiment": "Nötr", "score": 0.5}
    
    # Duyarlılık skoru (0 ile 1 arasında) 
    sentiment_score = positive_count / (positive_count + negative_count)
    
    # Duyarlılık sınıflandırması
    if sentiment_score > 0.65:
        return {"sentiment": "Olumlu", "score": sentiment_score}
    elif sentiment_score < 0.35:
        return {"sentiment": "Olumsuz", "score": sentiment_score}
    else:
        return {"sentiment": "Nötr", "score": sentiment_score}

# Haber analiz fonksiyonu
def analyze_news_with_gemini(url, log_container=None):
    """
    Gemini API kullanarak haber URL'sini analiz eder.
    
    Args:
        url (str): Haber URL'si
        log_container: Log mesajlarını göstermek için Streamlit container
        
    Returns:
        dict: Analiz sonuçları
    """
    # Log mesajlarını yönetmek için yardımcı fonksiyon
    def log_info(message):
        if log_container is not None:
            log_container.info(message)
        else:
            logger.info(message)
    
    def log_error(message):
        if log_container is not None:
            log_container.error(message)
        else:
            logger.error(message)
    
    try:
        log_info(f"Haber analizi başlatılıyor: {url}")
        
        # Newspaper3k ile haber içeriğini al
        try:
            import newspaper
            from newspaper import Article
            
            log_info("Haber içeriği alınıyor...")
            article = Article(url)
            article.download()
            article.parse()
            
            # Makaleyi analiz et
            title = article.title
            authors = ", ".join(article.authors) if article.authors else "Belirtilmemiş"
            publish_date = article.publish_date.strftime("%d.%m.%Y") if article.publish_date else "Belirtilmemiş"
            content = article.text
            
            if not content or len(content) < 100:
                log_info("Haberde yeterli içerik bulunamadı...")
                return {
                    "success": False,
                    "error": "Haber içeriği alınamadı veya çok kısa."
                }
            
            log_info(f"Haber başlığı: {title}")
            log_info(f"İçerik uzunluğu: {len(content)} karakter")
            
        except Exception as e:
            log_error(f"Haber içeriği alınırken hata: {str(e)}")
            return {
                "success": False,
                "error": f"Haber içeriği alınamadı: {str(e)}"
            }
        
        # Haber duyarlılık analizi
        log_info("Duyarlılık analizi yapılıyor...")
        sentiment_result = analyze_sentiment(content)
        
        # Gemini API'ye bağlan ve analizi yap
        from ai.api import initialize_gemini_api
        
        gemini_pro = initialize_gemini_api()
        if gemini_pro is None:
            log_error("Gemini API bağlantısı kurulamadı! API anahtarı kontrol edin.")
            return {
                "success": True,
                "title": title,
                "authors": authors,
                "publish_date": publish_date,
                "content": content,
                "sentiment": sentiment_result["sentiment"],
                "sentiment_score": sentiment_result["score"],
                "ai_summary": "Yapay zeka hizmeti şu anda kullanılamıyor.",
                "ai_analysis": {
                    "etki": "nötr",
                    "etki_sebebi": "Yapay zeka analizi yapılamadı. API bağlantısını kontrol edin.",
                    "önemli_noktalar": ["Analiz için API bağlantısı gerekiyor."]
                }
            }
        
        # Haberi özetle ve analiz et
        log_info("Yapay zeka analizi yapılıyor...")
        prompt = f"""
        Aşağıdaki finans/ekonomi haberi metnini dikkatlice analiz et:
        
        BAŞLIK: {title}
        
        İÇERİK:
        {content[:4000]}  # En fazla 4000 karakter kullan (API limiti için)
        
        Bir finans uzmanı olarak, lütfen bu haberi analiz et ve aşağıdaki formatla yanıt ver:
        
        1. ÖZET: Haberin 2-3 cümlelik kısa bir özeti. Finans açısından en önemli bilgileri içer.
        
        2. ANALİZ:
           - etki: "olumlu", "olumsuz" veya "nötr" olarak haberin piyasa etkisi 
           - etki_sebebi: Haberin neden bu etkiye sahip olduğuna dair 1-2 cümlelik açıklama
           - önemli_noktalar: Haberdeki finansal açıdan önemli 2-4 noktayı madde işaretleriyle liste halinde belirt
        
        JSON formatında yanıt ver.
        """
        
        try:
            response = gemini_pro.generate_content(prompt)
            result_text = response.text
            
            # JSON içeriğini çıkar
            import json
            import re
            
            # JSON formatında yanıt alınabilirse doğrudan parse et
            try:
                # Muhtemel JSON bloğunu bul
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', result_text)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = result_text
                
                ai_result = json.loads(json_str)
                
                # Eğer beklenen alanlar yoksa düzenle
                if isinstance(ai_result, dict):
                    ai_summary = ai_result.get("ÖZET", ai_result.get("özet", ""))
                    
                    # ANALİZ bölümü
                    ai_analysis = ai_result.get("ANALİZ", ai_result.get("analiz", {}))
                    if not isinstance(ai_analysis, dict):
                        ai_analysis = {
                            "etki": "nötr",
                            "etki_sebebi": "Analiz yapılamadı.",
                            "önemli_noktalar": ["Yapısal analiz yapılamadı."]
                        }
                else:
                    ai_summary = "Analiz yapılamadı."
                    ai_analysis = {
                        "etki": "nötr",
                        "etki_sebebi": "Analiz yapılamadı.",
                        "önemli_noktalar": ["Yapısal analiz yapılamadı."]
                    }
                
            except json.JSONDecodeError:
                # JSON olarak parse edilemezse manuel olarak analiz et
                log_info("JSON parsing hatası, manuel analiz yapılıyor...")
                
                # Özet kısmını çıkar
                summary_match = re.search(r'(?:ÖZET|özet):\s*(.*?)(?=\n\n|\n\d\.|\Z)', result_text, re.DOTALL)
                ai_summary = summary_match.group(1).strip() if summary_match else "Özet yapılamadı."
                
                # Etki kısmını çıkar
                etki_match = re.search(r'(?:etki|ETKİ):\s*(.*?)(?=\n|\Z)', result_text, re.DOTALL)
                etki = etki_match.group(1).strip().lower() if etki_match else "nötr"
                
                # Etki sebebi kısmını çıkar
                etki_sebebi_match = re.search(r'(?:etki_sebebi|ETKİ SEBEBİ):\s*(.*?)(?=\n|\Z)', result_text, re.DOTALL)
                etki_sebebi = etki_sebebi_match.group(1).strip() if etki_sebebi_match else "Belirtilmemiş"
                
                # Önemli noktaları çıkar
                önemli_noktalar_match = re.search(r'(?:önemli_noktalar|ÖNEMLİ NOKTALAR):\s*(.*?)(?=\n\n|\Z)', result_text, re.DOTALL)
                if önemli_noktalar_match:
                    önemli_noktalar_text = önemli_noktalar_match.group(1)
                    önemli_noktalar = re.findall(r'[-*]\s*(.*?)(?=\n[-*]|\Z)', önemli_noktalar_text, re.DOTALL)
                    önemli_noktalar = [point.strip() for point in önemli_noktalar if point.strip()]
                else:
                    önemli_noktalar = ["Önemli nokta belirtilmemiş."]
                
                ai_analysis = {
                    "etki": etki,
                    "etki_sebebi": etki_sebebi,
                    "önemli_noktalar": önemli_noktalar
                }
                
            log_info("Yapay zeka analizi tamamlandı.")
                
            return {
                "success": True,
                "title": title,
                "authors": authors,
                "publish_date": publish_date,
                "content": content,
                "sentiment": sentiment_result["sentiment"],
                "sentiment_score": sentiment_result["score"],
                "ai_summary": ai_summary,
                "ai_analysis": ai_analysis
            }
            
        except Exception as api_error:
            log_error(f"Gemini API analiz hatası: {str(api_error)}")
            return {
                "success": True,
                "title": title,
                "authors": authors,
                "publish_date": publish_date,
                "content": content,
                "sentiment": sentiment_result["sentiment"],
                "sentiment_score": sentiment_result["score"],
                "ai_summary": "Yapay zeka analizi sırasında hata oluştu.",
                "ai_analysis": {
                    "etki": "nötr",
                    "etki_sebebi": f"Analiz hatası: {str(api_error)}",
                    "önemli_noktalar": ["Analiz tamamlanamadı."]
                }
            }
            
    except Exception as e:
        log_error(f"Haber analizi sırasında beklenmeyen hata: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# Sabitler
DEFAULT_TIMEOUT = 20  # İstekler için varsayılan timeout (saniye)
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'DNT': '1', # Do Not Track
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

# API anahtarları
BING_NEWS_API_KEY = os.environ.get("BING_NEWS_API_KEY", "")
NEWS_API_KEY = "ac20645b55e94f07b8c69a830d17a473"  # News API anahtarı ekledik

# Retry mekanizması ile Session oluşturma
def requests_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504), session=None):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# Haber kaynakları için sınıf tanımları
class NewsSource:
    """Haber kaynağı için temel sınıf"""
    def __init__(self, name):
        self.name = name
        # Her kaynak için kendi session'ını oluştur, retry mekanizması ile
        self.session = requests_retry_session()
        self.session.headers.update(REQUEST_HEADERS)
        # Rastgele bir User-Agent seç
        self.update_random_user_agent()
        
    def update_random_user_agent(self):
        """Her istek öncesi rastgele User-Agent ata"""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:124.0) Gecko/20100101 Firefox/124.0'
        ]
        self.session.headers['User-Agent'] = random.choice(user_agents)

    def is_relevant(self, title, link, stock_code):
        """Haber başlığı ve linkini ilgili hisse senedi kodu ile karşılaştırır"""
        if not title or not stock_code:
            return False
            
        title_upper = title.upper()
        stock_code_upper = stock_code.upper().replace('.IS', '') # .IS uzantısını kaldır
        
        # Doğrudan kod kontrolü (örn: THYAO)
        if stock_code_upper in title_upper:
            return True
        
        # Şirket adları (daha geniş kontrol) - artık lokal bir sözlük kullanacağız
        common_names = {
            'THYAO': ['TÜRK HAVA YOLLARI', 'THY', 'TURKISH AIRLINES'],
            'ASELS': ['ASELSAN', 'ASELSAN A.Ş.'],
            'GARAN': ['GARANTİ', 'GARANTİ BANKASI', 'GARANTI BBVA', 'GARANTI BANKASI'],
            'SASA': ['SASA POLYESTER', 'SASA A.Ş.'],
            'KCHOL': ['KOÇ HOLDİNG', 'KOÇ GRUBU', 'KOC HOLDING', 'KOC GRUP'],
            'AKBNK': ['AKBANK', 'AKBANK T.A.Ş.'],
            'ISCTR': ['İŞ BANKASI', 'IS BANKASI', 'İŞBANK', 'ISBANK'],
            'TCELL': ['TURKCELL', 'TURKCELL İLETİŞİM'],
            'TUPRS': ['TÜPRAŞ', 'TUPRAS', 'TÜRKİYE PETROL RAFİNERİLERİ'],
            'PETKM': ['PETKİM', 'PETKIM PETROKIMYA', 'PETKİM PETROKİMYA'],
            'EREGL': ['EREĞLİ DEMİR', 'EREGLI DEMIR', 'ERDEMİR', 'ERDEMIR', 'EREĞLİ DEMİR ÇELİK'],
            'BIMAS': ['BİM', 'BİM', 'BİM MAĞAZALAR', 'BIM MAGAZALAR', 'BİRLEŞİK MAĞAZALAR'],
            'TOASO': ['TOFAŞ', 'TOFAS', 'TOFAŞ OTOMOBİL', 'TOFAS OTOMOBIL', 'FCA'],
            'FROTO': ['FORD OTOSAN', 'FORD OTOMOTİV', 'FORD OTOMOTIV', 'FORD MOTOR'],
            'HEKTS': ['HEKTAŞ', 'HEKTAS', 'HEKTAŞ TARIM'],
            'VESTL': ['VESTEL', 'VESTEL ELEKTRONİK', 'VESTEL ELEKTRONIK'],
            'PGSUS': ['PEGASUS', 'PEGASUS HAVA YOLLARI', 'PEGASUS AIRLINES'],
            'SAHOL': ['SABANCI', 'SABANCI HOLDİNG', 'SABANCI HOLDING'],
            'AKSEN': ['AKSA ENERJİ', 'AKSA ENERJI', 'AKSA'],
            'KRDMD': ['KARDEMİR', 'KARDEMIR', 'KARDEMİR D', 'KARDEMIR D'],
            'SISE': ['ŞİŞECAM', 'SISECAM', 'ŞİŞE CAM', 'SISE CAM', 'CAM İŞ', 'CAM IS'],
            'ALARK': ['ALARKO HOLDİNG', 'ALARKO HOLDING', 'ALARKO'],
            'ARCLK': ['ARÇELİK', 'ARCELIK', 'ARÇELİK A.Ş.', 'KOÇ ARÇELİK']
        }
        
        # Başlıkta şirket adı geçiyor mu kontrol et
        if stock_code_upper in common_names:
            for name in common_names[stock_code_upper]:
                # Kelime sınırlarını kontrol et () - birebir eşleşme için
                if re.search(r'\b' + re.escape(name) + r'\b', title_upper):
                    return True
        
        # Link içinde hisse kodu geçiyor mu kontrol et (daha az güvenilir)
        # Örnek: /haberler/thyao-icin-yeni-hedef-fiyat
        if f"/{stock_code_upper.lower()}" in link.lower() or f"-{stock_code_upper.lower()}" in link.lower():
            return True
        
        # Eğer hiçbir eşleşme yoksa, ilgisiz kabul et
        return False
    
    def extract_summary(self, article_url):
        """URL'den haber içerik özetini çıkarır ve sentiment analizi yapar"""
        if not NEWSPAPER_AVAILABLE:
            return {"summary": "Özet alınamadı.", "sentiment_score": 0.5}
        
        if not article_url or article_url == "#":
            return {"summary": "Özet alınamadı.", "sentiment_score": 0.5}
        
        try:
            # newspaper yapılandırması
            news_config = Config()
            news_config.request_timeout = 10
            news_config.browser_user_agent = self.session.headers['User-Agent']
            
            article = Article(article_url, config=news_config, language='tr')
            article.download()
            article.parse()
            
            # Görsel URL'sini al
            image_url = article.top_image if hasattr(article, 'top_image') else ""
            
            if not article.text or len(article.text.strip()) < 50:
                if article.meta_description:
                    summary = article.meta_description
                else:
                    return {"summary": "Özet alınamadı.", "sentiment_score": 0.5, "image_url": image_url}
            else:
                # İçeriği özet olarak kullan
                summary = article.text.strip()[:500] + "..." if len(article.text) > 500 else article.text.strip()
            
            # Duyarlılık analizi
            sentiment_result = analyze_sentiment(summary)
            
            return {
                "summary": summary,
                "sentiment_score": sentiment_result.get("score", 0.5),
                "image_url": image_url
            }
            
        except ArticleException as article_ex:
            return {"summary": "Özet alınamadı.", "sentiment_score": 0.5, "image_url": ""}
        except Exception as e:
            return {"summary": "Özet alınamadı.", "sentiment_score": 0.5, "image_url": ""}
    
    def format_date(self, date_obj, now):
        """Tarih nesnesini okunabilir formata çevirir"""
        if not date_obj:
            return "Tarih Yok"
            
        if date_obj.date() == now.date():
            return f"Bugün {date_obj.strftime('%H:%M')}"
        
        # Dün ise
        yesterday = now.date() - timedelta(days=1)
        if date_obj.date() == yesterday:
            return f"Dün {date_obj.strftime('%H:%M')}"
            
        return date_obj.strftime("%d.%m.%Y %H:%M")

class GoogleNewsSource(NewsSource):
    """Google News için haber kaynağı"""
    
    def __init__(self):
        super().__init__("Google News")
        self.update_random_user_agent()
        
    def fetch_news(self, search_term, cutoff_date, common_names_dict=None, max_items=10): 
        """Google News'ten haberleri getirir"""
        
        # Google News'de aranıyor olduğunu logla
        log_info(f"Google News'te aranıyor: {search_term}")
        
        news_items = []
        
        # Bu aşağıdaki anahtar sözlüğü search_term'in alternatif adlarını içerir
        # Örneğin THYAO için {'THYAO': ['Türk Hava Yolları']}
        if common_names_dict is None:
            common_names_dict = {}
            
        # cutoff_date'i timezone bilgisinden arındır, karşılaştırma sorunlarını önle
        if cutoff_date.tzinfo is not None:
            cutoff_date = cutoff_date.replace(tzinfo=None)
            
        # Arama URL'sini oluştur
        encoded_term = search_term.replace(" ", "+")
        
        # İki farklı URL formatını deneyelim
        urls = [
            f"https://news.google.com/search?q={encoded_term}&hl=tr&gl=TR&ceid=TR:tr",
            f"https://news.google.com/rss/search?q={encoded_term}&hl=tr&gl=TR&ceid=TR:tr"
        ]
        
        for url in urls:
            try:
                # URL'nin RSS mi yoksa HTML mi olduğunu kontrol et
                is_rss = "rss" in url
                
                session = requests_retry_session()
                response = session.get(url, headers=self.session.headers, timeout=30)
                
                if response.status_code != 200:
                    log_warning(f"Google News yanıt kodu: {response.status_code}")
                    continue
                            
                # RSS ise XML işle, değilse HTML işle
                if is_rss:
                    import xml.etree.ElementTree as ET
                    
                    # XML içeriğini parse et
                    root = ET.fromstring(response.content)
                    
                    # RSS içinde 'item' öğelerini bul
                    items = root.findall('.//item')
                    
                    for item in items[:max_items]:
                        title_elem = item.find('title')
                        link_elem = item.find('link')
                        
                        if title_elem is None or link_elem is None:
                            continue
                            
                        title = title_elem.text
                        link = link_elem.text
                        
                        # Kaynak bilgisini çıkar - genellikle başlıkta "... - Kaynak Adı" şeklindedir
                        source = "Google News"
                        if " - " in title:
                            title_parts = title.split(" - ")
                            title = " - ".join(title_parts[:-1])
                            source = title_parts[-1]
                        
                        # Tarih bilgisini al
                        pub_date_elem = item.find('pubDate')
                        pub_date = datetime.now()
                        
                        if pub_date_elem is not None and pub_date_elem.text:
                            try:
                                # RSS tarih formatı: Wed, 17 Apr 2025 12:30:45 GMT
                                from email.utils import parsedate_to_datetime
                                pub_date = parsedate_to_datetime(pub_date_elem.text)
                                # timezone bilgisini kaldır, karşılaştırma sorunlarını önle
                                if pub_date.tzinfo is not None:
                                    pub_date = pub_date.replace(tzinfo=None)
                            except Exception as date_err:
                                log_warning(f"RSS tarih çözümleme hatası: {date_err}")
                        
                        # Tarihi kontrol et
                        if pub_date < cutoff_date:
                            continue
                            
                        # İçerik özetini al
                        description_elem = item.find('description')
                        summary = description_elem.text if description_elem is not None else ""
                        
                        if not summary:
                            summary = "Özet bulunamadı."
                            
                        # Özet içeriğini al ve duyarlılık analizi yap
                        content_info = self.extract_summary(link)
                        if content_info.get("summary", "") != "Özet alınamadı." and len(content_info.get("summary", "")) > len(summary):
                            summary = content_info.get("summary", summary)
                        
                        # Haber görseli varsa ekle
                        image_url = content_info.get("image_url", "")
                        
                        # Haberi listeye ekle
                        news_items.append({
                            'title': title,
                            'link': link,
                            'source': source,
                            'summary': summary,
                            'published_datetime': pub_date,
                            'provider': self.name,
                            'sentiment': content_info.get("sentiment_score", 0.5),
                            'image_url': image_url
                        })
                
                else:
                    # HTML içeriğini işle
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Modern Google News formatı
                    articles = soup.select('div[class*="NiLAwe"]')
                    
                    if not articles:
                        # Alternatif seçici deneme
                        articles = soup.select('article, div.xrnccd, main article')
                    
                    # Yine bulamazsak daha genel seçici dene
                    if not articles:
                        articles = soup.select('div[role="article"], div.S8PBwe, main > div > div > div')
                    
                    # Elde edilen sonuç sayısını raporla
                    if articles:
                        log_info(f"Google News'den {len(articles)} sonuç alındı")
                    else:
                        log_warning("Google News'den sonuç alınamadı, HTML yapısı değişmiş olabilir", visible=False)
                        continue
                    
                    for article in articles[:max_items]:
                        # Başlık elementi bul
                        title_elem = article.select_one('h3 a, h4 a, a[aria-label], a.DY5T1d')
                        
                        if not title_elem:
                            # Alternatif seçici dene
                            title_elem = article.select_one('a, span[class*="title"]')
                            
                        if not title_elem:
                            continue
                            
                        # Başlık metni
                        title = title_elem.get_text(strip=True)
                        
                        # Link 
                        link = title_elem.get('href', '#')
                        
                        # Google News link düzeltme
                        if link.startswith('./'):
                            link = "https://news.google.com" + link[1:]
                        elif link.startswith('/'):
                            link = "https://news.google.com" + link
                            
                        # Kaynak bilgisini bul
                        source_elem = article.select_one('a[data-n-tid], span.SVJrMe, div.vr1PYe')
                        source = source_elem.get_text(strip=True) if source_elem else "Google News"
                        
                        # Zaman bilgisini bul
                        time_elem = article.select_one('time, span[class*="time"], div.OSrXXb')
                        time_text = time_elem.get_text(strip=True) if time_elem else ""
                        
                        # Zaman bilgisini işle
                        published_date = datetime.now()
                        
                        if time_text:
                            try:
                                # Google News'in Türkçe zaman formatını analiz et
                                if 'dakika önce' in time_text or 'dk. önce' in time_text:
                                    minutes = int(''.join(filter(str.isdigit, time_text)))
                                    published_date = datetime.now() - timedelta(minutes=minutes)
                                elif 'saat önce' in time_text or 'sa. önce' in time_text:
                                    hours = int(''.join(filter(str.isdigit, time_text)))
                                    published_date = datetime.now() - timedelta(hours=hours)
                                elif 'gün önce' in time_text:
                                    days = int(''.join(filter(str.isdigit, time_text)))
                                    published_date = datetime.now() - timedelta(days=days)
                                elif 'ay önce' in time_text:
                                    months = int(''.join(filter(str.isdigit, time_text)))
                                    published_date = datetime.now() - timedelta(days=months*30)
                                # Belirli tarih formatları
                                else:
                                    try:
                                        # Türkçe tarih formatları
                                        formats = [
                                            '%d %b', # 17 Nis
                                            '%d.%m.%Y', # 17.04.2025
                                            '%d %B %Y', # 17 Nisan 2025
                                            '%d %b %Y' # 17 Nis 2025
                                        ]
                                        
                                        for date_format in formats:
                                            try:
                                                published_date = datetime.strptime(time_text, date_format)
                                                # Yıl bilgisi olmayan formatlarda bugünün yılını ekle
                                                if '%Y' not in date_format:
                                                    published_date = published_date.replace(year=datetime.now().year)
                                                break
                                            except ValueError:
                                                continue
                                    except Exception as e:
                                        log_warning(f"Tarih ayrıştırma hatası: {str(e)}")
                            except Exception as time_err:
                                log_warning(f"Zaman ayrıştırma hatası: {str(time_err)}")
                        
                        # timezone bilgisini kaldır (eğer varsa)
                        if published_date.tzinfo is not None:
                            published_date = published_date.replace(tzinfo=None)
                        
                        # Özet bilgisini bul
                        summary_elem = article.select_one('span[class*="xBbh9"], div.GI74Re')
                        summary = summary_elem.get_text(strip=True) if summary_elem else "Özet bulunamadı."
                        
                        # Tarihi kontrol et
                        if published_date < cutoff_date:
                            continue
                            
                        # Görsel url'sini bul
                        image_url = ""
                        img_elem = article.select_one('img[src*="https"]')
                        if img_elem and img_elem.get('src'):
                            image_url = img_elem.get('src')
                            
                        # Özet içeriğini al ve duyarlılık analizi yap
                        content_info = self.extract_summary(link)
                        if content_info.get("summary", "") != "Özet alınamadı." and len(content_info.get("summary", "")) > len(summary):
                            summary = content_info.get("summary", summary)
                            
                        # Haber görseli alınamadıysa içerikten almayı dene
                        if not image_url and content_info.get("image_url"):
                            image_url = content_info.get("image_url")
                        
                        # Haberi listeye ekle
                        news_items.append({
                            'title': title,
                            'link': link,
                            'source': source,
                            'summary': summary,
                            'published_datetime': published_date,
                            'provider': self.name,
                            'sentiment': content_info.get("sentiment_score", 0.5),
                            'image_url': image_url
                        })
                
                # Yeterli haber bulunduğunda döngüden çık
                if len(news_items) >= max_items:
                    break
                    
            except Exception as e:
                log_warning(f"Google News veri çekme hatası ({url}): {str(e)}")
                continue
                
        return news_items

class YahooNewsSource(NewsSource):
    """Yahoo Finance için haber kaynağı"""
    
    def __init__(self):
        super().__init__("Yahoo Finance")
        self.update_random_user_agent()
        
    def fetch_news(self, search_term, cutoff_date, common_names_dict=None, max_items=10):
        """Yahoo Finance'ten haberleri getirir"""
        
        # Yahoo'da aranıyor olduğunu logla
        log_info(f"Yahoo Finance'te aranıyor: {search_term}")
        
        news_items = []
        
        try:
            # Yahoo Finance arama URL'si
            encoded_term = search_term.replace(" ", "+")
            url = f"https://search.yahoo.com/search?p={encoded_term}+finance+news&fr=finance"
            
            session = requests_retry_session()
            response = session.get(url, headers=self.session.headers, timeout=30)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Arama sonuçlarını bul
                results = soup.select('div.algo')
                
                for result in results[:max_items]:
                    # Başlık ve link
                    title_elem = result.select_one('h3')
                    if not title_elem:
                        continue
                        
                    link_elem = title_elem.select_one('a')
                    if not link_elem:
                        continue
                        
                    title = title_elem.get_text(strip=True)
                    link = link_elem.get('href')
                    
                    # Özet metni bul
                    summary_elem = result.select_one('p, div.compText, span.fz-ms')
                    summary = summary_elem.get_text(strip=True) if summary_elem else "Özet alınamadı."
                    
                    # Kaynak bilgisi
                    source_domain = link.split('//')[-1].split('/')[0]
                    
                    # Alan adından temiz kaynak adı çıkar
                    if 'tr.investing.com' in source_domain:
                        source = 'Investing.com'
                    elif 'finance.yahoo.com' in source_domain or 'tr.finance.yahoo.com' in source_domain:
                        source = 'Yahoo Finance'
                    elif 'bloomberght.com' in source_domain:
                        source = 'BloombergHT'
                    elif 'finans.mynet.com' in source_domain:
                        source = 'Mynet Finans'
                    elif 'bigpara.hurriyet.com.tr' in source_domain:
                        source = 'BigPara'
                    elif 'finansgundem.com' in source_domain:
                        source = 'Finans Gündem'
                    elif 'businessht.com.tr' in source_domain:
                        source = 'Business HT'
                    elif 'ekonomi.haber7.com' in source_domain:
                        source = 'Haber7 Ekonomi'
                    elif 'paraanaliz.com' in source_domain:
                        source = 'Para Analiz'
                    else:
                        # Alan adından basit kaynak adı oluştur
                        parts = source_domain.split('.')
                        if len(parts) >= 2:
                            source = parts[-2].capitalize()
                        else:
                            source = source_domain.capitalize()
                    
                    # Tarih yaklaşık olarak şu andır (Yahoo Search sonuçlarında tarih genellikle yoktur)
                    published_date = datetime.now() - timedelta(days=random.randint(0, 5))
                    
                    # Özet içeriğini al ve duyarlılık analizi yap
                    content_info = self.extract_summary(link)
                    if content_info.get("summary", "") != "Özet alınamadı." and len(content_info.get("summary", "")) > len(summary):
                        summary = content_info.get("summary", summary)
                    
                    # Haberi listeye ekle
                    news_items.append({
                        'title': title,
                        'link': link,
                        'source': source,
                        'summary': summary,
                        'published_datetime': published_date,
                        'provider': self.name,
                        'sentiment': content_info.get("sentiment_score", 0.5)
                    })
                    
                    # Maksimum sonuç sayısına ulaştıysak döngüden çık
                    if len(news_items) >= max_items:
                        break
            
            # Toplam sonuçları raporla
            log_info(f"[{self.name}] {len(news_items)} haber döndürüldü.")
            
        except Exception as e:
            log_warning(f"[{self.name}] Haber alınırken hata: {str(e)}")
            
        return news_items

# News Data fonksiyonlarının sonu

def get_stock_news(stock_symbol, max_results=10, news_period="1m", progress_container=None, providers=None):
    """
    Belirtilen hisse senedi kodu için haberleri çeker.
    
    Parametreler:
    - stock_symbol: Hisse senedi kodu
    - max_results: Maksimum sonuç sayısı
    - news_period: Haber dönemi (1d, 1w, 1m, 3m, 1y)
    - progress_container: İlerleme göstergeleri için container
    - providers: Kullanılacak haber kaynakları listesi
    
    Dönüş:
    - Haber bilgilerini içeren liste
    """
    if not stock_symbol:
        log_error("Hisse senedi kodu belirtilmedi!")
        return None
    
    # Günlük fonksiyonu
    def log_progress(message, is_warning=False, is_error=False, icon=None):
        if progress_container is not None:
            if is_error:
                progress_container.error(message, icon=icon)
            elif is_warning:
                progress_container.warning(message, icon=icon)
            else:
                progress_container.info(message)
        logger.info(message)
    
    try:
        # Kaynak listesini kontrol et
        log_progress(f"{stock_symbol} için haberler aranıyor...")
        
        # Kaynaklar belirtilmemişse varsayılanları kullan
        if not providers:
            providers = ["Google News", "Yahoo Finance"]
        
        # Kesme tarihi hesapla
        now = datetime.now()
        
        if news_period == "1d":
            cutoff_date = now - timedelta(days=1)
        elif news_period == "3d":
            cutoff_date = now - timedelta(days=3)
        elif news_period == "1w":
            cutoff_date = now - timedelta(days=7)
        elif news_period == "1m":
            cutoff_date = now - timedelta(days=30)
        elif news_period == "3m":
            cutoff_date = now - timedelta(days=90)
        elif news_period == "1y": 
            cutoff_date = now - timedelta(days=365)
        else:
            # Varsayılan olarak 1 ay
            cutoff_date = now - timedelta(days=30)
            
        # Arama terimi
        search_term = f"{stock_symbol}"
        
        log_progress(f"Tarih aralığı: {cutoff_date.strftime('%Y-%m-%d')} - {now.strftime('%Y-%m-%d')}")
        
        all_news = []
        
        # NewsAPI kullanarak haber getirme (eğer seçildiyse)
        if "Google News" in providers or "NewsAPI" in providers:
            log_progress("NewsAPI üzerinden haberler getiriliyor...")
            # API anahtarını doğrudan kullan
            api_key = "ac20645b55e94f07b8c69a830d17a473"
            
            # NewsAPI anahtarını kullan
            newsapi_results = get_news_from_newsapi(search_term, max_results, api_key)
            if newsapi_results:
                log_progress(f"NewsAPI'den {len(newsapi_results)} haber bulundu.")
                all_news.extend(newsapi_results)
            else:
                log_progress("NewsAPI'den sonuç alınamadı.", is_warning=True)
        
        # Google News kaynak olarak seçilmişse ve NewsAPI'den yeterli sonuç gelmemişse
        if "Google News" in providers and len(all_news) < max_results:
            log_progress("Google News üzerinden haberler getiriliyor...")
            google_news = GoogleNewsSource()
            try:
                google_results = google_news.fetch_news(search_term, cutoff_date, None, max_items=max_results)
                if google_results:
                    log_progress(f"Google News'ten {len(google_results)} haber bulundu.")
                    all_news.extend(google_results)
                else:
                    log_progress("Google News'ten sonuç alınamadı.", is_warning=True)
            except Exception as gnews_err:
                log_progress(f"Google News hatası: {str(gnews_err)}", is_error=True)
        
        # Yahoo Finance kaynak olarak seçilmişse
        if "Yahoo Finance" in providers and len(all_news) < max_results:
            log_progress("Yahoo Finance üzerinden haberler getiriliyor...")
            yahoo_news = YahooNewsSource()
            try:
                yahoo_results = yahoo_news.fetch_news(search_term, cutoff_date, None, max_items=max_results)
                if yahoo_results:
                    log_progress(f"Yahoo Finance'den {len(yahoo_results)} haber bulundu.")
                    all_news.extend(yahoo_results)
                else:
                    log_progress("Yahoo Finance'den sonuç alınamadı.", is_warning=True)
            except Exception as yahoo_err:
                log_progress(f"Yahoo Finance hatası: {str(yahoo_err)}", is_error=True)
        
        # Haberlerin benzersiz olmasını sağla
        unique_news = []
        titles = set()
        
        for news in all_news:
            title = news.get("title", "")
            if title and title not in titles:
                titles.add(title)
                unique_news.append(news)
        
        log_progress(f"Toplam {len(unique_news)} benzersiz haber bulundu.")
        
        # Maksimum sonuç sayısını kontrol et
        if len(unique_news) > max_results:
            unique_news = unique_news[:max_results]
            log_progress(f"Sonuçlar {max_results} ile sınırlandırıldı.")
        
        return unique_news
        
    except Exception as e:
        log_error(f"Haber arama sırasında hata: {str(e)}", e)
        return None

def get_general_market_news(max_results=5, news_period="1w", progress_container=None):
    """
    Genel piyasa ve ekonomi haberlerini farklı finans kaynaklarından getirir
    
    Args:
        max_results (int): Maksimum sonuç sayısı
        news_period (str): Zaman dilimi ('1d', '1w', '1m', '3m')
        progress_container: Log mesajları için konteyner
        
    Returns:
        list: Haberler listesi
    """
    # Log mesajlarını göstermek için fonksiyon
    def log_progress(message, is_warning=False, is_error=False, icon=None):
        """Logging fonksiyonu - container'a veya doğrudan UI'ye gönderilebilir"""
        if progress_container is not None:
            # Log mesajını container içinde göster
            if is_error:
                progress_container.error(message, icon=icon)
            elif is_warning:
                progress_container.warning(message, icon=icon)
            else:
                progress_container.info(message)
        else:
            # Log mesajını doğrudan UI'de göster
            progress_placeholder = st.empty()
            if is_error:
                progress_placeholder.error(message, icon=icon)
            elif is_warning:
                progress_placeholder.warning(message, icon=icon)
            else:
                progress_placeholder.info(message)
            return progress_placeholder
    
    # İşlem başlangıcı
    progress_placeholder = log_progress("Genel piyasa haberleri alınıyor...")
    
    news_items = []
    processed_links = set()
    
    # Zaman periyodunu datetime'a çevir
    period_map = {
        "1d": timedelta(days=1),
        "3d": timedelta(days=3),
        "1w": timedelta(days=7),
        "2w": timedelta(days=14),
        "1m": timedelta(days=30),
        "3m": timedelta(days=90)
    }
    
    now = datetime.now()
    cutoff_date = now - period_map.get(news_period, timedelta(days=7))
    
    # Arama terimleri
    search_queries = [
        # Genel Terimler
        "Borsa İstanbul",
        "Türkiye Ekonomi",
        "Küresel Ekonomi",
        "Jeopolitik Riskler",
        # Spesifik Terimler
        "Faiz Kararı",
        "Enflasyon Verisi",
        "ABD Enflasyon",
        "Fed Faiz",
        "Avrupa Merkez Bankası",
        "Petrol Fiyatları",
        # Investing.com Hedefli Aramalar
        "hisse senedi haberleri site:tr.investing.com",
        "ekonomi haberleri site:tr.investing.com",
        "piyasa analizi site:tr.investing.com",
        "ABD Çin ticaret site:tr.investing.com",
        # Yahoo Finance Hedefli Aramalar
        "borsa haberleri site:finance.yahoo.com", 
        "Turkey stock market site:finance.yahoo.com",
        "BIST analysis site:finance.yahoo.com",
        "Turkish economy site:finance.yahoo.com", 
        "piyasalar site:yahoo.com"
    ]
    
    # Haber kaynaklarını oluştur
    news_sources = [
        GoogleNewsSource()
    ]
    
    # Her bir kaynak için arama yap
    for i, source in enumerate(news_sources):
        if len(news_items) >= max_results:
            break
            
        # Kaynağa göre maksimum öğe sayısını ayarla
        source_max_items = 5
            
        log_progress(f"Genel piyasa haberleri alınıyor ({i+1}/{len(news_sources)})...")
        
        # Her kaynakta sırayla arama terimlerini dene
        source_queries = search_queries.copy()
        
        for query in source_queries:
            if len(news_items) >= max_results:
                break
                
            try:
                # GoogleNewsSource için common_names_dict gerekli
                if source.name == "Google News":
                    source_news = source.fetch_news(query, cutoff_date, common_names_dict={}, max_items=source_max_items)
                else:
                    source_news = source.fetch_news(query, cutoff_date, max_items=source_max_items)
                
                if source_news:
                    for news in source_news:
                        # Link kontrolü
                        if news['link'] in processed_links:
                            continue
                            
                        if len(news_items) >= max_results:
                            break
                            
                        # Özet çekmeyi dene (newspaper3k varsa)
                        summary = news.get('summary', "Özet alınamadı.")
                        if (summary == "Özet alınamadı." or summary == "Özet alınıyor...") and news.get('link'):
                            try:
                                from newspaper import Article, Config
                                
                                # newspaper yapılandırması
                                news_config = Config()
                                news_config.request_timeout = 10
                                news_config.browser_user_agent = source.headers['User-Agent'] if hasattr(source, 'headers') else None
                                
                                article = Article(news['link'], config=news_config, language='tr')
                                article.download()
                                time.sleep(0.5)  # Rate limiting için bekleme
                                article.parse()
                                
                                if article.text and len(article.text.strip()) > 20:
                                    summary = article.text.strip()[:200] + "..."
                                elif article.meta_description:
                                    summary = article.meta_description.strip()[:200] + "..."
                            except Exception as article_ex:
                                log_progress(f"Haber özeti alınamadı: {article_ex}", is_warning=True, icon="⚠️")
                        
                        # Başlık, kaynak, link ve özeti ekle
                        news_items.append({
                            'title': news.get('title', 'Başlık Yok'),
                            'source': news.get('source', 'Kaynak Yok'),
                            'link': news.get('link'),
                            'summary': summary,
                            'published_datetime': news.get('published_datetime')
                        })
                        processed_links.add(news['link'])
                        
            except Exception as e:
                log_progress(f"{source.name} üzerinde '{query}' araması sırasında hata: {str(e)}", is_warning=True, icon="⚠️")
                continue
    
    # Haberleri tarihe göre sırala (yeniden eskiye)
    try:
        news_items.sort(key=lambda x: x.get('published_datetime') or datetime.min, reverse=True)
    except Exception as sort_e:
        log_progress(f"Haberler sıralanırken hata: {str(sort_e)}", is_warning=True, icon="⚠️")
    
    return news_items[:max_results]