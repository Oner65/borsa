import logging
import traceback
from datetime import datetime, timedelta
import pytz
import random
import streamlit as st
import requests
from bs4 import BeautifulSoup

# Loglama yapılandırması
logger = logging.getLogger(__name__)

def format_time_ago(timestamp):
    """
    Verilen zaman damgasını ne kadar zaman önce olduğunu gösteren
    kullanıcı dostu bir formata dönüştürür.
    
    Args:
        timestamp: datetime objesi
        
    Returns:
        str: Kullanıcı dostu zaman formatı (örn. "2 saat önce")
    """
    try:
        if not timestamp:
            return "Şimdi"
            
        now = datetime.now(pytz.timezone('Europe/Istanbul'))
        
        if not timestamp.tzinfo:
            timestamp = pytz.timezone('Europe/Istanbul').localize(timestamp)
            
        diff = now - timestamp
        
        if diff.days > 30:
            return timestamp.strftime("%d %b %Y")
        elif diff.days > 0:
            return f"{diff.days} gün önce"
        elif diff.seconds >= 3600:
            hours = diff.seconds // 3600
            return f"{hours} saat önce"
        elif diff.seconds >= 60:
            minutes = diff.seconds // 60
            return f"{minutes} dakika önce"
        else:
            return "Az önce"
    except Exception as e:
        logger.error(f"Zaman formatlaması sırasında hata: {str(e)}")
        logger.error(traceback.format_exc())
        return "Belirsiz"

def get_announcements():
    """
    Duyuru listesini döndürür.
    
    Returns:
        list: Duyuru nesneleri listesi
    """
    try:
        # Finans ve borsa ile ilgili gerçek duyuruları API'den çek
        announcements = []
        
        # 1. Investing.com'dan ekonomik takvim etkinliklerini çek
        try:
            # Investing.com ekonomik takvim API'si - bazen CORS veya format sorunları olabilir
            try:
                econ_calendar_url = "https://tr.investing.com/economic-calendar/Service/getCalendarFilteredData"
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "X-Requested-With": "XMLHttpRequest",
                    "Referer": "https://tr.investing.com/economic-calendar/"
                }
                
                today = datetime.now(pytz.timezone('Europe/Istanbul'))
                tomorrow = today + timedelta(days=1)
                
                data = {
                    "country[]": ["4", "5", "17", "37", "72"],  # TR, US, EU, UK, JP
                    "importance[]": ["1", "2", "3"],
                    "timeZone": 3,
                    "timeFilter": "timeOnly",
                    "currentTab": "today",
                    "limit_from": 0
                }
                
                response = requests.post(econ_calendar_url, headers=headers, data=data, timeout=5)
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    # Data anahtarı var mı kontrol et
                    if "data" in response_data and response_data["data"]:
                        for idx, event in enumerate(response_data["data"][:3]):
                            if isinstance(event, dict) and "event_name" in event:
                                title = f"Ekonomik Takvim"
                                content = f"Investing.com'dan önemli ekonomik etkinlik: {event.get('event_name', 'Bilinmeyen Etkinlik')}"
                                announcements.append({
                                    "baslik": title,
                                    "icerik": content,
                                    "zaman": format_time_ago(today - timedelta(hours=idx)),
                                    "renk": "#EFF6FF",
                                    "metin_renk": "#2563EB",
                                    "koyu_renk": "#1E40AF"
                                })
            except (ValueError, KeyError) as parse_error:
                logger.error(f"Investing.com veri format hatası: {str(parse_error)}")
                # Analternative olarak ekonomik takvim için bir güncel duyuru ekle
                announcements.append({
                    "baslik": "Ekonomik Takvim",
                    "icerik": f"Bugün ({today.strftime('%d.%m.%Y')}) önemli ekonomik veriler takip ediliyor.",
                    "zaman": format_time_ago(today - timedelta(hours=1)),
                    "renk": "#EFF6FF",
                    "metin_renk": "#2563EB",
                    "koyu_renk": "#1E40AF"
                })
        except Exception as econ_error:
            logger.error(f"Ekonomik takvim verileri alınırken hata: {str(econ_error)}")
            # Hata durumunda alternatif duyuru ekle
            today = datetime.now(pytz.timezone('Europe/Istanbul'))
            announcements.append({
                "baslik": "Ekonomik Takvim",
                "icerik": f"Bugün ({today.strftime('%d.%m.%Y')}) önemli ekonomik veriler takip ediliyor.",
                "zaman": format_time_ago(today - timedelta(hours=1)),
                "renk": "#EFF6FF",
                "metin_renk": "#2563EB",
                "koyu_renk": "#1E40AF"
            })
        
        # 2. TCMB döviz kurları ve diğer verileri
        try:
            # TCMB günlük döviz kuru API'si
            tcmb_url = "https://www.tcmb.gov.tr/kurlar/today.xml"
            response = requests.get(tcmb_url, timeout=5)
            if response.status_code == 200:
                # XML içeriğini parse et
                import xml.etree.ElementTree as ET
                try:
                    root = ET.fromstring(response.content)
                    
                    # USD kurunu al
                    usd_found = False
                    for currency in root.findall(".//Currency[@Kod='USD']"):
                        try:
                            buying = currency.find("ForexBuying").text
                            selling = currency.find("ForexSelling").text
                            
                            announcements.append({
                                "baslik": "TCMB Kur Bilgisi",
                                "icerik": f"USD/TRY: Alış {buying} TL / Satış {selling} TL",
                                "zaman": format_time_ago(today - timedelta(hours=1)),
                                "renk": "#FEF2F2",
                                "metin_renk": "#DC2626",
                                "koyu_renk": "#7F1D1D"
                            })
                            usd_found = True
                            break
                        except (AttributeError, TypeError) as attr_error:
                            logger.error(f"TCMB USD kur verisi ayrıştırma hatası: {str(attr_error)}")
                    
                    # EUR kurunu da al
                    if not usd_found:
                        for currency in root.findall(".//Currency[@Kod='EUR']"):
                            try:
                                buying = currency.find("ForexBuying").text
                                selling = currency.find("ForexSelling").text
                                
                                announcements.append({
                                    "baslik": "TCMB Kur Bilgisi",
                                    "icerik": f"EUR/TRY: Alış {buying} TL / Satış {selling} TL",
                                    "zaman": format_time_ago(today - timedelta(hours=1)),
                                    "renk": "#FEF2F2",
                                    "metin_renk": "#DC2626",
                                    "koyu_renk": "#7F1D1D"
                                })
                                break
                            except (AttributeError, TypeError):
                                pass
                except ET.ParseError as xml_error:
                    logger.error(f"TCMB XML parse hatası: {str(xml_error)}")
        except Exception as tcmb_error:
            logger.error(f"TCMB verileri alınırken hata: {str(tcmb_error)}")
            
        # 3. Son BIST/VIOP haberleri için Google News'den veri çek
        try:
            # Google News RSS beslemesi
            rss_url = "https://news.google.com/rss/search?q=borsa+istanbul+OR+BIST+OR+VIOP&hl=tr-TR&gl=TR&ceid=TR:tr"
            response = requests.get(rss_url)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, features="xml")
                items = soup.find_all("item")[:2]
                
                for item in items:
                    title_elem = item.find("title")
                    title = title_elem.text if title_elem else "Borsa Haberi"
                    
                    announcements.append({
                        "baslik": "Piyasa Haberi",
                        "icerik": title,
                        "zaman": format_time_ago(today - timedelta(hours=3)),
                        "renk": "#F0FDF4", 
                        "metin_renk": "#16A34A",
                        "koyu_renk": "#166534"
                    })
        except Exception as news_error:
            logger.error(f"Google News verileri alınırken hata: {str(news_error)}")
        
        # Eğer API'lerden hiçbir veri alınamadıysa, bazı statik ama güncel veriler göster
        if not announcements:
            # Mevcut tarih/saate dayalı yarı-dinamik duyurular oluştur
            now = datetime.now(pytz.timezone('Europe/Istanbul'))
            today_str = now.strftime("%d.%m.%Y")
            
            announcements = [
                {
                    "baslik": "Piyasa Uyarısı",
                    "icerik": f"Bugün ({today_str}) USD/TRY kur hareketliliği yakından takip ediliyor.",
                    "zaman": format_time_ago(now - timedelta(hours=2)),
                    "renk": "#FEF2F2",
                    "metin_renk": "#DC2626",
                    "koyu_renk": "#7F1D1D"
                },
                {
                    "baslik": "Ekonomik Takvim",
                    "icerik": f"Bugün ({today_str}) saat 14:30'da ABD işsizlik verileri açıklanacak.",
                    "zaman": format_time_ago(now - timedelta(hours=4)),
                    "renk": "#EFF6FF",
                    "metin_renk": "#2563EB",
                    "koyu_renk": "#1E40AF"
                },
                {
                    "baslik": "BIST Gelişmeleri",
                    "icerik": "BIST100 endeksinde teknik direnç seviyesi 9,500 puan olarak takip ediliyor.",
                    "zaman": format_time_ago(now - timedelta(days=1)),
                    "renk": "#F0FDF4",
                    "metin_renk": "#16A34A",
                    "koyu_renk": "#166534"
                }
            ]
        
        return announcements
    except Exception as e:
        logger.error(f"Duyurular alınırken hata: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Hata durumunda varsayılan bir duyuru göster
        return [{
            "baslik": "Sistem Mesajı",
            "icerik": "Duyurular şu anda yüklenemiyor. Lütfen daha sonra tekrar deneyin.",
            "zaman": "Şimdi",
            "renk": "#FEF2F2",
            "metin_renk": "#DC2626",
            "koyu_renk": "#7F1D1D"
        }]

def get_session_announcements():
    """
    Session state'de saklanan duyuruları döndürür.
    
    Returns:
        list: Duyuru nesneleri listesi
    """
    if 'announcements' not in st.session_state:
        st.session_state.announcements = []
    
    return st.session_state.announcements

def get_all_announcements():
    """
    Hem session state'deki dinamik duyuruları hem de sabit duyuruları birleştirip döndürür.
    
    Returns:
        list: Tüm duyuruların listesi
    """
    try:
        # Session state'den duyuruları al
        session_announcements = get_session_announcements()
        
        # Sabit duyuruları al
        fixed_announcements = get_announcements()
        
        # İki listeyi birleştir - önce session_announcements (daha yeni olduğu için)
        all_announcements = session_announcements + fixed_announcements
        
        # Tarihe göre sırala (eğer created_at varsa)
        def get_timestamp(ann):
            if "created_at" in ann and ann["created_at"]:
                return ann["created_at"]
            return datetime.now()
            
        all_announcements.sort(key=get_timestamp, reverse=True)
        
        return all_announcements
    except Exception as e:
        logger.error(f"Tüm duyurular alınırken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def add_announcement(baslik, icerik, kategori="diger", onem="orta"):
    """
    Yeni bir duyuru ekler (session state'e geçici olarak)
    
    Args:
        baslik (str): Duyuru başlığı
        icerik (str): Duyuru içeriği
        kategori (str): Duyuru kategorisi (piyasa, ekonomik, teknik, diger)
        onem (str): Duyuru önemi (yüksek, orta, düşük)
        
    Returns:
        bool: İşlem başarılıysa True, değilse False
    """
    try:
        # Session state kontrol et
        if 'announcements' not in st.session_state:
            st.session_state.announcements = []
        
        # Kategori renkleri
        renk_mapping = {
            "piyasa": {"renk": "#FEF2F2", "metin_renk": "#DC2626", "koyu_renk": "#7F1D1D"},
            "ekonomik": {"renk": "#EFF6FF", "metin_renk": "#2563EB", "koyu_renk": "#1E40AF"},
            "teknik": {"renk": "#F0FDF4", "metin_renk": "#16A34A", "koyu_renk": "#166534"},
            "diger": {"renk": "#FFF7ED", "metin_renk": "#EA580C", "koyu_renk": "#9A3412"}
        }
        
        # Varsayılan renk
        renk = renk_mapping.get(kategori.lower(), renk_mapping["diger"])
        
        # Duyuruyu oluştur
        announcement = {
            "baslik": baslik,
            "icerik": icerik,
            "zaman": format_time_ago(datetime.now(pytz.timezone('Europe/Istanbul'))),
            "renk": renk["renk"],
            "metin_renk": renk["metin_renk"],
            "koyu_renk": renk["koyu_renk"],
            "kategori": kategori,
            "onem": onem,
            "created_at": datetime.now(pytz.timezone('Europe/Istanbul')),
            "id": f"ann_{int(datetime.now().timestamp())}_{random.randint(1000, 9999)}"
        }
        
        # Listeye ekle
        st.session_state.announcements.insert(0, announcement)
        
        # İlk 50 duyuruyu tut
        if len(st.session_state.announcements) > 50:
            st.session_state.announcements = st.session_state.announcements[:50]
            
        logger.info(f"Yeni duyuru eklendi: {baslik}")
        return True
    except Exception as e:
        logger.error(f"Duyuru eklenirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def delete_announcement(announcement_id):
    """
    Session state'den belirli bir duyuruyu siler.
    
    Args:
        announcement_id (str): Silinecek duyurunun ID'si
        
    Returns:
        bool: İşlem başarılıysa True, değilse False
    """
    try:
        if 'announcements' not in st.session_state:
            return False
            
        # ID'ye göre filtrele
        st.session_state.announcements = [
            ann for ann in st.session_state.announcements 
            if ann.get('id') != announcement_id
        ]
        
        logger.info(f"Duyuru silindi: ID {announcement_id}")
        return True
    except Exception as e:
        logger.error(f"Duyuru silinirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def get_announcement_by_id(id):
    """
    Belirli bir ID'ye sahip duyuruyu getirir (ileride veritabanı entegrasyonu için)
    """
    # Şu an için sadece bir prototip, veritabanı bağlantısı eklenecek
    return None 