import streamlit as st
import logging
import traceback
from datetime import datetime, timedelta
import pytz
import random
from data.announcements import get_all_announcements, add_announcement, get_session_announcements, delete_announcement, format_time_ago
from data.db_utils import (
    save_analysis_result as db_save_analysis_result,
    get_analysis_results,
    save_favorite_stock,
    remove_favorite_stock,
    is_favorite_stock,
    get_favorite_stocks,
    save_ml_prediction,
    get_ml_predictions,
    update_ml_prediction_result,
    get_ml_prediction_stats,
    save_announcement,
    get_announcements,
    delete_announcement as db_delete_announcement,
    save_user_note,
    get_user_notes,
    create_database,
    save_ml_model,
    load_ml_model,
    get_model_update_status
)

# Loglama yapılandırması
logger = logging.getLogger(__name__)

# Veritabanını başlangıçta oluştur
try:
    create_database()
except Exception as e:
    logger.error(f"Veritabanı oluşturulurken hata: {str(e)}")

def save_analysis_result(stock_symbol, analysis_data, analysis_type=None, price=None, indicators=None, notes=None):
    """
    Belirli bir hisse senedi için yapılan analiz sonucunu depolar.
    
    Args:
        stock_symbol (str): Hisse senedi sembolü
        analysis_data (dict): Analiz sonuç verisi
        analysis_type (str, optional): Analiz tipi, belirtilmezse veri içinden alınır
        price (float, optional): Hisse fiyatı, belirtilmezse veri içinden alınır
        indicators (dict, optional): Teknik göstergeler
        notes (str, optional): Notlar
    """
    try:
        # Tarihi ekle
        now = datetime.now()
        analysis_data["analysis_time"] = now.strftime("%d.%m.%Y %H:%M")
        
        # Backward compatibility - session state'e de kaydet
        if 'stock_analysis_results' not in st.session_state:
            st.session_state.stock_analysis_results = {}
        
        # Session state'e kaydet
        st.session_state.stock_analysis_results[stock_symbol] = analysis_data
        
        # Analiz geçmişine ekle (eğer yoksa)
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        
        # Zaten varsa önce kaldır (hisse senedi sembolü aynı olduğunda)
        if stock_symbol in st.session_state.analysis_history:
            st.session_state.analysis_history.remove(stock_symbol)
        
        # Listeye en başa ekle
        st.session_state.analysis_history.insert(0, stock_symbol)
        
        # Maksimum 20 kayıt tut
        if len(st.session_state.analysis_history) > 20:
            st.session_state.analysis_history = st.session_state.analysis_history[:20]
        
        # Veritabanına kaydet - parametreleri kontrol et
        if analysis_type is None:
            analysis_type = analysis_data.get("analysis_type", "teknik")
            
        if price is None:
            # Önce last_price'ı kontrol et (ML için)
            price = analysis_data.get("last_price", None)
            
            # Eğer yoksa current_price'ı dene
            if price is None:
                price = analysis_data.get("current_price", 0)
        
        if indicators is None:
            indicators = analysis_data.get("indicators", {})
            
        if notes is None:
            notes = analysis_data.get("notes", "")
        
        # Veritabanına kaydet (kalıcı depolama)
        db_result = db_save_analysis_result(
            stock_symbol, 
            analysis_type, 
            price, 
            analysis_data, 
            indicators, 
            notes
        )
        
        logger.info(f"{stock_symbol} için analiz sonucu başarıyla kaydedildi")
        return db_result
    except Exception as e:
        logger.error(f"Analiz sonucu kaydedilirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def get_analysis_result(stock_symbol):
    """
    Belirli bir hisse senedi için kaydedilmiş analiz sonucunu getirir.
    
    Args:
        stock_symbol (str): Hisse senedi sembolü
        
    Returns:
        dict: Analiz sonuç verisi, yoksa None
    """
    try:
        # Önce veritabanından getir
        results = get_analysis_results(stock_symbol, limit=1)
        if results and len(results) > 0:
            return results[0]
        
        # Eğer veritabanında yoksa session state'ten kontrol et (eski veriler için)
        if 'stock_analysis_results' not in st.session_state:
            return None
        
        return st.session_state.stock_analysis_results.get(stock_symbol)
    except Exception as e:
        logger.error(f"Analiz sonucu alınırken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def load_analysis_results(analysis_type=None):
    """
    Kaydedilmiş tüm analiz sonuçlarını getirir.
    
    Args:
        analysis_type (str, optional): Analiz tipi (örn: "teknik", "ml")
        
    Returns:
        list: Analiz sonuçlarının listesi
    """
    try:
        # Doğrudan veritabanından getir
        return get_analysis_results(analysis_type=analysis_type)
    except Exception as e:
        logger.error(f"Analiz sonuçları yüklenirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def add_to_favorites(stock_symbol):
    """
    Bir hisse senedini favorilere ekler.
    
    Args:
        stock_symbol (str): Eklenecek hisse senedi sembolü
    
    Returns:
        bool: İşlem başarılıysa True, aksi halde False
    """
    try:
        # Sembolü düzenle
        stock_symbol = stock_symbol.upper().strip()
        
        # Backward compatibility - session state'e de kaydet
        if 'favorite_stocks' not in st.session_state:
            st.session_state.favorite_stocks = []
            
        if stock_symbol not in st.session_state.favorite_stocks:
            st.session_state.favorite_stocks.append(stock_symbol)
        
        # Veritabanına kaydet
        result = save_favorite_stock(stock_symbol)
        
        logger.info(f"{stock_symbol} favorilere eklendi")
        return result
    except Exception as e:
        logger.error(f"Favori eklenirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def remove_from_favorites(stock_symbol):
    """
    Bir hisseyi favorilerden çıkarır.
    
    Args:
        stock_symbol (str): Çıkarılacak hisse sembolü
    """
    try:
        # Backward compatibility - session state'ten de çıkar
        if 'favorite_stocks' in st.session_state and stock_symbol in st.session_state.favorite_stocks:
            st.session_state.favorite_stocks.remove(stock_symbol)
        
        # Veritabanından çıkar
        result = remove_favorite_stock(stock_symbol)
        
        logger.info(f"{stock_symbol} favorilerden çıkarıldı")
        return result
    except Exception as e:
        logger.error(f"Favori çıkarılırken hata: {str(e)}")
        return False

def is_favorite(stock_symbol):
    """
    Bir hisse senedinin favorilerde olup olmadığını kontrol eder.
    
    Args:
        stock_symbol (str): Hisse senedi sembolü
        
    Returns:
        bool: Favorilerde ise True, değilse False
    """
    return is_favorite_stock(stock_symbol.upper().strip())

def get_favorites():
    """
    Tüm favori hisse senetlerini getirir.
    
    Returns:
        list: Favori hisse senetlerinin listesi
    """
    try:
        # Önce veritabanından getir
        favorites = get_favorite_stocks()
        
        # Backward compatibility - session state'i de güncelle
        if 'favorite_stocks' not in st.session_state:
            st.session_state.favorite_stocks = []
        
        # Session state'teki favorilerle senkronize et
        for favorite in favorites:
            if favorite not in st.session_state.favorite_stocks:
                st.session_state.favorite_stocks.append(favorite)
        
        return favorites
    except Exception as e:
        logger.error(f"Favoriler getirilirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        # Hata durumunda session state'ten döndür
        return st.session_state.get('favorite_stocks', []) 