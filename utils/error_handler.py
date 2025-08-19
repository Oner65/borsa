"""
Borsa Uygulaması için Hata Yönetimi Yardımcı Modülü
Bu modül, uygulamada oluşabilecek hataları yönetmek için fonksiyonlar içerir.
"""

import streamlit as st
import logging
from functools import wraps
import traceback
import sys
from config import ERROR_MESSAGES

# Hata ayıklama için logger oluşturalım
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("borsa_app")

def log_exception(e, context=""):
    """Hata logunu kaydeder ve hata stack trace'ini döndürür."""
    error_message = f"{context}: {str(e)}" if context else str(e)
    logger.error(error_message)
    logger.error(traceback.format_exc())
    return traceback.format_exc()

def handle_api_error(func):
    """API çağrıları sırasında oluşabilecek hataları yakalamak için decorator."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = ERROR_MESSAGES.get("api_error", "Veri çekilirken bir hata oluştu.")
            log_exception(e, f"API Error in {func.__name__}")
            st.error(error_message)
            return None
    return wrapper

def handle_analysis_error(func):
    """Analiz sırasında oluşabilecek hataları yakalamak için decorator."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = ERROR_MESSAGES.get("analysis_error", "Analiz sırasında bir hata oluştu.")
            log_exception(e, f"Analysis Error in {func.__name__}")
            st.error(error_message)
            return None
    return wrapper

def show_error_message(error_type, additional_info=""):
    """Yapılandırılmış hata mesajlarını gösterir."""
    message = ERROR_MESSAGES.get(error_type, "Bir hata oluştu.")
    if additional_info:
        message += f" Detay: {additional_info}"
    st.error(message)

def try_except_block(error_type="analysis_error"):
    """İşlev için try-except bloğu oluşturan bir decorator."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                show_error_message(error_type)
                log_exception(e, f"Error in {func.__name__}")
                return None
        return wrapper
    return decorator 