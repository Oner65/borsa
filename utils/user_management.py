"""
Kullanıcı hesapları ve ayarları için yönetim sistemi.
Oturum açma, kayıt, kullanıcı tercihleri ve hesap yönetimini içerir.
"""

import streamlit as st
import json
import os
import pandas as pd
import hashlib
import string
import random
import logging
import traceback
from datetime import datetime, timedelta
import time
from pathlib import Path

try:
    import streamlit_authenticator as stauth
    AUTHENTICATOR_AVAILABLE = True
except ImportError:
    AUTHENTICATOR_AVAILABLE = False
    logging.warning("streamlit-authenticator bulunamadı. Otomatik kimlik doğrulama kullanılmayacak.")

try:
    from passlib.hash import pbkdf2_sha256
    PASSLIB_AVAILABLE = True
except ImportError:
    PASSLIB_AVAILABLE = False
    logging.warning("passlib bulunamadı. Gelişmiş şifre hashing kullanılmayacak.")

# Loglama yapılandırması
logger = logging.getLogger(__name__)

# Kullanıcı veritabanı dosya yolu
USER_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'users.json')
USER_PREFS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'user_preferences')

# Kullanıcı veritabanını oluştur (eğer yoksa)
def initialize_user_database():
    """
    Kullanıcı veritabanını oluşturur veya yükler.
    """
    try:
        # Veritabanı klasörü var mı kontrol et
        db_dir = os.path.dirname(USER_DB_PATH)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
        # Kullanıcı tercihleri klasörü var mı kontrol et
        if not os.path.exists(USER_PREFS_PATH):
            os.makedirs(USER_PREFS_PATH)
        
        # Kullanıcı veritabanı dosyası var mı kontrol et
        if not os.path.exists(USER_DB_PATH):
            # Yeni veritabanı oluştur
            default_users = {
                "users": {
                    "admin": {
                        "username": "admin",
                        "email": "admin@example.com",
                        "password": hash_password("admin123"),
                        "role": "admin",
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "last_login": None,
                        "active": True
                    }
                }
            }
            
            # Dosyaya kaydet
            with open(USER_DB_PATH, 'w', encoding='utf-8') as f:
                json.dump(default_users, f, indent=4)
                
            logger.info("Kullanıcı veritabanı başarıyla oluşturuldu.")
            return default_users
        
        # Varolan veritabanını yükle
        with open(USER_DB_PATH, 'r', encoding='utf-8') as f:
            users = json.load(f)
            
        logger.info("Kullanıcı veritabanı başarıyla yüklendi.")
        return users
    
    except Exception as e:
        logger.error(f"Kullanıcı veritabanı oluşturulurken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return {"users": {}}

def hash_password(password):
    """
    Parolayı güvenli bir şekilde hash'ler.
    
    Args:
        password (str): Ham parola
        
    Returns:
        str: Hash'lenmiş parola
    """
    if PASSLIB_AVAILABLE:
        return pbkdf2_sha256.hash(password)
    else:
        # Basit hash (güvenli değil - sadece demo amaçlı)
        return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_password, provided_password):
    """
    Sağlanan parolayı hash'lenmiş parola ile karşılaştırır.
    
    Args:
        stored_password (str): Kaydedilmiş hash
        provided_password (str): Kullanıcının girdiği parola
        
    Returns:
        bool: Parolalar eşleşirse True, aksi halde False
    """
    if PASSLIB_AVAILABLE:
        return pbkdf2_sha256.verify(provided_password, stored_password)
    else:
        # Basit hash doğrulama
        hashed_provided = hashlib.sha256(provided_password.encode()).hexdigest()
        return hashed_provided == stored_password

def register_user(username, email, password, role="user"):
    """
    Yeni bir kullanıcı kaydeder.
    
    Args:
        username (str): Kullanıcı adı
        email (str): E-posta adresi
        password (str): Parola
        role (str): Rol (varsayılan: "user")
        
    Returns:
        tuple: (success, message)
    """
    try:
        # Kullanıcı veritabanını yükle
        users = initialize_user_database()
        
        # Kullanıcı adı veya e-posta zaten kullanılıyor mu kontrol et
        for user in users["users"].values():
            if user["username"].lower() == username.lower():
                return False, "Bu kullanıcı adı zaten kullanılıyor."
            if user["email"].lower() == email.lower():
                return False, "Bu e-posta adresi zaten kullanılıyor."
        
        # Yeni kullanıcı oluştur
        new_user = {
            "username": username,
            "email": email,
            "password": hash_password(password),
            "role": role,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_login": None,
            "active": True
        }
        
        # Kullanıcıyı veritabanına ekle
        users["users"][username] = new_user
        
        # Veritabanını güncelle
        with open(USER_DB_PATH, 'w', encoding='utf-8') as f:
            json.dump(users, f, indent=4)
            
        # Kullanıcı tercihlerini oluştur
        create_user_preferences(username)
        
        logger.info(f"Yeni kullanıcı başarıyla kaydedildi: {username}")
        return True, "Kullanıcı başarıyla kaydedildi."
    
    except Exception as e:
        logger.error(f"Kullanıcı kaydedilirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return False, f"Bir hata oluştu: {str(e)}"

def authenticate_user(username, password):
    """
    Kullanıcıyı doğrular.
    
    Args:
        username (str): Kullanıcı adı
        password (str): Parola
        
    Returns:
        tuple: (success, user_data or error_message)
    """
    try:
        # Kullanıcı veritabanını yükle
        users = initialize_user_database()
        
        # Kullanıcı var mı kontrol et
        if username not in users["users"]:
            return False, "Kullanıcı adı veya parola hatalı."
        
        user = users["users"][username]
        
        # Kullanıcı aktif mi kontrol et
        if not user["active"]:
            return False, "Bu hesap devre dışı bırakılmış."
        
        # Parolayı doğrula
        if not verify_password(user["password"], password):
            return False, "Kullanıcı adı veya parola hatalı."
        
        # Son giriş zamanını güncelle
        user["last_login"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Veritabanını güncelle
        with open(USER_DB_PATH, 'w', encoding='utf-8') as f:
            json.dump(users, f, indent=4)
            
        logger.info(f"Kullanıcı başarıyla giriş yaptı: {username}")
        return True, user
    
    except Exception as e:
        logger.error(f"Kullanıcı doğrulanırken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return False, f"Bir hata oluştu: {str(e)}"

def update_user_profile(username, email=None, password=None, role=None, active=None):
    """
    Kullanıcı profilini günceller.
    
    Args:
        username (str): Kullanıcı adı
        email (str, optional): Yeni e-posta adresi
        password (str, optional): Yeni parola
        role (str, optional): Yeni rol
        active (bool, optional): Hesap durumu
        
    Returns:
        tuple: (success, message)
    """
    try:
        # Kullanıcı veritabanını yükle
        users = initialize_user_database()
        
        # Kullanıcı var mı kontrol et
        if username not in users["users"]:
            return False, "Kullanıcı bulunamadı."
        
        user = users["users"][username]
        
        # Güncellenecek alanları kontrol et
        if email is not None:
            # E-posta adresi başka bir kullanıcı tarafından kullanılıyor mu kontrol et
            for u_name, u_data in users["users"].items():
                if u_name != username and u_data["email"].lower() == email.lower():
                    return False, "Bu e-posta adresi zaten kullanılıyor."
            
            user["email"] = email
        
        if password is not None:
            user["password"] = hash_password(password)
        
        if role is not None:
            user["role"] = role
        
        if active is not None:
            user["active"] = active
        
        # Son güncelleme zamanını ekle
        user["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Veritabanını güncelle
        with open(USER_DB_PATH, 'w', encoding='utf-8') as f:
            json.dump(users, f, indent=4)
            
        logger.info(f"Kullanıcı profili başarıyla güncellendi: {username}")
        return True, "Kullanıcı profili başarıyla güncellendi."
    
    except Exception as e:
        logger.error(f"Kullanıcı profili güncellenirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return False, f"Bir hata oluştu: {str(e)}"

def get_user_profile(username):
    """
    Kullanıcı profilini getirir.
    
    Args:
        username (str): Kullanıcı adı
        
    Returns:
        dict or None: Kullanıcı profili veya None (kullanıcı bulunamazsa)
    """
    try:
        # Kullanıcı veritabanını yükle
        users = initialize_user_database()
        
        # Kullanıcı var mı kontrol et
        if username not in users["users"]:
            return None
        
        # Kullanıcı profilini döndür (parolayı hariç tut)
        user = users["users"][username].copy()
        user.pop("password", None)
        
        return user
    
    except Exception as e:
        logger.error(f"Kullanıcı profili getirilirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def list_users(active_only=False, role=None):
    """
    Kullanıcıları listeler.
    
    Args:
        active_only (bool): Sadece aktif kullanıcıları getir
        role (str): Belirli bir role sahip kullanıcıları getir
        
    Returns:
        list: Kullanıcı listesi
    """
    try:
        # Kullanıcı veritabanını yükle
        users = initialize_user_database()
        
        # Kullanıcıları filtrele
        filtered_users = []
        
        for username, user_data in users["users"].items():
            # Aktiflik kontrolü
            if active_only and not user_data["active"]:
                continue
            
            # Rol kontrolü
            if role is not None and user_data["role"] != role:
                continue
            
            # Parolayı hariç tut
            user_copy = user_data.copy()
            user_copy.pop("password", None)
            
            filtered_users.append(user_copy)
        
        return filtered_users
    
    except Exception as e:
        logger.error(f"Kullanıcılar listelenirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def create_user_preferences(username):
    """
    Kullanıcı tercihlerini oluşturur.
    
    Args:
        username (str): Kullanıcı adı
        
    Returns:
        bool: Başarılı ise True, değilse False
    """
    try:
        # Kullanıcı tercihleri dosya yolu
        user_prefs_file = os.path.join(USER_PREFS_PATH, f"{username}.json")
        
        # Dosya zaten var mı kontrol et
        if os.path.exists(user_prefs_file):
            logger.info(f"Kullanıcı tercihleri zaten mevcut: {username}")
            return True
        
        # Varsayılan tercihler
        default_preferences = {
            "theme": "light",
            "favorite_stocks": [],
            "chart_settings": {
                "default_period": "1mo",
                "default_interval": "1d",
                "preferred_indicators": ["RSI", "MACD", "Bollinger"]
            },
            "notifications": {
                "enable_email": False,
                "enable_browser": True,
                "alert_thresholds": []
            },
            "display_settings": {
                "show_market_summary": True,
                "show_portfolio_summary": True,
                "show_news": True
            },
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Dosyaya kaydet
        with open(user_prefs_file, 'w', encoding='utf-8') as f:
            json.dump(default_preferences, f, indent=4)
            
        logger.info(f"Kullanıcı tercihleri başarıyla oluşturuldu: {username}")
        return True
    
    except Exception as e:
        logger.error(f"Kullanıcı tercihleri oluşturulurken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def get_user_preferences(username):
    """
    Kullanıcı tercihlerini getirir.
    
    Args:
        username (str): Kullanıcı adı
        
    Returns:
        dict: Kullanıcı tercihleri
    """
    try:
        # Kullanıcı tercihleri dosya yolu
        user_prefs_file = os.path.join(USER_PREFS_PATH, f"{username}.json")
        
        # Dosya var mı kontrol et
        if not os.path.exists(user_prefs_file):
            # Tercihleri oluştur
            create_user_preferences(username)
        
        # Tercihleri oku
        with open(user_prefs_file, 'r', encoding='utf-8') as f:
            preferences = json.load(f)
            
        return preferences
    
    except Exception as e:
        logger.error(f"Kullanıcı tercihleri getirilirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Hata durumunda varsayılan değerleri döndür
        return {
            "theme": "light",
            "favorite_stocks": [],
            "chart_settings": {
                "default_period": "1mo",
                "default_interval": "1d"
            },
            "error": "Tercihler yüklenemedi"
        }

def update_user_preferences(username, preferences):
    """
    Kullanıcı tercihlerini günceller.
    
    Args:
        username (str): Kullanıcı adı
        preferences (dict): Yeni tercihler
        
    Returns:
        bool: Başarılı ise True, değilse False
    """
    try:
        # Mevcut tercihleri al
        current_prefs = get_user_preferences(username)
        
        # Gelen tercihleri mevcut tercihlerle birleştir
        for key, value in preferences.items():
            if isinstance(value, dict) and key in current_prefs and isinstance(current_prefs[key], dict):
                # Varolan alt sözlüğü güncelle
                current_prefs[key].update(value)
            else:
                # Tüm diğer değerleri güncelle
                current_prefs[key] = value
        
        # Son güncelleme zamanını ekle
        current_prefs["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Kullanıcı tercihleri dosya yolu
        user_prefs_file = os.path.join(USER_PREFS_PATH, f"{username}.json")
        
        # Dosyaya kaydet
        with open(user_prefs_file, 'w', encoding='utf-8') as f:
            json.dump(current_prefs, f, indent=4)
            
        logger.info(f"Kullanıcı tercihleri başarıyla güncellendi: {username}")
        return True
    
    except Exception as e:
        logger.error(f"Kullanıcı tercihleri güncellenirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def setup_auth_ui():
    """
    Streamlit kimlik doğrulama arayüzünü kurar.
    
    Returns:
        tuple: (authenticator, authentication_status, username)
    """
    if not AUTHENTICATOR_AVAILABLE:
        st.error("Otomatik kimlik doğrulama sistemi kullanılamıyor. Gerekli kütüphaneyi yükleyin: streamlit-authenticator")
        return None, False, None
    
    try:
        # Kullanıcıları yükle
        users = initialize_user_database()
        
        # Kimlik doğrulayıcıyı ayarla
        credentials = {
            "usernames": {}
        }
        
        # Kullanıcı bilgilerini ekle
        for username, user_data in users["users"].items():
            if user_data["active"]:
                credentials["usernames"][username] = {
                    "name": username,
                    "password": user_data["password"],
                    "email": user_data["email"]
                }
        
        # Authenticator'ı oluştur (kullanıcı şifreleri zaten hash'lenmiş durumda)
        authenticator = stauth.Authenticate(
            credentials,
            cookie_name="borsa_uygulama_auth",
            key="borsa_auth",
            cookie_expiry_days=30,
            preauthorized=None
        )
        
        # Kimlik doğrulama arayüzünü göster
        name, authentication_status, username = authenticator.login("Giriş", "main")
        
        # Giriş başarısız olduysa
        if authentication_status == False:
            st.error("Kullanıcı adı veya parola hatalı")
            
        # Giriş bilgileri girilmediyse
        elif authentication_status == None:
            st.warning("Lütfen kullanıcı adı ve parolanızı girin")
            
        # Giriş başarılı olduysa
        elif authentication_status:
            st.success(f"Hoş geldiniz, {name}")
            
            # Giriş zamanını güncelle
            users["users"][username]["last_login"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(USER_DB_PATH, 'w', encoding='utf-8') as f:
                json.dump(users, f, indent=4)
        
        return authenticator, authentication_status, username
    
    except Exception as e:
        logger.error(f"Kimlik doğrulama arayüzü kurulurken hata: {str(e)}")
        logger.error(traceback.format_exc())
        st.error("Kimlik doğrulama sistemi başlatılırken bir hata oluştu")
        return None, False, None 