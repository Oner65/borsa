import sqlite3
import os
import logging
import pandas as pd
import json
from datetime import datetime
import traceback

# Loglama yapılandırması
logger = logging.getLogger(__name__)

# Veritabanı dosya yolu
DB_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'stock_analysis.db')

# Sektör Çeviri Sözlüğü (Basit Başlangıç)
SECTOR_TRANSLATIONS = {
    "Financials": "Finans",
    "Industrials": "Sanayi",
    "Technology": "Teknoloji",
    "Consumer Cyclical": "Tüketim - Döngüsel",
    "Consumer Defensive": "Tüketim - Savunma",
    "Energy": "Enerji",
    "Basic Materials": "Temel Malzemeler",
    "Utilities": "Kamu Hizmetleri",
    "Communication Services": "İletişim Hizmetleri",
    "Real Estate": "Gayrimenkul",
    "Healthcare": "Sağlık",
    "Aviation": "Havacılık",
    "Banking": "Bankacılık",
    "Insurance": "Sigortacılık",
    "Chemicals": "Kimya",
    "Holding": "Holding",
    "Conglomerates": "Holding ve Yatırım Şirketleri", # Örnek
    # ... Diğer sektörler eklenebilir
}

def create_database():
    """
    SQLite veritabanı şemasını oluşturur
    
    Returns:
        bool: İşlem başarılıysa True, değilse False
    """
    try:
        logger.info("Veritabanı oluşturuluyor: " + DB_FILE)
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # ML Tahminleri tablosu
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS ml_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            target_date TEXT,
            current_price REAL NOT NULL,
            prediction_percentage REAL NOT NULL,
            confidence_score REAL NOT NULL,
            prediction_result TEXT NOT NULL,
            model_type TEXT NOT NULL,
            features_used TEXT,
            was_correct INTEGER DEFAULT -1,
            actual_result REAL,
            verified_date TEXT
        )
        ''')
        
        # Kullanıcı notları tablosu
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            note_date TEXT NOT NULL,
            note_text TEXT NOT NULL
        )
        ''')
        
        # Analiz sonuçları tablosu
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            analysis_type TEXT NOT NULL,
            analysis_date TEXT NOT NULL,
            price REAL NOT NULL,
            result_data TEXT NOT NULL,
            indicators TEXT,
            notes TEXT
        )
        ''')
        
        # Favoriler tablosu
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS favorite_stocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT UNIQUE NOT NULL,
            added_date TEXT NOT NULL
        )
        ''')
        
        # Duyurular tablosu
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS announcements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            announcement_date TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            is_important INTEGER DEFAULT 0
        )
        ''')
        
        # Portföy tablosu
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            purchase_date TEXT NOT NULL,
            quantity REAL NOT NULL,
            purchase_price REAL NOT NULL,
            notes TEXT,
            sector TEXT,
            is_active INTEGER DEFAULT 1,
            target_price REAL,
            stop_loss REAL
        )
        ''')
        
        # Portföy işlemleri tablosu
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            transaction_date TEXT NOT NULL,
            transaction_type TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            commission REAL DEFAULT 0,
            total_amount REAL NOT NULL,
            notes TEXT
        )
        ''')
        
        # Hisse Senedi Bilgileri Tablosu
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_info (
            symbol TEXT PRIMARY KEY,
            name TEXT,
            sector_en TEXT,
            sector_tr TEXT,
            last_updated TEXT
        )
        ''')
        
        # ML Modelleri Tablosu - eğer yoksa oluştur
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS ml_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            model_type TEXT NOT NULL,
            model_data BLOB NOT NULL,
            features_used TEXT,
            training_date TEXT NOT NULL,
            last_update_date TEXT NOT NULL,
            performance_metrics TEXT,
            is_active INTEGER DEFAULT 1
        )
        ''')
        
        # ml_models tablosunun varlığını kontrol et
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ml_models'")
        if cursor.fetchone() is None:
            # Tablo yoksa tekrar oluşturma denemesi
            logger.warning("ml_models tablosu oluşturulamadı. Tekrar deneniyor...")
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS ml_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                model_type TEXT NOT NULL,
                model_data BLOB NOT NULL,
                features_used TEXT,
                training_date TEXT NOT NULL,
                last_update_date TEXT NOT NULL,
                performance_metrics TEXT,
                is_active INTEGER DEFAULT 1
            )
            ''')
            # Tekrar kontrol et
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ml_models'")
            if cursor.fetchone() is None:
                logger.error("ml_models tablosu ikinci denemede de oluşturulamadı!")
            else:
                logger.info("ml_models tablosu ikinci denemede başarıyla oluşturuldu.")
        
        conn.commit()
        conn.close()
        
        logger.info("Veritabanı başarıyla oluşturuldu")
        return True
    except Exception as e:
        logger.error(f"Veritabanı oluşturulurken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def save_favorite_stock(symbol):
    """
    Bir hisse senedini favorilere ekler
    
    Args:
        symbol (str): Hisse senedi sembolü
        
    Returns:
        bool: İşlem başarılıysa True, değilse False
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Sembolü düzenle
        symbol = symbol.upper().strip()
        
        # Mevcut kayıt kontrolü
        cursor.execute("SELECT * FROM favorite_stocks WHERE symbol = ?", (symbol,))
        if cursor.fetchone():
            logger.info(f"{symbol} zaten favorilerde")
            conn.close()
            return True
        
        # Yeni kayıt ekle
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO favorite_stocks (symbol, added_date) VALUES (?, ?)", (symbol, now))
        
        conn.commit()
        conn.close()
        
        logger.info(f"{symbol} favorilere eklendi")
        return True
    except Exception as e:
        logger.error(f"Favori eklenirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def remove_favorite_stock(symbol):
    """
    Bir hisse senedini favorilerden çıkarır
    
    Args:
        symbol (str): Hisse senedi sembolü
        
    Returns:
        bool: İşlem başarılıysa True, değilse False
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Sembolü düzenle
        symbol = symbol.upper().strip()
        
        # Silme işlemi
        cursor.execute("DELETE FROM favorite_stocks WHERE symbol = ?", (symbol,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"{symbol} favorilerden çıkarıldı")
        return True
    except Exception as e:
        logger.error(f"Favori çıkarılırken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def get_favorite_stocks():
    """
    Favori hisse senetlerini getirir
    
    Returns:
        list: Favori hisse sembollerinin listesi
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute("SELECT symbol FROM favorite_stocks ORDER BY added_date DESC")
        results = cursor.fetchall()
        
        conn.close()
        
        return [r[0] for r in results]
    except Exception as e:
        logger.error(f"Favoriler getirilirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def is_favorite_stock(symbol):
    """
    Bir hisse senedinin favorilerde olup olmadığını kontrol eder
    
    Args:
        symbol (str): Hisse senedi sembolü
        
    Returns:
        bool: Favorilerdeyse True, değilse False
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Sembolü düzenle
        symbol = symbol.upper().strip()
        
        cursor.execute("SELECT * FROM favorite_stocks WHERE symbol = ?", (symbol,))
        result = cursor.fetchone() is not None
        
        conn.close()
        
        return result
    except Exception as e:
        logger.error(f"Favori kontrolü sırasında hata: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def save_analysis_result(symbol, analysis_type, price, result_data, indicators=None, notes=None):
    """
    Bir hisse senedi için analiz sonucunu kaydeder
    
    Args:
        symbol (str): Hisse senedi sembolü
        analysis_type (str): Analiz tipi (teknik, temel, ml vb.)
        price (float): Analiz sırasındaki fiyat
        result_data (dict): Analiz sonuç verisi
        indicators (dict, optional): Teknik göstergeler
        notes (str, optional): Notlar
        
    Returns:
        bool: İşlem başarılıysa True, değilse False
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Sembolü düzenle
        symbol = symbol.upper().strip()
        
        # JSON'a dönüştür
        result_json = json.dumps(result_data, ensure_ascii=False)
        indicators_json = json.dumps(indicators, ensure_ascii=False) if indicators else None
        
        # Tarih
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Veriyi ekle
        cursor.execute("""
        INSERT INTO analysis_results 
        (symbol, analysis_type, analysis_date, price, result_data, indicators, notes) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (symbol, analysis_type, now, price, result_json, indicators_json, notes))
        
        conn.commit()
        conn.close()
        
        logger.info(f"{symbol} için {analysis_type} analiz sonucu kaydedildi")
        return True
    except Exception as e:
        logger.error(f"Analiz sonucu kaydedilirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def get_analysis_results(symbol=None, analysis_type=None, limit=20):
    """
    Analiz sonuçlarını getirir
    
    Args:
        symbol (str, optional): Hisse senedi sembolü
        analysis_type (str, optional): Analiz tipi
        limit (int, optional): Maksimum sonuç sayısı
        
    Returns:
        list: Analiz sonuçlarının listesi
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row  # Sonuçları dict olarak almak için
        cursor = conn.cursor()
        
        query = "SELECT * FROM analysis_results"
        params = []
        
        # Filtreler
        conditions = []
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol.upper().strip())
        
        if analysis_type:
            conditions.append("analysis_type = ?")
            params.append(analysis_type)
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += " ORDER BY analysis_date DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        # Sonuçları dict listesine dönüştür
        output = []
        for row in results:
            item = dict(row)
            # JSON alanları parse et
            item['result_data'] = json.loads(item['result_data']) if item['result_data'] else {}
            item['indicators'] = json.loads(item['indicators']) if item['indicators'] else {}
            output.append(item)
        
        conn.close()
        
        return output
    except Exception as e:
        logger.error(f"Analiz sonuçları getirilirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def save_ml_prediction(symbol, current_price, prediction_percentage, confidence_score, 
                      prediction_result, model_type, features_used, target_date=None):
    """
    ML tahmin sonucunu kaydeder
    
    Args:
        symbol (str): Hisse senedi sembolü
        current_price (float): Mevcut fiyat
        prediction_percentage (float): Tahmin edilen yüzde değişim
        confidence_score (float): Güven puanı
        prediction_result (str): Tahmin sonucu
        model_type (str): Model tipi
        features_used (list): Kullanılan özellikler
        target_date (str, optional): Hedef tarih
        
    Returns:
        bool: İşlem başarılıysa True, değilse False
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Sembolü düzenle
        symbol = symbol.upper().strip()
        
        # Tarihleri ayarla
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # JSON'a dönüştür
        features_json = json.dumps(features_used, ensure_ascii=False)
        
        # Veriyi ekle
        cursor.execute("""
        INSERT INTO ml_predictions 
        (symbol, prediction_date, target_date, current_price, prediction_percentage, 
         confidence_score, prediction_result, model_type, features_used) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol, now, target_date, current_price, prediction_percentage, 
              confidence_score, prediction_result, model_type, features_json))
        
        conn.commit()
        conn.close()
        
        logger.info(f"{symbol} için ML tahmin sonucu kaydedildi")
        return True
    except Exception as e:
        logger.error(f"ML tahmin sonucu kaydedilirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def get_ml_predictions(symbol=None, limit=50, include_verified=False):
    """
    ML tahmin sonuçlarını getirir
    
    Args:
        symbol (str, optional): Hisse senedi sembolü
        limit (int, optional): Maksimum sonuç sayısı
        include_verified (bool, optional): Doğrulanmış sonuçları da dahil et
        
    Returns:
        list: ML tahmin sonuçlarının listesi
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM ml_predictions"
        params = []
        
        # Filtreler
        conditions = []
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol.upper().strip())
        
        if not include_verified:
            conditions.append("was_correct = -1")
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += " ORDER BY prediction_date DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        # Sonuçları dict listesine dönüştür
        output = []
        for row in results:
            item = dict(row)
            # JSON alanları parse et
            item['features_used'] = json.loads(item['features_used']) if item['features_used'] else []
            output.append(item)
        
        conn.close()
        
        return output
    except Exception as e:
        logger.error(f"ML tahmin sonuçları getirilirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def update_ml_prediction_result(prediction_id, actual_result, was_correct):
    """
    ML tahmin sonucunu günceller (gerçekleşen sonuçla)
    
    Args:
        prediction_id (int): Tahmin ID'si
        actual_result (float): Gerçekleşen değer
        was_correct (int): Doğru olup olmadığı (1:doğru, 0:yanlış)
        
    Returns:
        bool: İşlem başarılıysa True, değilse False
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute("""
        UPDATE ml_predictions 
        SET actual_result = ?, was_correct = ? 
        WHERE id = ?
        """, (actual_result, was_correct, prediction_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"ML tahmin ID {prediction_id} güncellendi")
        return True
    except Exception as e:
        logger.error(f"ML tahmin sonucu güncellenirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def save_user_note(symbol, note_text):
    """
    Kullanıcı notunu kaydeder
    
    Args:
        symbol (str): Hisse senedi sembolü
        note_text (str): Not metni
        
    Returns:
        bool: İşlem başarılıysa True, değilse False
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Sembolü düzenle
        symbol = symbol.upper().strip()
        
        # Tarih
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Veriyi ekle
        cursor.execute("""
        INSERT INTO user_notes 
        (symbol, note_date, note_text) 
        VALUES (?, ?, ?)
        """, (symbol, now, note_text))
        
        conn.commit()
        conn.close()
        
        logger.info(f"{symbol} için kullanıcı notu kaydedildi")
        return True
    except Exception as e:
        logger.error(f"Kullanıcı notu kaydedilirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def get_user_notes(symbol=None):
    """
    Kullanıcı notlarını getirir
    
    Args:
        symbol (str, optional): Hisse senedi sembolü
        
    Returns:
        list: Notların listesi
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM user_notes"
        params = []
        
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol.upper().strip())
            
        query += " ORDER BY note_date DESC"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        output = [dict(row) for row in results]
        conn.close()
        
        return output
    except Exception as e:
        logger.error(f"Kullanıcı notları getirilirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def save_announcement(title, content, is_important=0):
    """
    Duyuru kaydeder
    
    Args:
        title (str): Duyuru başlığı
        content (str): Duyuru içeriği
        is_important (int, optional): Önemli mi? (1:önemli, 0:değil)
        
    Returns:
        bool: İşlem başarılıysa True, değilse False
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Tarih
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Veriyi ekle
        cursor.execute("""
        INSERT INTO announcements 
        (announcement_date, title, content, is_important) 
        VALUES (?, ?, ?, ?)
        """, (now, title, content, is_important))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Duyuru kaydedildi: {title}")
        return True
    except Exception as e:
        logger.error(f"Duyuru kaydedilirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def get_announcements(limit=10):
    """
    Duyuruları getirir
    
    Args:
        limit (int, optional): Maksimum duyuru sayısı
        
    Returns:
        list: Duyuruların listesi
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT * FROM announcements 
        ORDER BY announcement_date DESC LIMIT ?
        """, (limit,))
        
        results = cursor.fetchall()
        
        output = [dict(row) for row in results]
        conn.close()
        
        return output
    except Exception as e:
        logger.error(f"Duyurular getirilirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def delete_announcement(announcement_id):
    """
    Duyuru siler
    
    Args:
        announcement_id (int): Duyuru ID'si
        
    Returns:
        bool: İşlem başarılıysa True, değilse False
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM announcements WHERE id = ?", (announcement_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Duyuru silindi: ID {announcement_id}")
        return True
    except Exception as e:
        logger.error(f"Duyuru silinirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def get_ml_prediction_stats():
    """
    ML tahmin istatistiklerini hesaplar
    
    Returns:
        dict: İstatistikler
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Toplam tahmin sayısı
        cursor.execute("SELECT COUNT(*) FROM ml_predictions")
        total_predictions = cursor.fetchone()[0]
        
        # Doğrulanmış tahmin sayısı
        cursor.execute("SELECT COUNT(*) FROM ml_predictions WHERE was_correct != -1")
        verified_predictions = cursor.fetchone()[0]
        
        # Doğru tahmin sayısı
        cursor.execute("SELECT COUNT(*) FROM ml_predictions WHERE was_correct = 1")
        correct_predictions = cursor.fetchone()[0]
        
        # Başarı oranı
        success_rate = 0
        if verified_predictions > 0:
            success_rate = (correct_predictions / verified_predictions) * 100
        
        # En başarılı model
        cursor.execute("""
        SELECT model_type, COUNT(*) as count 
        FROM ml_predictions 
        WHERE was_correct = 1 
        GROUP BY model_type 
        ORDER BY count DESC 
        LIMIT 1
        """)
        best_model_result = cursor.fetchone()
        best_model = best_model_result[0] if best_model_result else "Veri yok"
        
        conn.close()
        
        return {
            "total_predictions": total_predictions,
            "verified_predictions": verified_predictions,
            "correct_predictions": correct_predictions,
            "success_rate": success_rate,
            "best_model": best_model
        }
    except Exception as e:
        logger.error(f"ML istatistikleri hesaplanırken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "total_predictions": 0,
            "verified_predictions": 0,
            "correct_predictions": 0,
            "success_rate": 0,
            "best_model": "Hata"
        }

def get_detailed_analysis_history(symbol=None, analysis_type=None, start_date=None, end_date=None, limit=100):
    """
    Detaylı analiz geçmişini döndürür, çeşitli filtreleme seçenekleriyle
    
    Args:
        symbol (str, optional): Hisse senedi sembolü
        analysis_type (str, optional): Analiz tipi (teknik, temel, ml vb.)
        start_date (str, optional): Başlangıç tarihi (YYYY-MM-DD formatında)
        end_date (str, optional): Bitiş tarihi (YYYY-MM-DD formatında)
        limit (int, optional): Maksimum sonuç sayısı
        
    Returns:
        list: Detaylı analiz sonuçlarının listesi
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM analysis_results"
        params = []
        
        # Filtreler
        conditions = []
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol.upper().strip())
        
        if analysis_type:
            conditions.append("analysis_type = ?")
            params.append(analysis_type)
            
        if start_date:
            conditions.append("analysis_date >= ?")
            params.append(start_date + " 00:00:00")
            
        if end_date:
            conditions.append("analysis_date <= ?")
            params.append(end_date + " 23:59:59")
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += " ORDER BY analysis_date DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        # Sonuçları dict listesine dönüştür
        output = []
        for row in results:
            item = dict(row)
            # JSON alanları parse et
            item['result_data'] = json.loads(item['result_data']) if item['result_data'] else {}
            item['indicators'] = json.loads(item['indicators']) if item['indicators'] else {}
            output.append(item)
        
        conn.close()
        
        return output
    except Exception as e:
        logger.error(f"Detaylı analiz geçmişi getirilirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def export_analysis_results(symbol, format="csv", analysis_type=None, start_date=None, end_date=None):
    """
    Belirli bir hisse senedi için analiz sonuçlarını dışa aktarır
    
    Args:
        symbol (str): Hisse senedi sembolü
        format (str): Dışa aktarma formatı ("csv" veya "json")
        analysis_type (str, optional): Analiz tipi
        start_date (str, optional): Başlangıç tarihi (YYYY-MM-DD formatında)
        end_date (str, optional): Bitiş tarihi (YYYY-MM-DD formatında)
        
    Returns:
        str or bytes: CSV string veya JSON string
    """
    try:
        # Detaylı analiz sonuçlarını al
        results = get_detailed_analysis_history(
            symbol=symbol, 
            analysis_type=analysis_type, 
            start_date=start_date, 
            end_date=end_date, 
            limit=1000
        )
        
        if not results:
            return None
        
        if format.lower() == "csv":
            # Düzleştirilmiş veri yapısı oluştur
            flat_data = []
            for result in results:
                flat_item = {
                    "id": result["id"],
                    "symbol": result["symbol"],
                    "analysis_type": result["analysis_type"],
                    "analysis_date": result["analysis_date"],
                    "price": result["price"],
                    "notes": result["notes"]
                }
                
                # result_data içindeki verileri düzleştir
                for key, value in result["result_data"].items():
                    if isinstance(value, (int, float, str, bool)) or value is None:
                        flat_item[f"result_{key}"] = value
                
                # indicators içindeki verileri düzleştir
                for key, value in result["indicators"].items():
                    if isinstance(value, (int, float, str, bool)) or value is None:
                        flat_item[f"indicator_{key}"] = value
                
                flat_data.append(flat_item)
            
            # DataFrame'e dönüştür ve CSV'ye çevir
            df = pd.DataFrame(flat_data)
            return df.to_csv(index=False)
        
        elif format.lower() == "json":
            # JSON formatına dönüştür
            return json.dumps(results, ensure_ascii=False, indent=2)
        
        else:
            raise ValueError(f"Desteklenmeyen format: {format}")
    
    except Exception as e:
        logger.error(f"Analiz sonuçları dışa aktarılırken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def compare_analysis_results(analysis_id1, analysis_id2):
    """
    İki farklı analiz sonucunu karşılaştırır
    
    Args:
        analysis_id1 (int): Birinci analiz sonucunun ID'si
        analysis_id2 (int): İkinci analiz sonucunun ID'si
        
    Returns:
        dict: Karşılaştırma sonuçları
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Her iki analizi de getir
        cursor.execute("SELECT * FROM analysis_results WHERE id = ?", (analysis_id1,))
        analysis1 = cursor.fetchone()
        
        cursor.execute("SELECT * FROM analysis_results WHERE id = ?", (analysis_id2,))
        analysis2 = cursor.fetchone()
        
        conn.close()
        
        if not analysis1 or not analysis2:
            return {"error": "Bir veya her iki analiz bulunamadı"}
        
        # Dict'e dönüştür
        analysis1_dict = dict(analysis1)
        analysis2_dict = dict(analysis2)
        
        # JSON verilerini parse et
        analysis1_dict["result_data"] = json.loads(analysis1_dict["result_data"]) if analysis1_dict["result_data"] else {}
        analysis1_dict["indicators"] = json.loads(analysis1_dict["indicators"]) if analysis1_dict["indicators"] else {}
        
        analysis2_dict["result_data"] = json.loads(analysis2_dict["result_data"]) if analysis2_dict["result_data"] else {}
        analysis2_dict["indicators"] = json.loads(analysis2_dict["indicators"]) if analysis2_dict["indicators"] else {}
        
        # Farklılıkları hesapla
        differences = {
            "basic_info": {
                "symbol": [analysis1_dict["symbol"], analysis2_dict["symbol"]],
                "analysis_type": [analysis1_dict["analysis_type"], analysis2_dict["analysis_type"]],
                "analysis_date": [analysis1_dict["analysis_date"], analysis2_dict["analysis_date"]],
                "price": [analysis1_dict["price"], analysis2_dict["price"]],
                "price_change": round((analysis2_dict["price"] - analysis1_dict["price"]) / analysis1_dict["price"] * 100, 2) if analysis1_dict["price"] else None
            },
            "result_data": {},
            "indicators": {}
        }
        
        # result_data karşılaştırması
        all_keys = set(analysis1_dict["result_data"].keys()) | set(analysis2_dict["result_data"].keys())
        for key in all_keys:
            val1 = analysis1_dict["result_data"].get(key)
            val2 = analysis2_dict["result_data"].get(key)
            
            if val1 != val2:
                differences["result_data"][key] = [val1, val2]
                
                # Sayısal değerlerin değişim yüzdesi
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)) and val1 != 0:
                    pct_change = round((val2 - val1) / abs(val1) * 100, 2)
                    differences["result_data"][f"{key}_pct_change"] = f"{pct_change}%"
        
        # indicators karşılaştırması
        all_keys = set(analysis1_dict["indicators"].keys()) | set(analysis2_dict["indicators"].keys())
        for key in all_keys:
            val1 = analysis1_dict["indicators"].get(key)
            val2 = analysis2_dict["indicators"].get(key)
            
            if val1 != val2:
                differences["indicators"][key] = [val1, val2]
                
                # Sayısal değerlerin değişim yüzdesi
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)) and val1 != 0:
                    pct_change = round((val2 - val1) / abs(val1) * 100, 2)
                    differences["indicators"][f"{key}_pct_change"] = f"{pct_change}%"
        
        return differences
    
    except Exception as e:
        logger.error(f"Analiz sonuçları karşılaştırılırken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def delete_analysis_result(analysis_id):
    """
    Bir analiz sonucunu veritabanından siler
    
    Args:
        analysis_id (int): Silinecek analiz sonucunun ID'si
        
    Returns:
        bool: İşlem başarılıysa True, değilse False
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Silme işlemi
        cursor.execute("DELETE FROM analysis_results WHERE id = ?", (analysis_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"ID: {analysis_id} analiz sonucu silindi")
        return True
    except Exception as e:
        logger.error(f"Analiz sonucu silinirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def update_analysis_price(analysis_id, price):
    """
    Bir analiz sonucunun fiyat değerini günceller
    
    Args:
        analysis_id (int): Güncellenecek analiz sonucunun ID'si
        price (float): Yeni fiyat değeri
        
    Returns:
        bool: İşlem başarılıysa True, değilse False
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Fiyat değerini güncelle
        cursor.execute("""
        UPDATE analysis_results 
        SET price = ? 
        WHERE id = ?
        """, (price, analysis_id))
        
        # Aynı zamanda result_data içindeki fiyat bilgisini de güncelle
        cursor.execute("SELECT result_data FROM analysis_results WHERE id = ?", (analysis_id,))
        result = cursor.fetchone()
        
        if result and result[0]:
            result_data = json.loads(result[0])
            
            # Fiyat bilgilerini güncelle
            if "last_price" in result_data:
                result_data["last_price"] = price
            
            if "current_price" in result_data:
                result_data["current_price"] = price
            
            # Güncellenmiş veriyi kaydet
            result_data_json = json.dumps(result_data, ensure_ascii=False)
            cursor.execute("""
            UPDATE analysis_results 
            SET result_data = ? 
            WHERE id = ?
            """, (result_data_json, analysis_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Analiz ID: {analysis_id} fiyat değeri {price} olarak güncellendi")
        return True
    except Exception as e:
        logger.error(f"Analiz fiyatı güncellenirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Portföy işlemleri için fonksiyonlar
def add_portfolio_stock(symbol, purchase_date, quantity, purchase_price, notes=None, sector=None, target_price=None, stop_loss=None):
    """
    Portföye yeni bir hisse senedi ekler
    
    Args:
        symbol (str): Hisse senedi sembolü
        purchase_date (str): Alım tarihi (YYYY-MM-DD)
        quantity (float): Hisse adedi
        purchase_price (float): Alım fiyatı
        notes (str, optional): Notlar
        sector (str, optional): Sektör
        target_price (float, optional): Hedef satış fiyatı
        stop_loss (float, optional): Zarar kesme seviyesi
        
    Returns:
        bool: İşlem başarılıysa True, değilse False
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Sembolü düzenle
        symbol = symbol.upper().strip()
        
        # Veriyi ekle
        cursor.execute(
            "INSERT INTO portfolio (symbol, purchase_date, quantity, purchase_price, notes, sector, target_price, stop_loss) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (symbol, purchase_date, quantity, purchase_price, notes, sector, target_price, stop_loss)
        )
        
        # İşlem kaydını ekle
        total_amount = quantity * purchase_price
        cursor.execute(
            "INSERT INTO portfolio_transactions (symbol, transaction_date, transaction_type, quantity, price, total_amount, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (symbol, purchase_date, "ALIŞ", quantity, purchase_price, total_amount, notes)
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"{symbol} portföye eklendi. Adet: {quantity}, Fiyat: {purchase_price}")
        return True
    except Exception as e:
        logger.error(f"Portföye hisse eklenirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def update_portfolio_stock(portfolio_id, quantity=None, purchase_price=None, notes=None, target_price=None, stop_loss=None, is_active=None):
    """
    Portföydeki bir hisseyi günceller
    
    Args:
        portfolio_id (int): Portföy kaydının ID'si
        quantity (float, optional): Yeni hisse adedi
        purchase_price (float, optional): Yeni alım fiyatı
        notes (str, optional): Yeni notlar
        target_price (float, optional): Yeni hedef fiyat
        stop_loss (float, optional): Yeni zarar kesme seviyesi
        is_active (int, optional): Aktiflik durumu (1: aktif, 0: pasif)
        
    Returns:
        bool: İşlem başarılıysa True, değilse False
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Mevcut kaydı al
        cursor.execute("SELECT * FROM portfolio WHERE id = ?", (portfolio_id,))
        existing_record = cursor.fetchone()
        
        if not existing_record:
            logger.error(f"Portföy ID'si bulunamadı: {portfolio_id}")
            conn.close()
            return False
        
        # Güncellenecek alanları belirle
        update_fields = []
        update_values = []
        
        if quantity is not None:
            update_fields.append("quantity = ?")
            update_values.append(quantity)
        
        if purchase_price is not None:
            update_fields.append("purchase_price = ?")
            update_values.append(purchase_price)
        
        if notes is not None:
            update_fields.append("notes = ?")
            update_values.append(notes)
        
        if target_price is not None:
            update_fields.append("target_price = ?")
            update_values.append(target_price)
        
        if stop_loss is not None:
            update_fields.append("stop_loss = ?")
            update_values.append(stop_loss)
        
        if is_active is not None:
            update_fields.append("is_active = ?")
            update_values.append(is_active)
        
        if not update_fields:
            logger.info("Güncellenecek alan belirtilmedi")
            conn.close()
            return True
        
        # Güncelleme sorgusu
        update_query = f"UPDATE portfolio SET {', '.join(update_fields)} WHERE id = ?"
        update_values.append(portfolio_id)
        
        cursor.execute(update_query, update_values)
        conn.commit()
        conn.close()
        
        logger.info(f"Portföy kaydı güncellendi. ID: {portfolio_id}")
        return True
    except Exception as e:
        logger.error(f"Portföy güncellenirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def delete_portfolio_stock(portfolio_id):
    """
    Portföyden bir hisseyi siler
    
    Args:
        portfolio_id (int): Portföy kaydının ID'si
        
    Returns:
        bool: İşlem başarılıysa True, değilse False
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Kaydı sil
        cursor.execute("DELETE FROM portfolio WHERE id = ?", (portfolio_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Portföy kaydı silindi. ID: {portfolio_id}")
        return True
    except Exception as e:
        logger.error(f"Portföy kaydı silinirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def get_portfolio_stocks(only_active=True):
    """
    Portföydeki hisseleri getirir
    
    Args:
        only_active (bool): Sadece aktif hisseleri getir
        
    Returns:
        list: Portföy kayıtlarının listesi
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row  # Sonuçları sözlük formatında al
        cursor = conn.cursor()
        
        if only_active:
            cursor.execute("SELECT * FROM portfolio WHERE is_active = 1 ORDER BY symbol")
        else:
            cursor.execute("SELECT * FROM portfolio ORDER BY is_active DESC, symbol")
            
        results = cursor.fetchall()
        
        # Sonuçları sözlük listesine dönüştür
        portfolio_list = [dict(row) for row in results]
        
        conn.close()
        return portfolio_list
    except Exception as e:
        logger.error(f"Portföy getirilirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def get_portfolio_stock(portfolio_id):
    """
    Belirli bir portföy kaydını getirir
    
    Args:
        portfolio_id (int): Portföy kaydının ID'si
        
    Returns:
        dict: Portföy kaydı veya bulunamazsa None
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM portfolio WHERE id = ?", (portfolio_id,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return dict(result)
        return None
    except Exception as e:
        logger.error(f"Portföy kaydı getirilirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def add_portfolio_transaction(symbol, transaction_date, transaction_type, quantity, price, commission=0, notes=None):
    """
    Portföye yeni bir işlem ekler (alış/satış)
    
    Args:
        symbol (str): Hisse senedi sembolü
        transaction_date (str): İşlem tarihi (YYYY-MM-DD)
        transaction_type (str): İşlem tipi (ALIŞ/SATIŞ)
        quantity (float): Hisse adedi
        price (float): İşlem fiyatı
        commission (float, optional): Komisyon tutarı
        notes (str, optional): Notlar
        
    Returns:
        bool: İşlem başarılıysa True, değilse False
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Sembolü düzenle
        symbol = symbol.upper().strip()
        
        # Toplam tutarı hesapla
        total_amount = quantity * price
        
        # İşlemi ekle
        cursor.execute(
            "INSERT INTO portfolio_transactions (symbol, transaction_date, transaction_type, quantity, price, commission, total_amount, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (symbol, transaction_date, transaction_type.upper(), quantity, price, commission, total_amount, notes)
        )
        
        # Eğer bu bir SATIŞ işlemi ise portföydeki hisseyi güncelle
        if transaction_type.upper() == "SATIŞ":
            # Portföyde varolan hisseleri kontrol et
            cursor.execute("SELECT id, quantity FROM portfolio WHERE symbol = ? AND is_active = 1", (symbol,))
            portfolio_records = cursor.fetchall()
            
            remaining_quantity = quantity
            
            for record in portfolio_records:
                record_id, record_quantity = record
                
                if remaining_quantity <= 0:
                    break
                
                if record_quantity <= remaining_quantity:
                    # Bu kaydı tamamen sat
                    cursor.execute("UPDATE portfolio SET is_active = 0 WHERE id = ?", (record_id,))
                    remaining_quantity -= record_quantity
                else:
                    # Bu kaydın bir kısmını sat
                    new_quantity = record_quantity - remaining_quantity
                    cursor.execute("UPDATE portfolio SET quantity = ? WHERE id = ?", (new_quantity, record_id))
                    remaining_quantity = 0
        
        conn.commit()
        conn.close()
        
        logger.info(f"Portföy işlemi eklendi. Sembol: {symbol}, İşlem: {transaction_type}, Adet: {quantity}")
        return True
    except Exception as e:
        logger.error(f"Portföy işlemi eklenirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def get_portfolio_transactions(symbol=None, start_date=None, end_date=None, transaction_type=None):
    """
    Portföy işlemlerini getirir
    
    Args:
        symbol (str, optional): Hisse senedi sembolü
        start_date (str, optional): Başlangıç tarihi (YYYY-MM-DD)
        end_date (str, optional): Bitiş tarihi (YYYY-MM-DD)
        transaction_type (str, optional): İşlem tipi (ALIŞ/SATIŞ)
        
    Returns:
        list: İşlem kayıtlarının listesi
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Sorgu parametreleri
        query = "SELECT * FROM portfolio_transactions WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol.upper().strip())
        
        if start_date:
            query += " AND transaction_date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND transaction_date <= ?"
            params.append(end_date)
        
        if transaction_type:
            query += " AND transaction_type = ?"
            params.append(transaction_type.upper())
        
        query += " ORDER BY transaction_date DESC"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        # Sonuçları sözlük listesine dönüştür
        transactions = [dict(row) for row in results]
        
        conn.close()
        return transactions
    except Exception as e:
        logger.error(f"Portföy işlemleri getirilirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def get_portfolio_performance():
    """
    Portföy performansını hesaplar
    
    Returns:
        dict: Portföy performans bilgileri
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Aktif portföy hisselerini al
        cursor.execute("SELECT * FROM portfolio WHERE is_active = 1")
        stocks = cursor.fetchall()
        
        if not stocks:
            return {
                "total_investment": 0, 
                "current_value": 0, 
                "total_gain_loss": 0, 
                "total_gain_loss_percentage": 0,
                "cash": 60.67,  # Varsayılan nakit değeri
                "investment_fund": 15254.17,  # Varsayılan yatırım fonu değeri 
                "stocks": []
            }
        
        from data.stock_data import get_stock_data_cached
        
        # Sonuçları sözlük listesine dönüştür
        stock_list = []
        total_investment = 0
        total_current_value = 0
        
        for stock in stocks:
            symbol = stock["symbol"]
            quantity = stock["quantity"]
            purchase_price = stock["purchase_price"]
            
            # Alım maliyeti
            investment = quantity * purchase_price
            
            # Güncel fiyatı al
            stock_data = get_stock_data_cached(symbol, period="1d")
            
            if not stock_data.empty:
                # Güncel fiyat ve değer
                current_price = stock_data['Close'].iloc[-1]
                current_value = quantity * current_price
                
                # Kâr/zarar hesaplama
                gain_loss = current_value - investment
                gain_loss_percentage = (gain_loss / investment * 100) if investment > 0 else 0
                
                # Hisse bilgilerini listeye ekle
                stock_list.append({
                    "id": stock["id"],
                    "symbol": symbol,
                    "quantity": quantity,
                    "purchase_price": purchase_price,
                    "current_price": current_price,
                    "investment": investment,
                    "current_value": current_value,
                    "gain_loss": gain_loss,
                    "gain_loss_percentage": gain_loss_percentage
                })
                
                # Toplam değerleri güncelle
                total_investment += investment
                total_current_value += current_value
            else:
                # Fiyat alınamazsa alım değerini kullan
                stock_list.append({
                    "id": stock["id"],
                    "symbol": symbol,
                    "quantity": quantity,
                    "purchase_price": purchase_price,
                    "current_price": purchase_price,
                    "investment": investment,
                    "current_value": investment,
                    "gain_loss": 0,
                    "gain_loss_percentage": 0
                })
                
                # Toplam değerleri güncelle
                total_investment += investment
                total_current_value += investment
        
        # Toplam kâr/zarar
        total_gain_loss = total_current_value - total_investment
        total_gain_loss_percentage = (total_gain_loss / total_investment * 100) if total_investment > 0 else 0
        
        # Nakit ve yatırım fonu değerlerini ekle
        cash_value = 60.67  # Varsayılan nakit değeri
        investment_fund_value = 15254.17  # Varsayılan yatırım fonu değeri
        
        # Portföy sonuçları
        portfolio_results = {
            "total_investment": total_investment,
            "current_value": total_current_value,
            "total_gain_loss": total_gain_loss,
            "total_gain_loss_percentage": total_gain_loss_percentage,
            "cash": cash_value,
            "investment_fund": investment_fund_value,
            "stocks": stock_list
        }
        
        conn.close()
        return portfolio_results
    except Exception as e:
        logger.error(f"Portföy performansı hesaplanırken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "total_investment": 0, 
            "current_value": 0, 
            "total_gain_loss": 0, 
            "total_gain_loss_percentage": 0, 
            "cash": 60.67,
            "investment_fund": 15254.17,
            "stocks": []
        }

def get_portfolio_sector_distribution():
    """
    Portföyün sektör dağılımını hesaplar
    
    Returns:
        dict: Sektör dağılımı (sektör adı -> değer)
    """
    try:
        # Aktif portföy hisselerini al
        portfolio = get_portfolio_stocks(only_active=True)
        
        if not portfolio:
            return {}
        
        from data.stock_data import get_stock_data_cached, get_company_info
        
        sector_values = {}
        unknown_sector_value = 0
        
        for stock in portfolio:
            symbol = stock["symbol"]
            quantity = stock["quantity"]
            sector = stock["sector"]
            
            # Sektör bilgisi yoksa şirket bilgisinden al
            if not sector:
                company_info = get_company_info(symbol)
                sector = company_info.get('sector', 'Diğer')
            
            # Hala yoksa "Diğer" olarak ayarla
            if not sector:
                sector = "Diğer"
            
            # Güncel fiyatı al
            stock_data = get_stock_data_cached(symbol, period="1d")
            
            if not stock_data.empty:
                current_price = stock_data['Close'].iloc[-1]
                current_value = quantity * current_price
                
                if sector in sector_values:
                    sector_values[sector] += current_value
                else:
                    sector_values[sector] = current_value
            else:
                # Fiyat alınamazsa alım değerini kullan
                purchase_value = quantity * stock["purchase_price"]
                
                if sector in sector_values:
                    sector_values[sector] += purchase_value
                else:
                    sector_values[sector] = purchase_value
        
        return sector_values
    except Exception as e:
        logger.error(f"Portföy sektör dağılımı hesaplanırken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return {}

# --- Stock Info Fonksiyonları ---

def get_stock_info_from_db(symbol):
    """Belirli bir hissenin bilgisini veritabanından alır.
    
    Args:
        symbol (str): Hisse sembolü.
        
    Returns:
        dict: Hisse bilgileri veya None.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row # Satırları dict gibi erişilebilir yap
        cursor = conn.cursor()
        cursor.execute("SELECT symbol, name, sector_en, sector_tr FROM stock_info WHERE symbol = ?", (symbol.upper(),))
        result = cursor.fetchone()
        conn.close()
        return dict(result) if result else None
    except Exception as e:
        logger.error(f"DB'den hisse bilgisi alınırken hata ({symbol}): {str(e)}")
        return None

def save_stock_info_to_db(symbol, name, sector_en):
    """Çekilen hisse bilgisini veritabanına kaydeder (Türkçe çeviri ile).
    
    Args:
        symbol (str): Hisse sembolü.
        name (str): Şirket adı.
        sector_en (str): İngilizce sektör adı.
        
    Returns:
        str: Kaydedilen Türkçe sektör adı veya İngilizce adı (çeviri yoksa).
    """
    symbol = symbol.upper()
    sector_tr = SECTOR_TRANSLATIONS.get(sector_en, sector_en) # Çeviri yoksa İngilizce kalsın
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO stock_info (symbol, name, sector_en, sector_tr, last_updated)
        VALUES (?, ?, ?, ?, ?)
        """, (symbol, name, sector_en, sector_tr, now))
        conn.commit()
        conn.close()
        logger.info(f"Hisse bilgisi DB'ye kaydedildi/güncellendi: {symbol}")
        return sector_tr
    except Exception as e:
        logger.error(f"DB'ye hisse bilgisi kaydedilirken hata ({symbol}): {str(e)}")
        return sector_en # Hata durumunda İngilizce adı döndür

def get_or_fetch_stock_info(symbol):
    """Hisse bilgisini önce DB'den alır, yoksa API'dan çeker, kaydeder ve döndürür.
    
    Args:
        symbol (str): Hisse sembolü.
        
    Returns:
        dict: Hisse bilgileri ({'name': ..., 'sector_tr': ...}) veya None.
    """
    symbol = symbol.upper()
    # 1. DB'den kontrol et
    stock_info = get_stock_info_from_db(symbol)
    if stock_info:
        return {"name": stock_info["name"], "sector_tr": stock_info["sector_tr"]}
    
    # 2. DB'de yoksa API'dan çek
    try:
        from .stock_data import get_company_info # Dairesel import önlemek için burada import et
        
        logger.info(f"API'dan hisse bilgisi çekiliyor: {symbol}")
        api_info = get_company_info(symbol)
        
        if api_info and isinstance(api_info, dict):
            name = api_info.get('longName') or api_info.get('shortName', symbol) # İsim bulmaya çalış
            sector_en = api_info.get('sector', 'Bilinmiyor')
            
            # 3. DB'ye kaydet (Türkçe çeviri ile)
            sector_tr = save_stock_info_to_db(symbol, name, sector_en)
            return {"name": name, "sector_tr": sector_tr}
        else:
            logger.warning(f"API'dan hisse bilgisi alınamadı veya format hatalı: {symbol}")
            # API'dan bilgi alınamasa bile 'Bilinmiyor' olarak kaydedelim mi?
            # save_stock_info_to_db(symbol, symbol, 'Bilinmiyor') # Opsiyonel
            return {"name": symbol, "sector_tr": "Bilinmiyor"} # Varsayılan döndür
            
    except ImportError as ie:
         logger.error(f"stock_data modülü import edilemedi: {str(ie)}")
         return None
    except Exception as e:
        logger.error(f"API'dan hisse bilgisi alınırken hata ({symbol}): {str(e)}")
        # Hata durumunda da 'Bilinmiyor' olarak kaydedelim mi?
        # save_stock_info_to_db(symbol, symbol, 'Bilinmiyor') # Opsiyonel
        return {"name": symbol, "sector_tr": "Bilinmiyor"} # Varsayılan döndür 

def save_ml_model(symbol, model_type, model_data, features_used=None, performance_metrics=None):
    """Eğitilmiş ML modelini veritabanına kaydeder.
    
    Args:
        symbol (str): Hisse sembolü.
        model_type (str): Model tipi (RandomForest, XGBoost, LightGBM, Ensemble, Hibrit).
        model_data (bytes): Pickle ile serileştirilmiş model verisi.
        features_used (list, optional): Kullanılan özellikler.
        performance_metrics (dict, optional): Model performans metrikleri.
        
    Returns:
        bool: İşlem başarılıysa True, değilse False.
    """
    try:
        symbol = symbol.upper().strip()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        features_json = json.dumps(features_used) if features_used else None
        metrics_json = json.dumps(performance_metrics) if performance_metrics else None
        
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Daha önce kaydedilmiş aynı sembol ve model tipi var mı kontrol et
        cursor.execute(
            "SELECT id FROM ml_models WHERE symbol = ? AND model_type = ? AND is_active = 1", 
            (symbol, model_type)
        )
        existing_model = cursor.fetchone()
        
        if existing_model:
            # Varsa güncelle
            cursor.execute("""
            UPDATE ml_models 
            SET model_data = ?, features_used = ?, last_update_date = ?, performance_metrics = ?
            WHERE id = ?
            """, (model_data, features_json, now, metrics_json, existing_model[0]))
        else:
            # Yoksa yeni ekle
            cursor.execute("""
            INSERT INTO ml_models 
            (symbol, model_type, model_data, features_used, training_date, last_update_date, performance_metrics, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1)
            """, (symbol, model_type, model_data, features_json, now, now, metrics_json))
        
        conn.commit()
        conn.close()
        logger.info(f"{symbol} için {model_type} modeli veritabanına kaydedildi.")
        return True
    except Exception as e:
        logger.error(f"ML modeli kaydedilirken hata ({symbol}, {model_type}): {str(e)}")
        logger.error(traceback.format_exc())
        return False

def load_ml_model(symbol, model_type=None):
    """Veritabanından ML modelini yükler.
    
    Args:
        symbol (str): Hisse sembolü.
        model_type (str, optional): Model tipi. Belirtilmezse tüm modeller döndürülür.
        
    Returns:
        dict veya None: Model verileri içeren sözlük veya işlem başarısızsa None.
    """
    try:
        symbol = symbol.upper().strip()
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        if model_type:
            # Belirli bir model tipi için sorgu
            cursor.execute("""
            SELECT model_data, features_used, training_date, last_update_date, performance_metrics, model_type
            FROM ml_models 
            WHERE symbol = ? AND model_type = ? AND is_active = 1
            """, (symbol, model_type))
            result = cursor.fetchone()
            
            if result:
                model_data, features_json, training_date, last_update_date, metrics_json, model_type = result
                features = json.loads(features_json) if features_json else None
                metrics = json.loads(metrics_json) if metrics_json else None
                
                model_info = {
                    'model_data': model_data,
                    'features': features,
                    'training_date': training_date,
                    'last_update_date': last_update_date,
                    'metrics': metrics,
                    'model_type': model_type
                }
                conn.close()
                return model_info
        else:
            # Tüm model tipleri için sorgu
            cursor.execute("""
            SELECT model_data, features_used, training_date, last_update_date, performance_metrics, model_type
            FROM ml_models 
            WHERE symbol = ? AND is_active = 1
            """, (symbol,))
            results = cursor.fetchall()
            
            if results:
                models = {}
                for result in results:
                    model_data, features_json, training_date, last_update_date, metrics_json, model_type = result
                    features = json.loads(features_json) if features_json else None
                    metrics = json.loads(metrics_json) if metrics_json else None
                    
                    models[model_type] = {
                        'model_data': model_data,
                        'features': features,
                        'training_date': training_date,
                        'last_update_date': last_update_date,
                        'metrics': metrics
                    }
                conn.close()
                return models
        
        conn.close()
        logger.info(f"{symbol} için {model_type if model_type else 'hiçbir'} model bulunamadı.")
        return None
    except Exception as e:
        logger.error(f"ML modeli yüklenirken hata ({symbol}, {model_type}): {str(e)}")
        logger.error(traceback.format_exc())
        return None

def get_model_update_status(symbol, days_threshold=7):
    """Modelin güncelleme durumunu kontrol eder.
    
    Args:
        symbol (str): Hisse sembolü.
        days_threshold (int): Modelin güncel sayılması için gün eşiği.
        
    Returns:
        dict: Modellerin güncelleme durumları.
    """
    try:
        symbol = symbol.upper().strip()
        now = datetime.now()
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT model_type, last_update_date 
        FROM ml_models 
        WHERE symbol = ? AND is_active = 1
        """, (symbol,))
        results = cursor.fetchall()
        
        update_status = {}
        for model_type, last_update_date in results:
            last_update = datetime.strptime(last_update_date, "%Y-%m-%d %H:%M:%S")
            days_since_update = (now - last_update).days
            
            update_status[model_type] = {
                'last_update': last_update_date,
                'days_since_update': days_since_update,
                'needs_update': days_since_update > days_threshold
            }
        
        conn.close()
        return update_status
    except Exception as e:
        logger.error(f"Model güncelleme durumu kontrol edilirken hata ({symbol}): {str(e)}")
        logger.error(traceback.format_exc())
        return {} 