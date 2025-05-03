"""
WebSocket entegrasyonu ile gerçek zamanlı veri izleme modülü.
Borsa İstanbul ve diğer piyasalar için anlık veri akışı sağlar.
"""

import asyncio
import json
import logging
import threading
import time
import websocket
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
import traceback
import queue
import random

# Loglama yapılandırması
logger = logging.getLogger(__name__)

# WebSocket istemcisinin hazır olup olmadığını kontrol et
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logger.warning("websocket-client kütüphanesi yüklenemedi. WebSocket bağlantısı kullanılamayacak.")

# WebSocket bağlantı bilgileri (burada örnek bir bağlantı adresi verilmiştir)
WEBSOCKET_URL = "wss://example.com/bist/websocket"  # Gerçek bağlantı için değiştirilmeli
API_KEY = "your_api_key_here"  # API anahtarınız

# Veri depolama
realtime_data = {}
data_callbacks = {}
subscribers = {}
message_queue = queue.Queue()

class WebSocketClient:
    """WebSocket istemcisi sınıfı."""
    
    def __init__(self, url=WEBSOCKET_URL, api_key=API_KEY):
        """
        WebSocket istemcisini başlatır.
        
        Args:
            url (str): WebSocket bağlantı adresi
            api_key (str): API anahtarı
        """
        self.url = url
        self.api_key = api_key
        self.ws = None
        self.thread = None
        self.running = False
        self.connected = False
        self.reconnect_delay = 5  # Yeniden bağlanma gecikmesi (saniye)
        self.max_reconnect_delay = 300  # Maksimum yeniden bağlanma gecikmesi (saniye)
        self.subscriptions = set()
        
    def connect(self):
        """WebSocket bağlantısını başlatır."""
        if not WEBSOCKET_AVAILABLE:
            logger.error("WebSocket bağlantısı için websocket-client kütüphanesi gereklidir.")
            return False
        
        try:
            # WebSocket'i başlat
            websocket.enableTrace(False)
            self.ws = websocket.WebSocketApp(
                self.url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                header={"X-API-Key": self.api_key}
            )
            
            # Bağlantı iş parçacığını başlat
            self.thread = threading.Thread(target=self.ws.run_forever)
            self.thread.daemon = True
            self.thread.start()
            
            self.running = True
            logger.info("WebSocket istemcisi başlatıldı.")
            return True
        
        except Exception as e:
            logger.error(f"WebSocket bağlantısı başlatılırken hata: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def disconnect(self):
        """WebSocket bağlantısını kapatır."""
        if self.ws is not None:
            self.running = False
            self.ws.close()
            logger.info("WebSocket bağlantısı kapatıldı.")
    
    def subscribe(self, symbol):
        """
        Belirli bir sembol için abonelik başlatır.
        
        Args:
            symbol (str): Abone olunacak sembol
            
        Returns:
            bool: Başarılı ise True, değilse False
        """
        if not self.connected:
            logger.warning("WebSocket bağlantısı yok. Önce connect() çağrılmalı.")
            return False
        
        try:
            # Abonelik mesajı gönder
            subscription_message = {
                "type": "subscribe",
                "symbol": symbol
            }
            
            self.ws.send(json.dumps(subscription_message))
            self.subscriptions.add(symbol)
            
            logger.info(f"{symbol} sembolüne abone olundu.")
            return True
        
        except Exception as e:
            logger.error(f"{symbol} sembolüne abone olunurken hata: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def unsubscribe(self, symbol):
        """
        Belirli bir sembol için aboneliği sonlandırır.
        
        Args:
            symbol (str): Aboneliği sonlandırılacak sembol
            
        Returns:
            bool: Başarılı ise True, değilse False
        """
        if not self.connected:
            logger.warning("WebSocket bağlantısı yok.")
            return False
        
        try:
            # Abonelik sonlandırma mesajı gönder
            unsubscription_message = {
                "type": "unsubscribe",
                "symbol": symbol
            }
            
            self.ws.send(json.dumps(unsubscription_message))
            
            if symbol in self.subscriptions:
                self.subscriptions.remove(symbol)
            
            logger.info(f"{symbol} sembolünün aboneliği sonlandırıldı.")
            return True
        
        except Exception as e:
            logger.error(f"{symbol} sembolünün aboneliği sonlandırılırken hata: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def on_open(self, ws):
        """
        WebSocket bağlantısı açıldığında çağrılır.
        
        Args:
            ws: WebSocket nesnesi
        """
        self.connected = True
        logger.info("WebSocket bağlantısı açıldı.")
        
        # Kimlik doğrulama mesajı gönder
        auth_message = {
            "type": "auth",
            "api_key": self.api_key
        }
        
        ws.send(json.dumps(auth_message))
        
        # Önceki abonelikleri yeniden başlat
        for symbol in self.subscriptions:
            self.subscribe(symbol)
    
    def on_message(self, ws, message):
        """
        WebSocket'ten mesaj alındığında çağrılır.
        
        Args:
            ws: WebSocket nesnesi
            message (str): Alınan mesaj
        """
        try:
            # Mesajı ayrıştır
            data = json.loads(message)
            
            # Mesaj tipini kontrol et
            if data.get("type") == "auth_success":
                logger.info("Kimlik doğrulama başarılı.")
            
            elif data.get("type") == "price":
                # Fiyat verisi
                symbol = data.get("symbol")
                price = data.get("price")
                timestamp = data.get("timestamp", datetime.now().timestamp())
                
                # Veriyi depola
                if symbol not in realtime_data:
                    realtime_data[symbol] = []
                
                realtime_data[symbol].append({
                    "timestamp": timestamp,
                    "price": price,
                    "volume": data.get("volume", 0),
                    "bid": data.get("bid", price),
                    "ask": data.get("ask", price)
                })
                
                # Son 100 veriyi tut
                if len(realtime_data[symbol]) > 100:
                    realtime_data[symbol] = realtime_data[symbol][-100:]
                
                # Callback fonksiyonlarını çağır
                if symbol in data_callbacks:
                    for callback in data_callbacks[symbol]:
                        try:
                            callback(data)
                        except Exception as callback_error:
                            logger.error(f"Callback fonksiyonu çağrılırken hata: {str(callback_error)}")
                
                # Mesajı kuyruğa ekle
                message_queue.put(data)
            
            elif data.get("type") == "error":
                logger.error(f"WebSocket hatası: {data.get('message', 'Bilinmeyen hata')}")
        
        except Exception as e:
            logger.error(f"WebSocket mesajı işlenirken hata: {str(e)}")
            logger.error(traceback.format_exc())
    
    def on_error(self, ws, error):
        """
        WebSocket hatası oluştuğunda çağrılır.
        
        Args:
            ws: WebSocket nesnesi
            error: Hata nesnesi
        """
        logger.error(f"WebSocket hatası: {str(error)}")
        self.connected = False
    
    def on_close(self, ws, close_status_code, close_msg):
        """
        WebSocket bağlantısı kapandığında çağrılır.
        
        Args:
            ws: WebSocket nesnesi
            close_status_code: Kapatma durum kodu
            close_msg: Kapatma mesajı
        """
        self.connected = False
        logger.info(f"WebSocket bağlantısı kapandı. Kod: {close_status_code}, Mesaj: {close_msg}")
        
        # Yeniden bağlanma işlemi (eğer hala çalışıyorsa)
        if self.running:
            # Yeniden bağlanma gecikmesi ile bekle
            time.sleep(self.reconnect_delay)
            
            # Yeniden bağlanma gecikmesini artır (exponential backoff)
            self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
            
            # Yeniden bağlan
            self.connect()


# Simüle edilmiş gerçek zamanlı veri
class SimulatedWebSocketClient:
    """Gerçek WebSocket olmadığında veri simülasyonu yapan sınıf."""
    
    def __init__(self):
        """Simüle edilmiş WebSocket istemcisini başlatır."""
        self.running = False
        self.thread = None
        self.subscriptions = set()
        self.connected = False
    
    def connect(self):
        """Simüle edilmiş bağlantıyı başlatır."""
        try:
            self.running = True
            self.connected = True
            
            # Veri simülasyonu için bir iş parçacığı başlat
            self.thread = threading.Thread(target=self._data_simulation)
            self.thread.daemon = True
            self.thread.start()
            
            logger.info("Simüle edilmiş WebSocket bağlantısı başlatıldı.")
            return True
        
        except Exception as e:
            logger.error(f"Simüle edilmiş WebSocket başlatılırken hata: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def disconnect(self):
        """Simüle edilmiş bağlantıyı sonlandırır."""
        self.running = False
        self.connected = False
        logger.info("Simüle edilmiş WebSocket bağlantısı sonlandırıldı.")
    
    def subscribe(self, symbol):
        """
        Belirli bir sembol için simüle edilmiş abonelik başlatır.
        
        Args:
            symbol (str): Abone olunacak sembol
            
        Returns:
            bool: Başarılı ise True, değilse False
        """
        try:
            self.subscriptions.add(symbol)
            logger.info(f"Simüle edilmiş {symbol} sembolüne abone olundu.")
            return True
        
        except Exception as e:
            logger.error(f"Simüle edilmiş abonelik başlatılırken hata: {str(e)}")
            return False
    
    def unsubscribe(self, symbol):
        """
        Belirli bir sembol için simüle edilmiş aboneliği sonlandırır.
        
        Args:
            symbol (str): Aboneliği sonlandırılacak sembol
            
        Returns:
            bool: Başarılı ise True, değilse False
        """
        try:
            if symbol in self.subscriptions:
                self.subscriptions.remove(symbol)
            
            logger.info(f"Simüle edilmiş {symbol} sembolünün aboneliği sonlandırıldı.")
            return True
        
        except Exception as e:
            logger.error(f"Simüle edilmiş abonelik sonlandırılırken hata: {str(e)}")
            return False
    
    def _data_simulation(self):
        """Gerçek zamanlı veri simülasyonu yapar."""
        # Başlangıç fiyatları
        base_prices = {
            "GARAN.IS": 35.12,
            "THYAO.IS": 140.25,
            "AKBNK.IS": 28.65,
            "EREGL.IS": 42.78,
            "TUPRS.IS": 650.40,
            "BIMAS.IS": 185.90,
            "ASELS.IS": 72.45,
            "KCHOL.IS": 110.20,
            "TCELL.IS": 45.30,
            "SISE.IS": 33.75
        }
        
        # Başlangıç zamanı
        last_update = time.time()
        
        # Simülasyon döngüsü
        while self.running:
            try:
                # 1 saniye bekle
                time.sleep(1)
                
                current_time = time.time()
                
                # Her 1 saniyede bir veri gönder
                if current_time - last_update >= 1:
                    last_update = current_time
                    
                    # Abone olunan her sembol için veri gönder
                    for symbol in self.subscriptions:
                        if symbol in base_prices:
                            # Fiyat değişimi simülasyonu (±%1)
                            price_change = base_prices[symbol] * random.uniform(-0.01, 0.01)
                            new_price = base_prices[symbol] + price_change
                            
                            # Fiyat değişimini uygula
                            base_prices[symbol] = new_price
                            
                            # Rastgele işlem hacmi
                            volume = random.randint(100, 10000)
                            
                            # Veri oluştur
                            data = {
                                "type": "price",
                                "symbol": symbol,
                                "price": round(new_price, 2),
                                "timestamp": datetime.now().timestamp(),
                                "volume": volume,
                                "bid": round(new_price - 0.02, 2),
                                "ask": round(new_price + 0.02, 2)
                            }
                            
                            # Veriyi depola
                            if symbol not in realtime_data:
                                realtime_data[symbol] = []
                            
                            realtime_data[symbol].append({
                                "timestamp": data["timestamp"],
                                "price": data["price"],
                                "volume": data["volume"],
                                "bid": data["bid"],
                                "ask": data["ask"]
                            })
                            
                            # Son 100 veriyi tut
                            if len(realtime_data[symbol]) > 100:
                                realtime_data[symbol] = realtime_data[symbol][-100:]
                            
                            # Callback fonksiyonlarını çağır
                            if symbol in data_callbacks:
                                for callback in data_callbacks[symbol]:
                                    try:
                                        callback(data)
                                    except Exception as callback_error:
                                        logger.error(f"Callback fonksiyonu çağrılırken hata: {str(callback_error)}")
                            
                            # Mesajı kuyruğa ekle
                            message_queue.put(data)
            
            except Exception as e:
                logger.error(f"Veri simülasyonu sırasında hata: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(5)  # Hata durumunda 5 saniye bekle


# Gerçek veya simüle edilmiş WebSocket istemcisini başlat
def initialize_websocket():
    """
    WebSocket istemcisini başlatır. Gerçek WebSocket bağlantısı kurulamazsa
    simüle edilmiş bir istemci başlatır.
    
    Returns:
        WebSocketClient or SimulatedWebSocketClient: WebSocket istemcisi
    """
    if not WEBSOCKET_AVAILABLE:
        logger.warning("WebSocket kütüphanesi bulunamadı, simüle edilmiş veri kullanılacak.")
        client = SimulatedWebSocketClient()
        client.connect()
        return client
    
    # Gerçek WebSocket bağlantısını dene
    client = WebSocketClient()
    success = client.connect()
    
    # Başarısız olursa simüle edilmiş istemciyi kullan
    if not success:
        logger.warning("Gerçek WebSocket bağlantısı kurulamadı, simüle edilmiş veri kullanılacak.")
        client = SimulatedWebSocketClient()
        client.connect()
    
    return client

def register_callback(symbol, callback):
    """
    Belirli bir sembol için callback fonksiyonu kaydeder.
    
    Args:
        symbol (str): Sembol
        callback (function): Çağrılacak fonksiyon
        
    Returns:
        bool: Başarılı ise True, değilse False
    """
    try:
        if symbol not in data_callbacks:
            data_callbacks[symbol] = []
        
        if callback not in data_callbacks[symbol]:
            data_callbacks[symbol].append(callback)
        
        return True
    
    except Exception as e:
        logger.error(f"Callback fonksiyonu kaydedilirken hata: {str(e)}")
        return False

def unregister_callback(symbol, callback):
    """
    Belirli bir sembol için callback fonksiyonunu kaldırır.
    
    Args:
        symbol (str): Sembol
        callback (function): Kaldırılacak fonksiyon
        
    Returns:
        bool: Başarılı ise True, değilse False
    """
    try:
        if symbol in data_callbacks and callback in data_callbacks[symbol]:
            data_callbacks[symbol].remove(callback)
            return True
        
        return False
    
    except Exception as e:
        logger.error(f"Callback fonksiyonu kaldırılırken hata: {str(e)}")
        return False

def get_latest_price(symbol):
    """
    Belirli bir sembol için en son fiyatı döndürür.
    
    Args:
        symbol (str): Sembol
        
    Returns:
        float or None: En son fiyat veya None (veri yoksa)
    """
    try:
        if symbol in realtime_data and realtime_data[symbol]:
            return realtime_data[symbol][-1]["price"]
        
        return None
    
    except Exception as e:
        logger.error(f"En son fiyat alınırken hata: {str(e)}")
        return None

def get_realtime_data(symbol, limit=10):
    """
    Belirli bir sembol için gerçek zamanlı veriyi döndürür.
    
    Args:
        symbol (str): Sembol
        limit (int): Kaç veri noktası alınacak
        
    Returns:
        pd.DataFrame or None: Gerçek zamanlı veri veya None (veri yoksa)
    """
    try:
        if symbol in realtime_data and realtime_data[symbol]:
            data = pd.DataFrame(realtime_data[symbol][-limit:])
            
            # Zaman damgasını datetime'a dönüştür
            data["datetime"] = pd.to_datetime(data["timestamp"], unit="s")
            
            # Sütunları yeniden düzenle
            columns = ["datetime", "price", "volume", "bid", "ask"]
            return data[columns]
        
        return None
    
    except Exception as e:
        logger.error(f"Gerçek zamanlı veri alınırken hata: {str(e)}")
        return None

def process_message_queue():
    """
    Mesaj kuyruğundaki mesajları işler.
    Bu fonksiyon düzenli olarak çağrılmalıdır.
    
    Returns:
        list: İşlenen mesajlar
    """
    messages = []
    
    try:
        # Kuyrukta mesaj varsa al
        while not message_queue.empty():
            message = message_queue.get_nowait()
            messages.append(message)
            message_queue.task_done()
        
        return messages
    
    except Exception as e:
        logger.error(f"Mesaj kuyruğu işlenirken hata: {str(e)}")
        return [] 