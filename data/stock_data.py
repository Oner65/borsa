import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import requests
import numpy as np

@st.cache_data(ttl=3600) # Veriyi 1 saat cache'le
def get_stock_data_cached(symbol, period="1y", interval="1d"):
    """
    Belirtilen sembolün hisse senedi verilerini cache'leyerek alır
    """
    try:
        if not symbol.endswith('.IS'):
            symbol = f"{symbol}.IS"
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        if len(data) > 0:
            return data
    except Exception as e:
        st.error(f"Veri alınırken hata oluştu: {e}")
    return pd.DataFrame()

def get_stock_data(symbol, period="6mo"):
    """
    Hisse senedi verilerini alır
    """
    try:
        if not symbol.endswith('.IS'):
            symbol = f"{symbol}.IS"
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        # Veri kontrolü yap
        if len(data) > 0:
            # Gerekli sütunların varlığını kontrol et
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in data.columns:
                    st.error(f"{symbol} verisinde {col} sütunu eksik.")
                    return pd.DataFrame()
                    
            # Boş değer kontrolü
            if data['Close'].isnull().any():
                data = data.fillna(method='ffill')  # İleri doldurma ile boş değerleri doldur
                
            return data
        else:
            st.error(f"{symbol} için veri alınamadı. Sembol doğru olmayabilir.")
    except Exception as e:
        st.error(f"Veri alınırken hata oluştu: {e}")
    
    return pd.DataFrame() 

@st.cache_data(ttl=86400)  # 24 saat cache'le
def get_company_info(symbol):
    """
    Belirtilen sembol için şirket bilgilerini döndürür
    
    Args:
        symbol (str): Hisse senedi sembolü (BIST)
    
    Returns:
        dict: Şirket bilgilerini içeren sözlük 
        {
            'name': Şirket adı,
            'sector': Sektör,
            'industry': Endüstri,
            'website': Web sitesi,
            'description': Açıklama,
            ...
        }
    """
    try:
        if symbol and not symbol.endswith('.IS'):
            yahoo_symbol = f"{symbol}.IS"
        else:
            yahoo_symbol = symbol
            
        # Yahoo Finance'den şirket bilgilerini al
        stock = yf.Ticker(yahoo_symbol)
        info = stock.info
        
        # Şirket bilgilerini döndür
        if info and 'longName' in info:
            return {
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'website': info.get('website', ''),
                'description': info.get('longBusinessSummary', ''),
                'country': info.get('country', 'Türkiye'),
                'exchange': info.get('exchange', 'BIST'),
                'currency': info.get('currency', 'TRY'),
                'symbol': symbol
            }
    except Exception as e:
        print(f"Şirket bilgileri alınamadı ({symbol}): {e}")
        
    # Hata durumunda boş sözlük döndür
    return {} 

@st.cache_data(ttl=300)  # 5 dakika cachele
def get_market_summary():
    """
    Piyasa özeti verilerini çeker:
    - BIST100 Endeksi
    - USD/TRY Kuru
    - ALTIN/ONS Fiyatı
    
    Returns:
        dict: Piyasa özet verileri
    """
    market_data = {
        "bist100": {"symbol": "XU100.IS", "name": "BIST100", "value": 0, "change": 0, "change_percent": 0, "volume": 0, "status": "nötr"},
        "usdtry": {"symbol": "USDTRY=X", "name": "USD/TRY", "value": 0, "change": 0, "change_percent": 0, "range": "0-0", "status": "nötr"},
        "gold": {"symbol": "GC=F", "name": "ALTIN/ONS", "value": 0, "change": 0, "change_percent": 0, "range": "0-0", "status": "nötr"}
    }
    
    try:
        # BIST100 verilerini al
        bist100 = yf.Ticker("XU100.IS")
        bist100_info = bist100.history(period="2d")
        
        if not bist100_info.empty:
            # Bugün ve dün kapaış fiyatları
            current_close = bist100_info['Close'].iloc[-1]
            prev_close = bist100_info['Close'].iloc[-2] if len(bist100_info) > 1 else current_close
            
            change = current_close - prev_close
            change_percent = (change / prev_close) * 100 if prev_close > 0 else 0
            
            # Durum belirleme (yükseliş, düşüş veya nötr)
            status = "yükseliş" if change > 0 else ("düşüş" if change < 0 else "nötr")
            
            # Hacim hesaplama (Milyon TL)
            volume = bist100_info['Volume'].iloc[-1] / 1_000_000 if 'Volume' in bist100_info else 0
            
            market_data["bist100"].update({
                "value": round(current_close, 2),
                "change": round(change, 2),
                "change_percent": round(change_percent, 2),
                "volume": round(volume, 1),
                "status": status
            })
    except Exception as e:
        print(f"BIST100 verisi alınırken hata: {e}")
    
    try:
        # USD/TRY verilerini al
        usdtry = yf.Ticker("USDTRY=X")
        usdtry_info = usdtry.history(period="2d")
        
        if not usdtry_info.empty:
            current_close = usdtry_info['Close'].iloc[-1]
            prev_close = usdtry_info['Close'].iloc[-2] if len(usdtry_info) > 1 else current_close
            
            change = current_close - prev_close
            change_percent = (change / prev_close) * 100 if prev_close > 0 else 0
            
            # Durum belirleme
            status = "yükseliş" if change > 0 else ("düşüş" if change < 0 else "nötr")
            
            # 24 saat aralığı hesaplama
            day_high = usdtry_info['High'].max()
            day_low = usdtry_info['Low'].min()
            range_str = f"{day_low:.2f}-{day_high:.2f}"
            
            market_data["usdtry"].update({
                "value": round(current_close, 2),
                "change": round(change, 2),
                "change_percent": round(change_percent, 2),
                "range": range_str,
                "status": status
            })
    except Exception as e:
        print(f"USD/TRY verisi alınırken hata: {e}")
    
    try:
        # Altın/Ons verilerini al
        gold = yf.Ticker("GC=F")
        gold_info = gold.history(period="2d")
        
        if not gold_info.empty:
            current_close = gold_info['Close'].iloc[-1]
            prev_close = gold_info['Close'].iloc[-2] if len(gold_info) > 1 else current_close
            
            change = current_close - prev_close
            change_percent = (change / prev_close) * 100 if prev_close > 0 else 0
            
            # Durum belirleme
            status = "yükseliş" if change > 0 else ("düşüş" if change < 0 else "nötr")
            
            # 24 saat aralığı hesaplama
            day_high = gold_info['High'].max()
            day_low = gold_info['Low'].min()
            range_str = f"{int(day_low)}-{int(day_high)}"
            
            market_data["gold"].update({
                "value": round(current_close, 1),
                "change": round(change, 1),
                "change_percent": round(change_percent, 2),
                "range": range_str,
                "status": status
            })
    except Exception as e:
        print(f"Altın/Ons verisi alınırken hata: {e}")
    
    return market_data

@st.cache_data(ttl=600)  # 10 dakika cachele
def get_popular_stocks():
    """
    BIST'teki popüler hisselerin son verilerini getirir
    
    Returns:
        list: Popüler hisselerin güncel verileri
    """
    popular_symbols = ["THYAO", "ASELS", "GARAN", "SASA", "KCHOL"]
    popular_data = []
    
    for symbol in popular_symbols:
        try:
            stock_symbol = f"{symbol}.IS"
            stock = yf.Ticker(stock_symbol)
            stock_info = stock.history(period="2d")
            
            if not stock_info.empty:
                company_info = get_company_info(symbol)
                company_name = company_info.get('name', symbol)
                
                current_close = stock_info['Close'].iloc[-1]
                prev_close = stock_info['Close'].iloc[-2] if len(stock_info) > 1 else current_close
                
                change = current_close - prev_close
                change_percent = (change / prev_close) * 100 if prev_close > 0 else 0
                
                # Durum belirleme
                status = "yükseliş" if change > 0 else ("düşüş" if change < 0 else "nötr")
                
                popular_data.append({
                    "symbol": symbol,
                    "name": company_name,
                    "value": round(current_close, 2),
                    "change": round(change, 2),
                    "change_percent": round(change_percent, 2),
                    "status": status
                })
        except Exception as e:
            print(f"{symbol} verisi alınırken hata: {e}")
    
    return popular_data 

@st.cache_data(ttl=3600)  # 1 saat cachele
def get_stock_news(symbol, limit=5):
    """
    Belirtilen hisse senedi sembolü için haberleri getirir.
    
    Args:
        symbol (str): Hisse senedi sembolü (örn: THYAO)
        limit (int): Gösterilecek maksimum haber sayısı
        
    Returns:
        list: Haberler listesi
        [
            {
                "title": Haber başlığı,
                "publisher": Yayıncı,
                "link": Haber linki, 
                "published": Yayın tarihi
            },
            ...
        ]
    """
    try:
        # Sembol formatını kontrol et
        if not symbol.endswith('.IS'):
            yahoo_symbol = f"{symbol}.IS"
        else:
            yahoo_symbol = symbol
            
        # Yahoo Finance'den haberleri al
        stock = yf.Ticker(yahoo_symbol)
        news = stock.news
        
        # Sonuçları formatlama
        formatted_news = []
        if news:
            for item in news[:limit]:
                formatted_news.append({
                    "title": item.get("title", ""),
                    "publisher": item.get("publisher", ""),
                    "link": item.get("link", ""),
                    "published": datetime.fromtimestamp(item.get("providerPublishTime", 0)).strftime("%d.%m.%Y %H:%M")
                })
        
        return formatted_news
    except Exception as e:
        st.error(f"Haberler alınırken hata oluştu: {e}")
        return [] 

@st.cache_data(ttl=86400)  # 24 saat cache'le
def get_stock_list(index_name="BIST 100"):
    """
    Belirtilen endekse ait hisse listesini döndürür
    
    Args:
        index_name (str): Endeks adı ('BIST 30', 'BIST 50', 'BIST 100')
    
    Returns:
        list: Endeksteki hisse kodlarını içeren liste (.IS uzantılı)
    """
    try:
        bist30_stocks = [
            "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "EKGYO.IS", 
            "EREGL.IS", "FROTO.IS", "GARAN.IS", "HALKB.IS", "HEKTS.IS", 
            "ISCTR.IS", "KCHOL.IS", "KOZAA.IS", "KOZAL.IS", "KRDMD.IS", 
            "PETKM.IS", "PGSUS.IS", "SAHOL.IS", "SASA.IS", "SISE.IS", 
            "TCELL.IS", "THYAO.IS", "TOASO.IS", "TUPRS.IS", "VAKBN.IS", 
            "VESTL.IS", "YKBNK.IS", "TAVHL.IS", "ENJSA.IS", "SOKM.IS"
        ]
        
        bist50_stocks = bist30_stocks + [
            "ALARK.IS", "ALBRK.IS", "CIMSA.IS", "DOHOL.IS", "ESEN.IS", 
            "GESAN.IS", "IPEKE.IS", "ISGYO.IS", "KLMSN.IS", "KONTR.IS", 
            "ODAS.IS", "OYAKC.IS", "PRKME.IS", "SMRTG.IS", "TSKB.IS", 
            "TRGYO.IS", "TTKOM.IS", "ULKER.IS", "YBTAS.IS", "ZOREN.IS"
        ]
        
        bist100_stocks = bist50_stocks + [
            "AEFES.IS", "AGHOL.IS", "AKFYE.IS", "AKSA.IS", "AKSEN.IS", 
            "ALGYO.IS", "ASUZU.IS", "AYDEM.IS", "BAGFS.IS", "BASGZ.IS", 
            "BRSAN.IS", "BRISA.IS", "CCOLA.IS", "CEMTS.IS", "ENKAI.IS", 
            "ERBOS.IS", "EUPWR.IS", "GLYHO.IS", "GUBRF.IS", "HSVGY.IS", 
            "INDES.IS", "ISDMR.IS", "ISGSY.IS", "ISMEN.IS", "KARTN.IS", 
            "KERVT.IS", "LOGO.IS", "MGROS.IS", "MPARK.IS", "NETAS.IS", 
            "NUHCM.IS", "OTKAR.IS", "OYAKT.IS", "SELEC.IS", "SKBNK.IS",
            "TATGD.IS", "TKFEN.IS", "TUKAS.IS", "VESBE.IS", "YATAS.IS",
            "ZRGYO.IS", "AGESA.IS", "AKSA.IS", "BRYAT.IS", "DOAS.IS", 
            "GWIND.IS", "KMPUR.IS", "MAVI.IS", "SDTTR.IS", "SMRTG.IS"
        ]
        
        if index_name == "BIST 30":
            return bist30_stocks
        elif index_name == "BIST 50":
            return bist50_stocks
        elif index_name == "BIST 100":
            return bist100_stocks
        else:
            st.warning(f"Bilinmeyen endeks: {index_name}. BIST 100 döndürülüyor.")
            return bist100_stocks
            
    except Exception as e:
        st.error(f"Hisse listesi alınırken hata: {str(e)}")
        return [] 