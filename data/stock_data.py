import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import requests
import numpy as np

@st.cache_data(ttl=300) # Veriyi 5 dakika cache'le (daha sık güncelleme)
def get_stock_data_cached(symbol, period="1y", interval="1d"):
    """
    Belirtilen sembolün hisse senedi verilerini cache'leyerek alır.
    İlk olarak yfinance kullanarak gerçek veri almayı dener, başarısız olursa simüle edilmiş veri döndürür.
    """
    try:
        # Symbol formatını düzelt
        if not symbol.endswith('.IS'):
            yahoo_symbol = f"{symbol}.IS"
        else:
            yahoo_symbol = symbol
            
        stock = yf.Ticker(yahoo_symbol)
        data = stock.history(period=period, interval=interval)
        
        # Veri boş değilse, başarıyla verileri aldık demektir
        if len(data) > 0:
            # GERÇEK ZAMANLI FİYAT GÜNCELLEMESİ EKLE
            try:
                # İlk önce fast_info ile dene (daha hızlı)
                try:
                    fast_info = stock.fast_info
                    if hasattr(fast_info, 'last_price') and fast_info.last_price is not None:
                        current_market_price = fast_info.last_price
                        print(f"Anlık fiyat (fast_info) - {symbol}: {current_market_price}")
                        # En son veri noktasının Close değerini güncelle
                        data.loc[data.index[-1], 'Close'] = current_market_price
                    else:
                        raise Exception("fast_info kullanılamadı")
                except:
                    # fast_info çalışmazsa info ile dene
                    info = stock.info
                    current_price_keys = ['regularMarketPrice', 'currentPrice', 'previousClose']
                    current_market_price = None
                    
                    for key in current_price_keys:
                        if key in info and info[key] is not None and info[key] > 0:
                            current_market_price = info[key]
                            print(f"Anlık fiyat ({key}) - {symbol}: {current_market_price}")
                            break
                    
                    if current_market_price:
                        # En son veri noktasının Close değerini güncelle
                        data.loc[data.index[-1], 'Close'] = current_market_price
                        # High ve Low değerlerini de kontrol et
                        last_high = data.loc[data.index[-1], 'High']
                        last_low = data.loc[data.index[-1], 'Low']
                        
                        # Current price, high'tan büyükse high'ı güncelle
                        if current_market_price > last_high:
                            data.loc[data.index[-1], 'High'] = current_market_price
                        
                        # Current price, low'dan küçükse low'ı güncelle
                        if current_market_price < last_low:
                            data.loc[data.index[-1], 'Low'] = current_market_price
            except Exception as price_e:
                print(f"Anlık fiyat güncelleme hatası - {symbol}: {str(price_e)}")
            
            print(f"Gerçek veri alındı: {symbol}")
            return data
            
        # Veri boşsa, farklı periyotları dene
        for backup_period in ["1y", "6mo", "3mo", "1mo", "5d"]:
            if backup_period != period:
                try:
                    data = stock.history(period=backup_period, interval=interval)
                    if not data.empty:
                        print(f"Yedek periyot kullanıldı - {symbol}: {backup_period}")
                        return data
                except:
                    continue
        
        # Hiçbir periyotta veri alınamazsa simüle edilmiş veriyi döndür
        print(f"Gerçek veri alınamadı, simülasyon kullanılıyor: {symbol}")
        return get_simulated_stock_data(symbol, period, interval)
            
    except Exception as e:
        print(f"Veri alınırken hata oluştu ({symbol}): {e}")
        # Hata durumunda simüle edilmiş veriyi döndür
        return get_simulated_stock_data(symbol, period, interval)

def get_simulated_stock_data(symbol, period="1y", interval="1d"):
    """
    Test için simüle edilmiş veri üretir.
    
    Args:
        symbol (str): Hisse senedi sembolü
        period (str): Veri periyodu (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval (str): Veri aralığı (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    Returns:
        pd.DataFrame: Simüle edilmiş hisse verisi
    """
    from datetime import datetime, timedelta
    
    # Sembolden sayısal değer üret (tutarlılık için)
    seed = sum(ord(c) for c in symbol) 
    np.random.seed(seed)  
    
    # Periyodu işle
    end_date = datetime.now()
    if period == "1d":
        days = 1
    elif period == "5d":
        days = 5
    elif period == "1mo":
        days = 30
    elif period == "3mo":
        days = 90
    elif period == "6mo":
        days = 180
    elif period == "1y":
        days = 365
    elif period == "2y":
        days = 365 * 2
    elif period == "5y":
        days = 365 * 5
    elif period == "10y":
        days = 365 * 10
    elif period == "ytd":
        days = (end_date - datetime(end_date.year, 1, 1)).days
    else:  # max veya bilinmeyen format
        days = 365
    
    start_date = end_date - timedelta(days=days)
    
    # Aralığı işle ve veri noktası sayısını belirle
    is_intraday = interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']
    
    if is_intraday:
        # Gün içi veri için sadece son 7 günü kullan
        if days > 7:
            start_date = end_date - timedelta(days=7)
        
        # Yalnızca çalışma saatlerini (9:30-18:00) içeren tarih aralığı oluştur
        business_hours = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Pazartesi-Cuma
                for hour in range(9, 18):
                    for minute in range(0, 60, 5):  # 5 dakikalık aralıklar
                        if hour == 9 and minute < 30:
                            continue  # 9:30'dan önce işlem yok
                        if hour == 17 and minute > 0:
                            break  # 17:00'dan sonra işlem yok
                        business_hours.append(datetime(current_date.year, current_date.month, current_date.day, hour, minute))
            current_date += timedelta(days=1)
        
        dates = business_hours
    else:
        # Günlük veya daha büyük aralıklar için normal tarih aralığı oluştur
        if interval == '1d':
            # Her iş günü
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
        elif interval == '5d':
            # Her 5 iş günü
            dates = pd.date_range(start=start_date, end=end_date, freq='5B')
        elif interval == '1wk':
            # Her hafta
            dates = pd.date_range(start=start_date, end=end_date, freq='W')
        elif interval == '1mo':
            # Her ay
            dates = pd.date_range(start=start_date, end=end_date, freq='M')
        elif interval == '3mo':
            # Her 3 ay
            dates = pd.date_range(start=start_date, end=end_date, freq='3M')
        else:
            # Varsayılan olarak iş günleri
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Temel fiyat değerleri (hisse sembolü bazında)
    # BIST hisseleri için gerçekçi fiyat aralıkları (10-500 TL arası)
    base_price = np.random.uniform(10, 500)
    
    # Fiyat trendini belirle (yükseliş, düşüş veya yatay)
    trend = np.random.choice(['up', 'down', 'sideways'], p=[0.5, 0.3, 0.2])
    
    n = len(dates)
    if n == 0:  # Boş tarih aralığı oluşursa
        # Varsayılan olarak bir günlük veri oluştur
        dates = [end_date]
        n = 1
    
    # Trend bazlı fiyat hareketi oluştur
    if trend == 'up':
        daily_change = np.random.uniform(0.001, 0.01, n)  # Ortalama %0.1 - %1 yükseliş
    elif trend == 'down':
        daily_change = np.random.uniform(-0.01, -0.001, n)  # Ortalama %0.1 - %1 düşüş
    else:  # sideways
        daily_change = np.random.uniform(-0.005, 0.005, n)  # -%0.5 - %0.5 değişim
    
    # Rastgele dalgalanma ekle
    volatility = np.random.uniform(0.005, 0.03)  # %0.5 - %3 volatilite
    noise = np.random.normal(0, volatility, n)
    daily_returns = daily_change + noise
    
    # Kümülatif getiri hesapla
    cumulative_returns = np.cumprod(1 + daily_returns)
    # Başlangıç fiyatını uygula
    close_prices = base_price * cumulative_returns
    
    # Açılış, yüksek ve düşük fiyatları oluştur
    intraday_volatility = volatility * 0.5
    open_prices = close_prices * (1 + np.random.normal(0, intraday_volatility, n))
    high_prices = np.maximum(close_prices, open_prices) * (1 + np.abs(np.random.normal(0, intraday_volatility, n)))
    low_prices = np.minimum(close_prices, open_prices) * (1 - np.abs(np.random.normal(0, intraday_volatility, n)))
    
    # Hacim değerlerini oluştur - fiyat hareketine bağlı olarak hacim değişimi
    base_volume = np.random.randint(50000, 5000000)
    volume_change = np.abs(daily_returns) * 10  # Büyük fiyat değişimleri daha yüksek hacim
    volume = base_volume * (1 + volume_change)
    
    # DataFrame oluştur
    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume.astype(int)
    }, index=dates)
    
    return df

def get_stock_data(symbol, period="6mo"):
    """
    Hisse senedi verilerini alır
    
    Args:
        symbol (str): Hisse senedi sembolü
        period (str): Veri periyodu ("1mo", "3mo", "6mo", "1y", "2y", "5y", "max")
        
    Returns:
        pd.DataFrame: Hisse senedi verileri
    """
    try:
        # Symbol formatını düzelt
        if not symbol.endswith('.IS'):
            yahoo_symbol = f"{symbol}.IS"
        else:
            yahoo_symbol = symbol
            symbol = symbol.replace('.IS', '')
            
        # İstenen periyodun uzunluğunu değerlendir
        # Çok uzun periyotlar için veri yoksa kademeli olarak azalt
        periods_to_try = [period]
        
        # 5y istendiyse ve başarısız olursa, daha kısa periyotları dene
        if period == "5y":
            periods_to_try.extend(["3y", "2y", "1y", "6mo"])
        elif period == "3y":
            periods_to_try.extend(["2y", "1y", "6mo"])
        elif period == "2y":
            periods_to_try.extend(["1y", "6mo"])
        
        data = pd.DataFrame()
        used_period = period
        stock = yf.Ticker(yahoo_symbol)
        
        # İlk başarılı periyodu kullan
        for p in periods_to_try:
            temp_data = stock.history(period=p)
            
            if len(temp_data) >= 30:  # En az 30 gün veri olsun
                data = temp_data
                used_period = p
                break
        
        if data.empty:
            print(f"Hisse verisi alınamadı: {symbol}")
            return pd.DataFrame()
        
        # GÜNCEL FİYAT KONTROLÜ VE DÜZELTME
        try:
            # Şu anki piyasa fiyatını yf.Ticker.info'dan al
            stock_info = stock.info
            
            # Güncel fiyat bilgisini al
            current_price = None
            # Farklı alanları kontrol et (güncel fiyat için)
            if 'regularMarketPrice' in stock_info and stock_info['regularMarketPrice'] is not None:
                current_price = stock_info['regularMarketPrice']
            elif 'currentPrice' in stock_info and stock_info['currentPrice'] is not None:
                current_price = stock_info['currentPrice']
            elif 'lastPrice' in stock_info and stock_info['lastPrice'] is not None:
                current_price = stock_info['lastPrice']
            
            if current_price is not None:
                # Son günün verilerini güncelle - gerçek zamanlı fiyat için
                last_date = data.index[-1]
                if datetime.now().date() == last_date.date():
                    # Bugünkü veri varsa, Close değerini güncelle
                    data.loc[last_date, 'Close'] = current_price
                    
                    # Eğer current_price, günün en yüksek değerinden büyükse, High değerini de güncelle
                    if current_price > data.loc[last_date, 'High']:
                        data.loc[last_date, 'High'] = current_price
                        
                    # Eğer current_price, günün en düşük değerinden küçükse, Low değerini de güncelle
                    if current_price < data.loc[last_date, 'Low']:
                        data.loc[last_date, 'Low'] = current_price
                else:
                    # Bugünkü veri yoksa ve piyasa açıksa, yeni bir satır ekle
                    now = datetime.now()
                    # Türkiye saatiyle piyasa açık mı kontrol et (9:30-18:00 arası)
                    is_market_open = (
                        now.weekday() < 5 and  # Pazartesi-Cuma
                        ((now.hour == 9 and now.minute >= 30) or now.hour > 9) and  # 9:30'dan sonra
                        now.hour < 18  # 18:00'den önce
                    )
                    
                    if is_market_open:
                        # Yeni bir satır ekle
                        new_row = pd.DataFrame({
                            'Open': [current_price],
                            'High': [current_price],
                            'Low': [current_price],
                            'Close': [current_price],
                            'Volume': [0]  # Hacim bilgisi henüz yok
                        }, index=[pd.Timestamp(now)])
                        
                        # Yeni satırı veri setine ekle
                        data = pd.concat([data, new_row])
                
                print(f"Güncel fiyat alındı ({symbol}): {current_price}")
        except Exception as e:
            print(f"Güncel fiyat alınamadı ({symbol}): {str(e)}")
        
        if used_period != period:
            print(f"İstenen periyot ({period}) için veri bulunamadı. Bunun yerine {used_period} kullanıldı.")
            
        return data
        
    except Exception as e:
        print(f"Hisse verisi alınırken hata oluştu ({symbol}): {str(e)}")
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

@st.cache_data(ttl=300)  # 5 dakika cachele (daha sık güncelleme)
def get_market_summary():
    """
    Piyasa özeti verilerini alır
    """
    try:
        # BIST 100 endeks verileri
        bist100 = yf.Ticker("XU100.IS")
        bist100_data = bist100.history(period="2d")
        
        # USD/TRY verileri
        usdtry = yf.Ticker("USDTRY=X")
        usdtry_data = usdtry.history(period="2d")
        
        # Altın verileri
        gold = yf.Ticker("GC=F")
        gold_data = gold.history(period="2d")
        
        result = {}
        
        # BIST 100 verilerini işle
        if not bist100_data.empty and len(bist100_data) >= 2:
            current_price = bist100_data["Close"].iloc[-1]
            prev_close = bist100_data["Close"].iloc[-2]
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100
            volume = bist100_data["Volume"].iloc[-1] if "Volume" in bist100_data.columns else 0
            
            status = "yükseliş" if change > 0 else ("düşüş" if change < 0 else "sabit")
            
            result["bist100"] = {
                "value": current_price,
                "change": change,
                "change_percent": change_percent,
                "volume": volume / 1e9,  # Milyar TL cinsinden
                "status": status
            }
        else:
            result["bist100"] = {
                "value": 0,
                "change": 0,
                "change_percent": 0,
                "volume": 0,
                "status": "sabit"
            }
        
        # USD/TRY verilerini işle
        if not usdtry_data.empty and len(usdtry_data) >= 2:
            current_price = usdtry_data["Close"].iloc[-1]
            prev_close = usdtry_data["Close"].iloc[-2]
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100
            high = usdtry_data["High"].iloc[-1]
            low = usdtry_data["Low"].iloc[-1]
            
            status = "yükseliş" if change > 0 else ("düşüş" if change < 0 else "sabit")
            
            result["usdtry"] = {
                "value": current_price,
                "change": change,
                "change_percent": change_percent,
                "range": f"{low:.2f} - {high:.2f}",
                "status": status
            }
        else:
            result["usdtry"] = {
                "value": 0,
                "change": 0,
                "change_percent": 0,
                "range": "0.00 - 0.00",
                "status": "sabit"
            }
        
        # Altın verilerini işle
        if not gold_data.empty and len(gold_data) >= 2:
            current_price = gold_data["Close"].iloc[-1]
            prev_close = gold_data["Close"].iloc[-2]
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100
            high = gold_data["High"].iloc[-1]
            low = gold_data["Low"].iloc[-1]
            
            status = "yükseliş" if change > 0 else ("düşüş" if change < 0 else "sabit")
            
            result["gold"] = {
                "value": current_price,
                "change": change,
                "change_percent": change_percent,
                "range": f"{low:.1f} - {high:.1f}",
                "status": status
            }
        else:
            result["gold"] = {
                "value": 0,
                "change": 0,
                "change_percent": 0,
                "range": "0.0 - 0.0",
                "status": "sabit"
            }
        
        return result
        
    except Exception as e:
        # Hata durumunda varsayılan değerler döndür
        return {
            "bist100": {
                "value": 0,
                "change": 0,
                "change_percent": 0,
                "volume": 0,
                "status": "sabit"
            },
            "usdtry": {
                "value": 0,
                "change": 0,
                "change_percent": 0,
                "range": "0.00 - 0.00",
                "status": "sabit"
            },
            "gold": {
                "value": 0,
                "change": 0,
                "change_percent": 0,
                "range": "0.0 - 0.0",
                "status": "sabit"
            },
            "error": f"Piyasa özeti alınamadı: {str(e)}"
        }

@st.cache_data(ttl=300)  # 5 dakika cachele (daha sık güncelleme)
def get_popular_stocks():
    """
    Popüler hisselerin detaylı verilerini döndürür
    """
    popular_stocks_symbols = [
        "THYAO", "GARAN", "ASELS", "SISE", "AKBNK", "TCELL", "EREGL", "KCHOL",
        "VAKBN", "PETKM", "BIMAS", "TUPRS", "SAHOL", "HALKB", "ISCTR", "KOZAA",
        "PGSUS", "ARCLK", "DOHOL", "GUBRF"
    ]
    
    # Hisse isimlerini tanımla
    stock_names = {
        "THYAO": "Türk Hava Yolları",
        "GARAN": "Garanti BBVA",
        "ASELS": "Aselsan",
        "SISE": "Şişe Cam",
        "AKBNK": "Akbank",
        "TCELL": "Turkcell",
        "EREGL": "Erdemir",
        "KCHOL": "Koç Holding",
        "VAKBN": "VakıfBank",
        "PETKM": "Petkim",
        "BIMAS": "BİM",
        "TUPRS": "Tüpraş",
        "SAHOL": "Sabancı Holding",
        "HALKB": "Halkbank",
        "ISCTR": "İş Bankası",
        "KOZAA": "Koza Altın",
        "PGSUS": "Pegasus",
        "ARCLK": "Arçelik",
        "DOHOL": "Doğan Holding",
        "GUBRF": "Gübre Fabrikaları"
    }
    
    result = []
    
    for symbol in popular_stocks_symbols:
        try:
            # Hisse verilerini al
            stock_data = get_stock_data_cached(f"{symbol}.IS", period="2d")
            
            if stock_data is not None and not stock_data.empty and len(stock_data) >= 2:
                current_price = stock_data["Close"].iloc[-1]
                prev_close = stock_data["Close"].iloc[-2]
                change_percent = ((current_price - prev_close) / prev_close) * 100
                
                result.append({
                    "symbol": symbol,
                    "name": stock_names.get(symbol, symbol),
                    "value": current_price,
                    "change_percent": change_percent
                })
            else:
                # Veri alınamazsa varsayılan değerler
                result.append({
                    "symbol": symbol,
                    "name": stock_names.get(symbol, symbol),
                    "value": 0.0,
                    "change_percent": 0.0
                })
        except Exception as e:
            # Hata durumunda varsayılan değerler
            result.append({
                "symbol": symbol,
                "name": stock_names.get(symbol, symbol),
                "value": 0.0,
                "change_percent": 0.0
            })
    
    return result

@st.cache_data(ttl=600)  # 10 dakika cachele (haberler için)
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

@st.cache_data(ttl=86400)  # 24 saat cache'le
def get_all_bist_stocks():
    """
    Borsa İstanbul'daki tüm hisse senetlerinin listesini döndürür.
    
    Returns:
        list: BIST'teki tüm hisse kodlarını içeren liste (uzantısız, örn: "THYAO")
    """
    try:
        # BIST'teki tüm hisselerin statik listesi
        # Gerçek uygulamada bu liste bir API veya web scraping ile dinamik olarak alınabilir
        all_bist_stocks = [
            "ACSEL", "ADEL", "ADESE", "AEFES", "AFYON", "AGESA", "AGHOL", "AGYO", "AHGAZ", "AKBNK", 
            "AKCNS", "AKFGY", "AKFYE", "AKGRT", "AKMGY", "AKSA", "AKSEN", "AKSGY", "AKSUE", "AKYHO", 
            "ALARK", "ALBRK", "ALCAR", "ALCTL", "ALFAS", "ALGYO", "ALKA", "ALKIM", "ALMAD", "ALTIN", 
            "ALYAG", "ANACM", "ANELE", "ANGEN", "ANHYT", "ANSGR", "ARASE", "ARCEN", "ARCLK", "ARDYZ", 
            "ARENA", "ARSAN", "ARZUM", "ASELS", "ASGYO", "ASTOR", "ASUZU", "ATAGY", "ATATP", "ATEKS", 
            "ATLAS", "ATSYH", "AVGYO", "AVHOL", "AVOD", "AVTUR", "AYCES", "AYDEM", "AYEN", "AYES", 
            "AYGAZ", "AZTEK", "BAGFS", "BAKAB", "BALAT", "BANVT", "BARMA", "BASGZ", "BAYRK", "BEYAZ", 
            "BIGCH", "BIMAS", "BIOEN", "BIZIM", "BJKAS", "BLCYT", "BNTAS", "BOBET", "BOSSA", "BRISA", 
            "BRKSN", "BRMEN", "BRLSM", "BRSAN", "BRYAT", "BSOKE", "BTCIM", "BUCIM", "BURCE", "BURVA", 
            "BVSAN", "CANTE", "CASA", "CCOLA", "CELHA", "CEMAS", "CEMTS", "CEOEM", "CIMSA", "CLEBI", 
            "CONSE", "COSMO", "CRDFA", "CRFSA", "CUSAN", "DAGHL", "DAGI", "DAPGM", "DENGE", "DERHL", 
            "DERIM", "DESA", "DESPC", "DEVA", "DGATE", "DIRIT", "DITAS", "DMSAS", "DNISI", "DOAS", 
            "DOBUR", "DOCO", "DOGUB", "DOHOL", "DOKTA", "DURDO", "DYOBY", "ECILC", "ECZYT", "EDIP", 
            "EGCEY", "EGCYO", "EGEEN", "EGEPO", "EGGUB", "EGPRO", "EGSER", "EKGYO", "EKIZ", "EKSUN", 
            "ELITE", "EMKEL", "EMNIS", "ENJSA", "ENKAI", "ENSRI", "EPLAS", "ERBOS", "EREGL", "ERSU", 
            "ESCAR", "ESCOM", "ESEN", "ETILR", "ETYAT", "EUPWR", "EUREN", "EUHOL", "EUKYO", "EUYO", 
            "FADE", "FBBNK", "FENER", "FLAP", "FMIZP", "FONET", "FORMT", "FRIGO", "FROTO", "GARAN", 
            "GARFA", "GEDIK", "GEDZA", "GEREL", "GESAN", "GLBMD", "GLCVY", "GLDTR", "GLRYH", "GLYHO", 
            "GMTAS", "GOODY", "GOZDE", "GRNYO", "GRSEL", "GRTRK", "GSDDE", "GSDHO", "GSRAY", "GUBRF", 
            "GWIND", "GZNMI", "HALKB", "HATEK", "HDFGS", "HEDEF", "HEKTS", "HKTM", "HLGYO", "HTTBT", 
            "HUBVC", "HUNER", "HURGZ", "ICBCT", "IDEAS", "IDGYO", "IEYHO", "IHEVA", "IHGZT", "IHLAS", 
            "IHLGM", "IHYAY", "INDES", "INFO", "INTEM", "INVEO", "IPEKE", "ISATR", "ISBTR", "ISCTR", 
            "ISDMR", "ISFIN", "ISGSY", "ISGYO", "ISKPL", "ISYAT", "ITTFH", "IZFAS", "IZINV", "JANTS", 
            "KAPLM", "KAREL", "KARSN", "KARTN", "KARYE", "KATMR", "KAYSE", "KCAER", "KCHOL", "KENT", 
            "KERVN", "KERVT", "KFEIN", "KGYO", "KLGYO", "KLMSN", "KLNMA", "KLRHO", "KMPUR", "KNFRT", 
            "KONKA", "KONTR", "KONYA", "KORDS", "KORFZ", "KORTS", "KOSLF", "KOZAA", "KOZAL", "KPHOL", 
            "KRDMA", "KRDMB", "KRDMD", "KRGYO", "KRONT", "KRPLS", "KRSTL", "KRTEK", "KRVGD", "KSTUR", 
            "KTSKR", "KUTPO", "KUYAS", "LIDFA", "LINK", "LKMNH", "LOGO", "LUKSK", "LVENT", "MACKO", 
            "MAKIM", "MAKTK", "MANAS", "MARKA", "MARTI", "MAVI", "MEDTR", "MEGAP", "MERCN", "MERIT", 
            "MERKO", "METRO", "METUR", "MGROS", "MIATK", "MIPAZ", "MMCAS", "MNDRS", "MNDTR", "MOBTL", 
            "MPARK", "MRGYO", "MRSHL", "MSGYO", "MTRKS", "MTRYO", "MZHLD", "NATEN", "NETAS", "NETHO", 
            "NIKAY", "NTGAZ", "NTHOL", "NUGYO", "NUHCM", "ODAS", "OLMIP", "ORGE", "ORMA", "OSMEN", 
            "OYLUM", "OZGYO", "OZKGY", "OZRDN", "OZSUB", "PAGYO", "PAMEL", "PAPIL", "PARSN", "PASEU", 
            "PCILT", "PEGYO", "PEKGY", "PENGD", "PENTA", "PETKM", "PETUN", "PGSUS", "PINSU", "PKART", 
            "PKENT", "PLTUR", "PNLSN", "POLHO", "POLTK", "PRKAB", "PRKME", "PRZMA", "PSDTC", "PSGYO", 
            "QNBFB", "QNBFL", "QUAGR", "RALYH", "RAYSG", "RHYAY", "RODRG", "ROYAL", "RTALB", "RUBNS", 
            "RYGYO", "RYSAS", "SAFKR", "SAHOL", "SAMAT", "SANEL", "SANFM", "SANKO", "SARKY", "SASA", 
            "SAYAS", "SDTTR", "SEGYO", "SELEC", "SELGD", "SELVA", "SEYKM", "SISE", "SKBNK", "SKTAS", 
            "SMART", "SMRTG", "SNGYO", "SNKRN", "SNPAM", "SODSN", "SOKM", "SONME", "SRVGY", "SUMAS", 
            "SUNTK", "SUWEN", "TARIS", "TATGD", "TAVHL", "TBORG", "TCELL", "TDGYO", "TEKTU", "TERA", 
            "TETMT", "TEZOL", "TFNVK", "TGSAS", "THYAO", "TIMUR", "TKFEN", "TKNSA", "TKURU", "TLMAN", 
            "TMPOL", "TMSN", "TOASO", "TRCAS", "TRGYO", "TRILC", "TSGYO", "TSKB", "TSPOR", "TTKOM", 
            "TTRAK", "TUCLK", "TUKAS", "TUPRS", "TUREX", "TURGG", "TURSG", "TVCKS", "TVRKS", "ULUFA", 
            "ULUSE", "ULUUN", "UMPAS", "UNLU", "USAK", "UZERB", "VAKBN", "VAKKO", "VAKVK", "VANGD", 
            "VBTYZ", "VERTU", "VERUS", "VESBE", "VESTL", "VKFYO", "VKGYO", "VKING", "YAPRK", "YATAS", 
            "YAYLA", "YBTAS", "YDSYO", "YEOTK", "YESIL", "YGGYO", "YGYO", "YKBNK", "YKSLN", "YONGA", 
            "YUNSA", "YYAPI", "YYLGD", "ZOREN", "ZRGYO"
        ]
        
        return all_bist_stocks
    except Exception as e:
        st.error(f"Tüm hisse listesi alınırken hata: {str(e)}")
        return [] 