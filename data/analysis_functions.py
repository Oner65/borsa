import pandas as pd
import numpy as np

def calculate_indicators(df):
    """
    Hisse senedi verisi için teknik göstergeleri hesaplar.
    
    Parametreler:
        df (pandas.DataFrame): Hisse senedi verisi (Open, High, Low, Close, Volume sütunları içermeli)
    
    Döndürür:
        pandas.DataFrame: Teknik göstergeler eklenmiş veri çerçevesi
    """
    # Veri çerçevesinin bir kopyasını oluştur
    df = df.copy()
    
    # SMA (Simple Moving Average) - Basit Hareketli Ortalama
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    
    # EMA (Exponential Moving Average) - Üssel Hareketli Ortalama
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # RSI (Relative Strength Index) - Göreceli Güç Endeksi
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    # Sonraki günler için RSI hesabı
    for i in range(14, len(df)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * 13 + gain.iloc[i]) / 14
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * 13 + loss.iloc[i]) / 14
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bantları
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    
    # Stokastik Osilatör
    window = 14
    df['Stoch_K'] = 100 * ((df['Close'] - df['Low'].rolling(window).min()) / 
                           (df['High'].rolling(window).max() - df['Low'].rolling(window).min()))
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
    
    # ADX (Average Directional Index) - Ortalama Yön Endeksi
    # True Range
    df['TR'] = np.maximum(
        np.maximum(
            df['High'] - df['Low'],
            np.abs(df['High'] - df['Close'].shift(1))
        ),
        np.abs(df['Low'] - df['Close'].shift(1))
    )
    
    # Directional Movement
    df['DM_plus'] = np.where(
        (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
        np.maximum(df['High'] - df['High'].shift(1), 0),
        0
    )
    
    df['DM_minus'] = np.where(
        (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
        np.maximum(df['Low'].shift(1) - df['Low'], 0),
        0
    )
    
    # Smoothed True Range and Directional Movement
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df['Smoothed_DM_plus'] = df['DM_plus'].rolling(window=14).mean()
    df['Smoothed_DM_minus'] = df['DM_minus'].rolling(window=14).mean()
    
    # Directional Indicators
    df['DI_plus'] = 100 * (df['Smoothed_DM_plus'] / df['ATR'])
    df['DI_minus'] = 100 * (df['Smoothed_DM_minus'] / df['ATR'])
    
    # Directional Index
    df['DX'] = 100 * np.abs(df['DI_plus'] - df['DI_minus']) / (df['DI_plus'] + df['DI_minus'])
    
    # Average Directional Index
    df['ADX'] = df['DX'].rolling(window=14).mean()
    
    # Volatilite (%)
    df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * 100
    
    # İşlem hacmi hareketli ortalaması
    df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()
    
    # Günlük değişim (%)
    df['Daily_Change'] = df['Close'].pct_change() * 100
    
    # NaN değerleri temizle
    df = df.fillna(0)
    
    return df

def get_stock_prediction(symbol, stock_data):
    """
    Bir hisse senedi için fiyat tahmini yapar.
    
    Args:
        symbol (str): Hisse senedi sembolü
        stock_data (pandas.DataFrame): Hisse senedi geçmiş verisi
        
    Returns:
        dict: Tahmin sonucu veya None (hata durumunda)
    """
    try:
        if stock_data.empty:
            return None
        
        # Teknik göstergeleri hesapla
        df = calculate_indicators(stock_data)
        
        # Son satırdaki verileri al
        last_row = df.iloc[-1]
        
        # Basit tahmin modeli - teknik göstergelere dayalı
        trend_signals = 0
        confidence = 0.5  # Başlangıç güven değeri
        
        # RSI sinyali
        if last_row['RSI'] > 70:
            trend_signals -= 1  # Aşırı alım bölgesi (düşüş sinyali)
            confidence += 0.1
        elif last_row['RSI'] < 30:
            trend_signals += 1  # Aşırı satım bölgesi (yükseliş sinyali)
            confidence += 0.1
        
        # MACD sinyali
        if last_row['MACD'] > last_row['MACD_Signal']:
            trend_signals += 1  # MACD, sinyal çizgisinin üstünde (yükseliş sinyali)
            confidence += 0.05
        else:
            trend_signals -= 1  # MACD, sinyal çizgisinin altında (düşüş sinyali)
            confidence += 0.05
        
        # SMA sinyalleri
        if last_row['SMA20'] > last_row['SMA50']:
            trend_signals += 1  # Kısa vadeli ortalama, uzun vadeli ortalamayı aştı (yükseliş sinyali)
            confidence += 0.1
        else:
            trend_signals -= 1
            confidence += 0.05
        
        # Bollinger Bant sinyali
        if last_row['Close'] < last_row['BB_Lower']:
            trend_signals += 1  # Fiyat alt bandın altında (olası yükseliş)
            confidence += 0.1
        elif last_row['Close'] > last_row['BB_Upper']:
            trend_signals -= 1  # Fiyat üst bandın üstünde (olası düşüş)
            confidence += 0.1
        
        # Stokastik Osilatör sinyali
        if last_row['Stoch_K'] < 20 and last_row['Stoch_K'] > last_row['Stoch_D']:
            trend_signals += 1  # Aşırı satış bölgesinden çıkış (yükseliş sinyali)
            confidence += 0.1
        elif last_row['Stoch_K'] > 80 and last_row['Stoch_K'] < last_row['Stoch_D']:
            trend_signals -= 1  # Aşırı alım bölgesinden çıkış (düşüş sinyali)
            confidence += 0.1
        
        # ADX sinyali - trend gücü
        if last_row['ADX'] > 25:
            confidence += 0.1  # Güçlü trend, tahmin güvenini artır
        
        # Volatilite etkisi
        if last_row['Volatility'] > 5:  # %5'ten fazla volatilite
            confidence -= 0.1  # Yüksek volatilite, tahmin güvenini azalt
        
        # Tahmin sonucunu belirle
        if trend_signals > 0:
            prediction_result = "YÜKSELIŞ"
            prediction_percentage = min(trend_signals * 2, 10)  # Maksimum %10 yükseliş tahmini
        elif trend_signals < 0:
            prediction_result = "DÜŞÜŞ"
            prediction_percentage = max(trend_signals * 2, -10)  # Maksimum %10 düşüş tahmini
        else:
            prediction_result = "YATAY"
            prediction_percentage = 0
        
        # Güven skorunu 0.3 ile 0.9 arasında sınırla
        confidence = max(0.3, min(0.9, confidence))
        
        # Tahmin sonucunu döndür
        return {
            "symbol": symbol,
            "prediction_result": prediction_result,
            "prediction_percentage": prediction_percentage,
            "confidence_score": confidence,
            "model_type": "Teknik Analiz Modeli",
            "features_used": ["RSI", "MACD", "SMA", "Bollinger", "Stochastic", "ADX"]
        }
    except Exception as e:
        print(f"Tahmin hatası ({symbol}): {str(e)}")
        return None 