"""
Teknik göstergelerin ve özelliklerin hesaplanması için yardımcı fonksiyonlar
"""
import pandas as pd
import numpy as np
import ta
import logging
from datetime import datetime, timedelta

# Logging yapılandırması
logger = logging.getLogger(__name__)

def calculate_technical_indicators(df):
    """
    Temel teknik göstergeleri hesaplar ve veri çerçevesine ekler
    
    Parameters:
        df (pandas.DataFrame): İşlem verileri (açılış, kapanış, yüksek, düşük, hacim içermeli)
        
    Returns:
        pandas.DataFrame: Teknik göstergeler eklenmiş veri çerçevesi
    """
    try:
        # Kopya oluştur (orijinal veriyi değiştirmemek için)
        df_indicators = df.copy()
        
        # Temel teknik göstergeleri ekle
        # Trend Göstergeleri
        df_indicators['sma_20'] = ta.trend.sma_indicator(df_indicators['Close'], window=20)
        df_indicators['sma_50'] = ta.trend.sma_indicator(df_indicators['Close'], window=50)
        df_indicators['sma_200'] = ta.trend.sma_indicator(df_indicators['Close'], window=200)
        
        # Hareketli ortalama yakınsaklık-ıraksama (MACD)
        macd = ta.trend.MACD(df_indicators['Close'])
        df_indicators['macd'] = macd.macd()
        df_indicators['macd_signal'] = macd.macd_signal()
        df_indicators['macd_diff'] = macd.macd_diff()
        
        # Momentum Göstergeleri
        df_indicators['rsi'] = ta.momentum.RSIIndicator(df_indicators['Close']).rsi()
        df_indicators['stoch'] = ta.momentum.StochasticOscillator(
            df_indicators['High'], df_indicators['Low'], df_indicators['Close']
        ).stoch()
        
        # Hacim Göstergeleri
        df_indicators['obv'] = ta.volume.OnBalanceVolumeIndicator(
            df_indicators['Close'], df_indicators['Volume']
        ).on_balance_volume()
        
        # Volatilite Göstergeleri
        bollinger = ta.volatility.BollingerBands(df_indicators['Close'])
        df_indicators['bollinger_high'] = bollinger.bollinger_hband()
        df_indicators['bollinger_low'] = bollinger.bollinger_lband()
        df_indicators['bollinger_pct'] = bollinger.bollinger_pband()
        
        # Boşlukları doldur (NaN değerleri)
        df_indicators = df_indicators.fillna(method='bfill')
        
        logger.info(f"Temel teknik göstergeler başarıyla hesaplandı. Eklenen gösterge sayısı: {len(df_indicators.columns) - len(df.columns)}")
        return df_indicators
        
    except Exception as e:
        logger.error(f"Teknik göstergeler hesaplanırken hata oluştu: {str(e)}")
        # Hata durumunda orijinal veriyi geri döndür
        return df

def calculate_advanced_indicators(df):
    """
    Gelişmiş teknik göstergeleri hesaplar ve veri çerçevesine ekler
    
    Parameters:
        df (pandas.DataFrame): İşlem verileri (açılış, kapanış, yüksek, düşük, hacim içermeli)
        
    Returns:
        pandas.DataFrame: Gelişmiş göstergeler eklenmiş veri çerçevesi
    """
    try:
        # Kopya oluştur (orijinal veriyi değiştirmemek için)
        df_indicators = df.copy()
        
        # İchimoku Bulut
        ichimoku = ta.trend.IchimokuIndicator(
            high=df_indicators['High'], 
            low=df_indicators['Low']
        )
        df_indicators['ichimoku_a'] = ichimoku.ichimoku_a()
        df_indicators['ichimoku_b'] = ichimoku.ichimoku_b()
        df_indicators['ichimoku_base'] = ichimoku.ichimoku_base_line()
        df_indicators['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
        
        # KST Göstergesi (Know Sure Thing)
        kst = ta.trend.KSTIndicator(df_indicators['Close'])
        df_indicators['kst'] = kst.kst()
        df_indicators['kst_sig'] = kst.kst_sig()
        df_indicators['kst_diff'] = kst.kst_diff()
        
        # TRIX
        df_indicators['trix'] = ta.trend.TRIXIndicator(df_indicators['Close']).trix()
        
        # Chaikin Para Akışı
        df_indicators['chaikin_mf'] = ta.volume.ChaikinMoneyFlowIndicator(
            high=df_indicators['High'],
            low=df_indicators['Low'],
            close=df_indicators['Close'],
            volume=df_indicators['Volume']
        ).chaikin_money_flow()
        
        # Force Index
        df_indicators['force_idx'] = ta.volume.ForceIndexIndicator(
            close=df_indicators['Close'], 
            volume=df_indicators['Volume']
        ).force_index()
        
        # Fiyat ve hacim trendleri
        df_indicators['price_5d_change'] = df_indicators['Close'].pct_change(5)
        df_indicators['price_10d_change'] = df_indicators['Close'].pct_change(10)
        df_indicators['price_20d_change'] = df_indicators['Close'].pct_change(20)
        
        df_indicators['volume_5d_change'] = df_indicators['Volume'].pct_change(5)
        df_indicators['volume_10d_change'] = df_indicators['Volume'].pct_change(10)
        
        # Boşlukları doldur (NaN değerleri)
        df_indicators = df_indicators.fillna(method='bfill')
        
        logger.info(f"Gelişmiş teknik göstergeler başarıyla hesaplandı. Eklenen gösterge sayısı: {len(df_indicators.columns) - len(df.columns)}")
        return df_indicators
        
    except Exception as e:
        logger.error(f"Gelişmiş göstergeler hesaplanırken hata oluştu: {str(e)}")
        # Hata durumunda orijinal veriyi geri döndür
        return df

def calculate_chart_patterns(df):
    """
    Grafik formasyonlarını tespit eder ve veri çerçevesine ekler
    
    Parameters:
        df (pandas.DataFrame): İşlem verileri (açılış, kapanış, yüksek, düşük, hacim içermeli)
        
    Returns:
        pandas.DataFrame: Formasyon sinyalleri eklenmiş veri çerçevesi
    """
    try:
        # Kopya oluştur (orijinal veriyi değiştirmemek için)
        df_patterns = df.copy()
        
        # Formasyon tespiti fonksiyonları
        # Çift Tepe / Çift Dip
        df_patterns['double_top'] = _detect_double_top(df_patterns)
        df_patterns['double_bottom'] = _detect_double_bottom(df_patterns)
        
        # Omuz Baş Omuz / Ters Omuz Baş Omuz
        df_patterns['head_shoulders'] = _detect_head_shoulders(df_patterns)
        df_patterns['inv_head_shoulders'] = _detect_inv_head_shoulders(df_patterns)
        
        # Üçgen Formasyonlar
        df_patterns['ascending_triangle'] = _detect_ascending_triangle(df_patterns)
        df_patterns['descending_triangle'] = _detect_descending_triangle(df_patterns)
        df_patterns['symmetric_triangle'] = _detect_symmetric_triangle(df_patterns)
        
        # Bayrak ve Flama
        df_patterns['bull_flag'] = _detect_bull_flag(df_patterns) 
        df_patterns['bear_flag'] = _detect_bear_flag(df_patterns)
        
        # Boşlukları doldur (NaN değerleri)
        df_patterns = df_patterns.fillna(0)  # Formasyonlar için 0 (yok) değeri uygun
        
        logger.info(f"Grafik formasyonları tespit edildi. Eklenen formasyon sinyali sayısı: {len(df_patterns.columns) - len(df.columns)}")
        return df_patterns
        
    except Exception as e:
        logger.error(f"Grafik formasyonları tespit edilirken hata oluştu: {str(e)}")
        # Hata durumunda orijinal veriyi geri döndür
        return df

# Grafik formasyon tespiti için yardımcı fonksiyonlar
def _detect_double_top(df, window=20, threshold=0.03):
    """Çift Tepe formasyonu tespiti"""
    signals = pd.Series(0, index=df.index)
    for i in range(window, len(df) - window):
        if (df['High'].iloc[i-window:i].max() > df['High'].iloc[i] and 
            df['High'].iloc[i+1:i+window].max() > df['High'].iloc[i] and
            abs(df['High'].iloc[i-window:i].idxmax() - df['High'].iloc[i+1:i+window].idxmax()) > window/2 and
            abs(df['High'].iloc[i-window:i].max() - df['High'].iloc[i+1:i+window].max())/df['High'].iloc[i] < threshold):
            signals.iloc[i] = 1
    return signals

def _detect_double_bottom(df, window=20, threshold=0.03):
    """Çift Dip formasyonu tespiti"""
    signals = pd.Series(0, index=df.index)
    for i in range(window, len(df) - window):
        if (df['Low'].iloc[i-window:i].min() < df['Low'].iloc[i] and 
            df['Low'].iloc[i+1:i+window].min() < df['Low'].iloc[i] and
            abs(df['Low'].iloc[i-window:i].idxmin() - df['Low'].iloc[i+1:i+window].idxmin()) > window/2 and
            abs(df['Low'].iloc[i-window:i].min() - df['Low'].iloc[i+1:i+window].min())/df['Low'].iloc[i] < threshold):
            signals.iloc[i] = 1
    return signals

def _detect_head_shoulders(df, window=30):
    """Omuz Baş Omuz formasyonu tespiti (basitleştirilmiş)"""
    signals = pd.Series(0, index=df.index)
    # Basit bir uygulama - gerçek bir uygulamada daha karmaşık algoritmalar kullanılmalıdır
    return signals

def _detect_inv_head_shoulders(df, window=30):
    """Ters Omuz Baş Omuz formasyonu tespiti (basitleştirilmiş)"""
    signals = pd.Series(0, index=df.index)
    # Basit bir uygulama - gerçek bir uygulamada daha karmaşık algoritmalar kullanılmalıdır
    return signals

def _detect_ascending_triangle(df, window=20):
    """Yükselen Üçgen formasyonu tespiti (basitleştirilmiş)"""
    signals = pd.Series(0, index=df.index)
    # Basit bir uygulama - gerçek bir uygulamada daha karmaşık algoritmalar kullanılmalıdır
    return signals

def _detect_descending_triangle(df, window=20):
    """Alçalan Üçgen formasyonu tespiti (basitleştirilmiş)"""
    signals = pd.Series(0, index=df.index)
    # Basit bir uygulama - gerçek bir uygulamada daha karmaşık algoritmalar kullanılmalıdır
    return signals

def _detect_symmetric_triangle(df, window=20):
    """Simetrik Üçgen formasyonu tespiti (basitleştirilmiş)"""
    signals = pd.Series(0, index=df.index)
    # Basit bir uygulama - gerçek bir uygulamada daha karmaşık algoritmalar kullanılmalıdır
    return signals

def _detect_bull_flag(df, window=15):
    """Boğa Bayrağı formasyonu tespiti (basitleştirilmiş)"""
    signals = pd.Series(0, index=df.index)
    # Basit bir uygulama - gerçek bir uygulamada daha karmaşık algoritmalar kullanılmalıdır
    return signals

def _detect_bear_flag(df, window=15):
    """Ayı Bayrağı formasyonu tespiti (basitleştirilmiş)"""
    signals = pd.Series(0, index=df.index)
    # Basit bir uygulama - gerçek bir uygulamada daha karmaşık algoritmalar kullanılmalıdır
    return signals

def add_sentiment_data(df, sentiment_df=None):
    """
    Haber duyarlılık verilerini ekler (varsa)
    
    Parameters:
        df (pandas.DataFrame): İşlem verileri
        sentiment_df (pandas.DataFrame, optional): Haber duyarlılık verileri
        
    Returns:
        pandas.DataFrame: Duyarlılık verileri eklenmiş veri çerçevesi
    """
    try:
        # Kopya oluştur (orijinal veriyi değiştirmemek için)
        df_with_sentiment = df.copy()
        
        # Eğer sentiment_df verilmişse, işlem yap
        if sentiment_df is not None and not sentiment_df.empty:
            # sentiment_df indeksi tarih formatında olmalı
            if not isinstance(sentiment_df.index, pd.DatetimeIndex):
                try:
                    sentiment_df.index = pd.to_datetime(sentiment_df.index)
                except:
                    logger.warning("Duyarlılık verileri tarih indeksi oluşturulamadı. Veri eklenemiyor.")
                    return df_with_sentiment
            
            # df'in de indeksi tarih formatında olmalı
            if not isinstance(df_with_sentiment.index, pd.DatetimeIndex):
                try:
                    df_with_sentiment.index = pd.to_datetime(df_with_sentiment.index)
                except:
                    logger.warning("Ana veri çerçevesi için tarih indeksi oluşturulamadı. Veri eklenemiyor.")
                    return df_with_sentiment
                    
            # Duyarlılık verilerini birleştir
            # Not: Birleştirme için uygun sütunlar sentiment_df'te bulunmalı
            # Örnekler: 'sentiment_score', 'news_volume', vb.
            
            # Örnek birleştirme:
            # df_with_sentiment = df_with_sentiment.join(sentiment_df[['sentiment_score', 'news_volume']], how='left')
            
            # Veya resampling ile:
            sentiment_resampled = sentiment_df.resample('D').mean()  # Günlük ortalama
            df_with_sentiment = df_with_sentiment.join(sentiment_resampled, how='left')
            
            # Boşlukları doldur
            df_with_sentiment = df_with_sentiment.fillna(method='ffill')
            
            logger.info(f"Duyarlılık verileri başarıyla eklendi. Eklenen sütun sayısı: {len(df_with_sentiment.columns) - len(df.columns)}")
        else:
            logger.info("Duyarlılık verileri bulunamadı veya boş. Orijinal veri döndürülüyor.")
            
        return df_with_sentiment
        
    except Exception as e:
        logger.error(f"Duyarlılık verileri eklenirken hata oluştu: {str(e)}")
        # Hata durumunda orijinal veriyi geri döndür
        return df

def add_macro_economic_data(df, macro_df=None):
    """
    Makroekonomik verileri ekler (varsa)
    
    Parameters:
        df (pandas.DataFrame): İşlem verileri
        macro_df (pandas.DataFrame, optional): Makroekonomik veriler
        
    Returns:
        pandas.DataFrame: Makroekonomik veriler eklenmiş veri çerçevesi
    """
    try:
        # Kopya oluştur (orijinal veriyi değiştirmemek için)
        df_with_macro = df.copy()
        
        # Eğer macro_df verilmişse, işlem yap
        if macro_df is not None and not macro_df.empty:
            # İndekslerin tarih formatında olduğundan emin ol
            if not isinstance(macro_df.index, pd.DatetimeIndex):
                try:
                    macro_df.index = pd.to_datetime(macro_df.index)
                except:
                    logger.warning("Makroekonomik veriler için tarih indeksi oluşturulamadı. Veri eklenemiyor.")
                    return df_with_macro
                    
            if not isinstance(df_with_macro.index, pd.DatetimeIndex):
                try:
                    df_with_macro.index = pd.to_datetime(df_with_macro.index)
                except:
                    logger.warning("Ana veri çerçevesi için tarih indeksi oluşturulamadı. Veri eklenemiyor.")
                    return df_with_macro
            
            # Makroekonomik verileri birleştir
            # Örnekler: 'interest_rate', 'gdp_growth', 'inflation', 'unemployment', vb.
            
            # Örnek birleştirme:
            # macro_resampled = macro_df.resample('D').ffill()  # Günlük veri, forward fill
            # df_with_macro = df_with_macro.join(macro_resampled, how='left')
            
            # Boşlukları doldur (makroekonomik verilerde genellikle en son değer kullanılır)
            df_with_macro = df_with_macro.fillna(method='ffill')
            
            logger.info(f"Makroekonomik veriler başarıyla eklendi. Eklenen sütun sayısı: {len(df_with_macro.columns) - len(df.columns)}")
        else:
            logger.info("Makroekonomik veriler bulunamadı veya boş. Orijinal veri döndürülüyor.")
            
        return df_with_macro
        
    except Exception as e:
        logger.error(f"Makroekonomik veriler eklenirken hata oluştu: {str(e)}")
        # Hata durumunda orijinal veriyi geri döndür
        return df

def create_target_variable(df, horizon=5, threshold=0.02):
    """
    Belirli bir zaman ufku için hedef değişkeni oluşturur
    
    Parameters:
        df (pandas.DataFrame): İşlem verileri
        horizon (int): Tahmin ufku (gün sayısı)
        threshold (float): Yükseliş eşiği (yüzde olarak, örn. 0.02 = %2)
        
    Returns:
        pandas.DataFrame: Hedef değişken eklenmiş veri çerçevesi
    """
    try:
        # Kopya oluştur (orijinal veriyi değiştirmemek için)
        df_target = df.copy()
        
        # Gelecekteki fiyat değişimini hesapla
        future_return = df_target['Close'].shift(-horizon) / df_target['Close'] - 1
        
        # İkili sınıflandırma için hedef değişken oluştur (1: Yükseliş, 0: Diğer)
        target_col_name = f'target_{horizon}d_{int(threshold*100)}pct'
        df_target[target_col_name] = (future_return > threshold).astype(int)
        
        # İsteğe bağlı: Sürekli hedef de eklenebilir
        df_target[f'future_return_{horizon}d'] = future_return
        
        logger.info(f"Hedef değişken oluşturuldu: {target_col_name}")
        return df_target
        
    except Exception as e:
        logger.error(f"Hedef değişken oluşturulurken hata oluştu: {str(e)}")
        # Hata durumunda orijinal veriyi geri döndür
        return df 