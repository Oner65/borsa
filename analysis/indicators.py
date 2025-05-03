import pandas as pd
import numpy as np
import streamlit as st

def calculate_rsi(data, window=14):
    """RSI (Relatif Güç İndeksi) hesaplar"""
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=window-1, adjust=False).mean()
    ema_down = down.ewm(com=window-1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

def calculate_sma(data, window):
    """Basit Hareketli Ortalama hesaplar"""
    return data.rolling(window=window).mean()

def calculate_ema(data, window):
    """Üssel Hareketli Ortalama hesaplar"""
    return data.ewm(span=window, adjust=False).mean()

def calculate_macd(data):
    """MACD (Hareketli Ortalama Yakınsama/Iraksama) hesaplar"""
    exp1 = data.ewm(span=12, adjust=False).mean()
    exp2 = data.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Bollinger Bantları hesaplar"""
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

def calculate_stochastic(df, k_period=14, d_period=3):
    """Stokastik Osilatör hesaplar"""
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_period).mean()
    return k, d

def calculate_adx(df, window=14):
    """ADX (Ortalama Yön İndeksi) hesaplar"""
    df = df.copy()
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['DMplus'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), 
                            df['High'] - df['High'].shift(1), 0)
    df['DMminus'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), 
                             df['Low'].shift(1) - df['Low'], 0)
    
    df['TRn'] = df['TR'].rolling(window).mean()
    df['DMplusN'] = df['DMplus'].rolling(window).mean()
    df['DMminusN'] = df['DMminus'].rolling(window).mean()
    
    df['DIplus'] = 100 * (df['DMplusN'] / df['TRn'])
    df['DIminus'] = 100 * (df['DMminusN'] / df['TRn'])
    
    df['DIdiff'] = abs(df['DIplus'] - df['DIminus'])
    df['DIsum'] = df['DIplus'] + df['DIminus']
    
    df['DX'] = 100 * (df['DIdiff'] / df['DIsum'])
    
    adx = df['DX'].rolling(window).mean()
    return adx, df['DIplus'], df['DIminus']

def calculate_williams_r(df, window=14):
    """Williams %R hesaplar"""
    highest_high = df['High'].rolling(window=window).max()
    lowest_low = df['Low'].rolling(window=window).min()
    wr = -100 * ((highest_high - df['Close']) / (highest_high - lowest_low))
    return wr

def calculate_cci(df, window=14):
    """CCI (Emtia Kanal İndeksi) hesaplar"""
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma = tp.rolling(window=window).mean()
    md = (tp - ma).abs().rolling(window=window).mean()
    cci = (tp - ma) / (0.015 * md)
    return cci

def calculate_atr(df, window=14):
    """ATR (Ortalama Gerçek Aralık) hesaplar"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    atr = true_range.rolling(window=window).mean()
    return atr

def calculate_highs_lows(df, window=14):
    """Yüksek ve Düşük fiyat sinyallerini hesaplar"""
    df['highest_high'] = df['High'].rolling(window=window).max()
    df['lowest_low'] = df['Low'].rolling(window=window).min()
    return df['highest_high'], df['lowest_low']

def calculate_ultimate_oscillator(df, short=7, medium=14, long=28):
    """Ultimate Oscillator hesaplar"""
    df = df.copy()
    df['prior_close'] = df['Close'].shift(1)
    df['true_high'] = df[['High', 'prior_close']].max(axis=1)
    df['true_low'] = df[['Low', 'prior_close']].min(axis=1)
    df['buying_pressure'] = df['Close'] - df['true_low']
    df['true_range'] = df['true_high'] - df['true_low']
    
    df['avg7'] = df['buying_pressure'].rolling(window=short).sum() / df['true_range'].rolling(window=short).sum()
    df['avg14'] = df['buying_pressure'].rolling(window=medium).sum() / df['true_range'].rolling(window=medium).sum()
    df['avg28'] = df['buying_pressure'].rolling(window=long).sum() / df['true_range'].rolling(window=long).sum()
    
    # (4 * avg7 + 2 * avg14 + avg28) / (4 + 2 + 1)
    uo = 100 * (4 * df['avg7'] + 2 * df['avg14'] + df['avg28']) / 7
    return uo

def calculate_roc(df, window=14):
    """ROC (Değişim Oranı) hesaplar"""
    if isinstance(df, pd.Series):
        roc = ((df - df.shift(window)) / df.shift(window)) * 100
    else:
        roc = ((df['Close'] - df['Close'].shift(window)) / df['Close'].shift(window)) * 100
    return roc

def calculate_bull_bear_power(df, window=13):
    """Bull/Bear Power göstergesini hesaplar"""
    ema = df['Close'].ewm(span=window, adjust=False).mean()
    bull_power = df['High'] - ema
    bear_power = df['Low'] - ema
    return bull_power, bear_power

def calculate_stoch_rsi(df, window=14, smooth_k=3, smooth_d=3):
    """Stochastic RSI hesaplar"""
    rsi = calculate_rsi(df['Close'], window)
    stoch_rsi = (rsi - rsi.rolling(window).min()) / (rsi.rolling(window).max() - rsi.rolling(window).min())
    k = stoch_rsi.rolling(window=smooth_k).mean()
    d = k.rolling(window=smooth_d).mean()
    return k, d

def calculate_indicators(df):
    """Tüm teknik göstergeleri hesaplar"""
    df = df.copy()
    
    # Hareketli Ortalamalar
    df['SMA5'] = calculate_sma(df['Close'], 5)
    df['SMA10'] = calculate_sma(df['Close'], 10)
    df['SMA20'] = calculate_sma(df['Close'], 20)
    df['SMA50'] = calculate_sma(df['Close'], 50)
    df['SMA100'] = calculate_sma(df['Close'], 100)
    df['SMA200'] = calculate_sma(df['Close'], 200)
    
    df['EMA5'] = calculate_ema(df['Close'], 5)
    df['EMA10'] = calculate_ema(df['Close'], 10)
    df['EMA20'] = calculate_ema(df['Close'], 20)
    df['EMA50'] = calculate_ema(df['Close'], 50)
    df['EMA100'] = calculate_ema(df['Close'], 100)
    df['EMA200'] = calculate_ema(df['Close'], 200)
    
    # Osilatörler
    df['RSI'] = calculate_rsi(df['Close'])
    macd, signal, hist = calculate_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = hist
    
    # Bollinger Bantları
    df['Upper_Band'], df['Middle_Band'], df['Lower_Band'] = calculate_bollinger_bands(df['Close'])
    
    # Momentum Göstergeleri
    df['ROC'] = calculate_roc(df['Close'])
    df['Williams_%R'] = calculate_williams_r(df)
    k, d = calculate_stochastic(df)
    df['Stoch_%K'] = k
    df['Stoch_%D'] = d
    
    # Trend Göstergeleri
    adx, plus_di, minus_di = calculate_adx(df)
    df['ADX'] = adx
    df['Plus_DI'] = plus_di
    df['Minus_DI'] = minus_di
    
    # Volatilite Göstergeleri
    df['ATR'] = calculate_atr(df)
    
    # Hacim Göstergeleri
    if 'Volume' in df.columns:
        df['Volume_SMA20'] = calculate_sma(df['Volume'], 20)
    
    return df

def get_signals(df):
    """Teknik analiz sinyallerini hesaplar"""
    signals = pd.DataFrame(index=df.index)
    
    # Temel Sinyaller
    signals['Price'] = df['Close']
    signals['Price_Change'] = df['Close'].pct_change() * 100
    
    # Hareketli Ortalama Sinyalleri
    signals['SMA5_Signal'] = np.where(df['Close'] > df['SMA5'], 1, -1)
    signals['SMA10_Signal'] = np.where(df['Close'] > df['SMA10'], 1, -1)
    signals['SMA20_Signal'] = np.where(df['Close'] > df['SMA20'], 1, -1)
    signals['SMA50_Signal'] = np.where(df['Close'] > df['SMA50'], 1, -1)
    signals['SMA100_Signal'] = np.where(df['Close'] > df['SMA100'], 1, -1)
    signals['SMA200_Signal'] = np.where(df['Close'] > df['SMA200'], 1, -1)
    
    signals['EMA5_Signal'] = np.where(df['Close'] > df['EMA5'], 1, -1)
    signals['EMA10_Signal'] = np.where(df['Close'] > df['EMA10'], 1, -1)
    signals['EMA20_Signal'] = np.where(df['Close'] > df['EMA20'], 1, -1)
    signals['EMA50_Signal'] = np.where(df['Close'] > df['EMA50'], 1, -1)
    signals['EMA100_Signal'] = np.where(df['Close'] > df['EMA100'], 1, -1)
    signals['EMA200_Signal'] = np.where(df['Close'] > df['EMA200'], 1, -1)
    
    # MACD Sinyali
    signals['MACD_Signal'] = np.where(df['MACD'] > df['MACD_Signal'], 1, -1)
    signals['MACD_Crossover'] = np.where(
        (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1)), 1, 
        np.where((df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1)), -1, 0)
    )
    
    # RSI Sinyali
    signals['RSI_Overbought'] = np.where(df['RSI'] > 70, -1, 0)
    signals['RSI_Oversold'] = np.where(df['RSI'] < 30, 1, 0)
    signals['RSI_Signal'] = signals['RSI_Overbought'] + signals['RSI_Oversold']
    
    # Bollinger Bant Sinyalleri
    signals['BB_Upper'] = np.where(df['Close'] > df['Upper_Band'], -1, 0)
    signals['BB_Lower'] = np.where(df['Close'] < df['Lower_Band'], 1, 0)
    signals['BB_Signal'] = signals['BB_Upper'] + signals['BB_Lower']
    
    # Stokastik Sinyaller
    signals['Stoch_Overbought'] = np.where((df['Stoch_%K'] > 80) & (df['Stoch_%D'] > 80), -1, 0)
    signals['Stoch_Oversold'] = np.where((df['Stoch_%K'] < 20) & (df['Stoch_%D'] < 20), 1, 0)
    signals['Stoch_Signal'] = signals['Stoch_Overbought'] + signals['Stoch_Oversold']
    
    # ADX Sinyali
    signals['ADX_Trend_Strength'] = np.where(df['ADX'] > 25, 1, 0)
    signals['ADX_Direction'] = np.where(df['Plus_DI'] > df['Minus_DI'], 1, -1)
    signals['ADX_Signal'] = signals['ADX_Trend_Strength'] * signals['ADX_Direction']
    
    # Williams %R
    signals['Williams_%R_Overbought'] = np.where(df['Williams_%R'] > -20, -1, 0)
    signals['Williams_%R_Oversold'] = np.where(df['Williams_%R'] < -80, 1, 0)
    signals['Williams_%R_Signal'] = signals['Williams_%R_Overbought'] + signals['Williams_%R_Oversold']
    
    # Toplam Sinyal
    signal_cols = [col for col in signals.columns if col.endswith('_Signal')]
    signals['Total_MA_Signal'] = signals[[col for col in signal_cols if 'SMA' in col or 'EMA' in col]].sum(axis=1)
    signals['Total_Oscillator_Signal'] = signals[[col for col in signal_cols if 'RSI' in col or 'Stoch' in col or 'MACD' in col or 'Williams' in col]].sum(axis=1)
    signals['Total_Signal'] = signals['Total_MA_Signal'] + signals['Total_Oscillator_Signal']
    
    return signals 