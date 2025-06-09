import pandas as pd
import numpy as np
from typing import Tuple

class TechnicalIndicators:
    """
    Basit teknik göstergeler hesaplama sınıfı
    """
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, period)
        rolling_std = data.rolling(window=period).std()
        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        return -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume"""
        direction = np.where(close.diff() > 0, 1, np.where(close.diff() < 0, -1, 0))
        return (volume * direction).cumsum()
    
    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """Money Flow Index"""
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume
        
        positive_flow = np.where(typical_price.diff() > 0, raw_money_flow, 0)
        negative_flow = np.where(typical_price.diff() < 0, raw_money_flow, 0)
        
        positive_mf = pd.Series(positive_flow).rolling(window=period).sum()
        negative_mf = pd.Series(negative_flow).rolling(window=period).sum()
        
        money_ratio = positive_mf / negative_mf
        return 100 - (100 / (1 + money_ratio))
    
    @staticmethod
    def roc(data: pd.Series, period: int = 10) -> pd.Series:
        """Rate of Change"""
        return ((data - data.shift(period)) / data.shift(period)) * 100
    
    @staticmethod
    def momentum(data: pd.Series, period: int = 10) -> pd.Series:
        """Momentum"""
        return data - data.shift(period)
    
    @staticmethod
    def parabolic_sar(high: pd.Series, low: pd.Series, acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
        """Parabolic SAR (Basitleştirilmiş ve Güvenli)"""
        try:
            sar = pd.Series(index=high.index, dtype=float)
            
            if len(high) < 2:
                return sar
            
            af = acceleration
            trend = 1  # 1 for uptrend, -1 for downtrend
            ep = high.iloc[0]  # Extreme point
            
            sar.iloc[0] = low.iloc[0]
            
            for i in range(1, len(high)):
                try:
                    if trend == 1:  # Uptrend
                        sar_value = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
                        
                        if high.iloc[i] > ep:
                            ep = high.iloc[i]
                            af = min(af + acceleration, maximum)
                        
                        if sar_value > low.iloc[i]:
                            trend = -1
                            sar_value = ep
                            ep = low.iloc[i]
                            af = acceleration
                    else:  # Downtrend
                        sar_value = sar.iloc[i-1] - af * (sar.iloc[i-1] - ep)
                        
                        if low.iloc[i] < ep:
                            ep = low.iloc[i]
                            af = min(af + acceleration, maximum)
                        
                        if sar_value < high.iloc[i]:
                            trend = 1
                            sar_value = ep
                            ep = high.iloc[i]
                            af = acceleration
                    
                    sar.iloc[i] = sar_value
                    
                except:
                    # Hata durumunda önceki değeri kullan
                    sar.iloc[i] = sar.iloc[i-1] if i > 0 else low.iloc[0]
            
            return sar
            
        except Exception:
            # Hata durumunda low değerlerini döndür
            return low.copy() 