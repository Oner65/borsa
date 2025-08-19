import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
try:
    from .technical_indicators import TechnicalIndicators
except ImportError:
    from technical_indicators import TechnicalIndicators
import logging
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class StockProfiler:
    """
    Her hisse senedi için kişiselleştirilmiş analiz profili oluşturan sınıf
    """
    
    def __init__(self, profile_dir="profiles"):
        self.profile_dir = profile_dir
        self.logger = logging.getLogger(__name__)
        
        # Profil klasörünü oluştur
        if not os.path.exists(profile_dir):
            os.makedirs(profile_dir)
        
        # Teknik göstergeler listesi
        self.indicators = {
            'momentum': ['RSI', 'STOCH_K', 'STOCH_D', 'WILLR', 'ROC', 'MOM'],
            'trend': ['SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26'],
            'volume': ['OBV', 'MFI'],
            'volatility': ['ATR', 'UPPER_BB', 'LOWER_BB'],
            'pattern': ['MACD', 'MACDSIGNAL', 'MACDHIST'],
            'overlap': ['SAR']
        }
        
        # En etkili göstergeler için skorlama sistemi
        self.indicator_scores = {}
        
    def calculate_all_indicators(self, data):
        """
        Tüm teknik göstergeleri hesapla (basit modül kullanarak)
        """
        df = data.copy()
        
        try:
            ta = TechnicalIndicators()
            
            # Momentum göstergeleri
            df['RSI'] = ta.rsi(df['Close'])
            df['STOCH_K'], df['STOCH_D'] = ta.stochastic(df['High'], df['Low'], df['Close'])
            df['WILLR'] = ta.williams_r(df['High'], df['Low'], df['Close'])
            df['ROC'] = ta.roc(df['Close'])
            df['MOM'] = ta.momentum(df['Close'])
            
            # Trend göstergeleri
            df['SMA_5'] = ta.sma(df['Close'], 5)
            df['SMA_10'] = ta.sma(df['Close'], 10)
            df['SMA_20'] = ta.sma(df['Close'], 20)
            df['SMA_50'] = ta.sma(df['Close'], 50)
            df['EMA_12'] = ta.ema(df['Close'], 12)
            df['EMA_26'] = ta.ema(df['Close'], 26)
            
            # Hacim göstergeleri
            df['OBV'] = ta.obv(df['Close'], df['Volume'])
            df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'])
            
            # Volatilite göstergeleri
            df['UPPER_BB'], df['MIDDLE_BB'], df['LOWER_BB'] = ta.bollinger_bands(df['Close'])
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'])
            
            # Pattern göstergeleri
            df['MACD'], df['MACDSIGNAL'], df['MACDHIST'] = ta.macd(df['Close'])
            
            # Diğer göstergeler
            df['SAR'] = ta.parabolic_sar(df['High'], df['Low'])
            
            # Özel göstergeler hesapla
            df['PRICE_TO_SMA20'] = df['Close'] / df['SMA_20']
            df['VOLUME_SMA'] = df['Volume'].rolling(20).mean()
            df['VOLUME_RATIO'] = df['Volume'] / df['VOLUME_SMA']
            df['HIGH_LOW_RATIO'] = (df['High'] - df['Low']) / df['Close']
            df['CLOSE_TO_HIGH'] = df['Close'] / df['High']
            
            # Support ve Resistance seviyeleri
            df['RESISTANCE'] = df['High'].rolling(20).max()
            df['SUPPORT'] = df['Low'].rolling(20).min()
            df['PRICE_POSITION'] = (df['Close'] - df['SUPPORT']) / (df['RESISTANCE'] - df['SUPPORT'])
                
        except Exception as e:
            self.logger.error(f"Gösterge hesaplama hatası: {e}")
        
        return df
    
    def detect_significant_moves(self, data, threshold=0.05):
        """
        Önemli fiyat hareketlerini tespit et
        """
        df = data.copy()
        
        # Günlük getiri hesapla
        df['Return'] = df['Close'].pct_change()
        
        # 1-5 gün sonraki getiriler
        for days in [1, 2, 3, 5]:
            df[f'Future_Return_{days}d'] = df['Close'].shift(-days) / df['Close'] - 1
            df[f'Significant_Move_{days}d'] = (df[f'Future_Return_{days}d'] > threshold).astype(int)
        
        return df
    
    def analyze_indicator_effectiveness(self, data, indicator_name, target_column='Significant_Move_3d'):
        """
        Belirli bir göstergenin etkinliğini analiz et
        """
        df = data.dropna()
        
        if len(df) < 50 or indicator_name not in df.columns:
            return {'score': 0, 'accuracy': 0, 'precision': 0, 'recall': 0}
        
        try:
            # Gösterge değerlerini normalize et
            scaler = StandardScaler()
            indicator_values = scaler.fit_transform(df[[indicator_name]])
            
            # Hedef değişken
            y = df[target_column].values
            
            if len(np.unique(y)) < 2:
                return {'score': 0, 'accuracy': 0, 'precision': 0, 'recall': 0}
            
            # Train/Test split (son %20 test için)
            split_idx = int(len(df) * 0.8)
            X_train, X_test = indicator_values[:split_idx], indicator_values[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Random Forest ile öğren
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X_train, y_train)
            
            # Tahmin yap
            y_pred = rf.predict(X_test)
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            # Metrikleri hesapla
            accuracy = accuracy_score(y_test, y_pred_binary)
            precision = precision_score(y_test, y_pred_binary, average='binary', zero_division=0)
            recall = recall_score(y_test, y_pred_binary, average='binary', zero_division=0)
            
            # Özellik önemini al
            feature_importance = rf.feature_importances_[0]
            
            # Genel skor hesapla
            score = (accuracy * 0.4 + precision * 0.3 + recall * 0.3) * feature_importance
            
            return {
                'score': score,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            self.logger.error(f"Gösterge analizi hatası {indicator_name}: {e}")
            return {'score': 0, 'accuracy': 0, 'precision': 0, 'recall': 0}
    
    def find_optimal_thresholds(self, data, indicator_name):
        """
        Gösterge için optimal eşik değerleri bul
        """
        df = data.dropna()
        
        if indicator_name not in df.columns:
            return {'buy_threshold': None, 'sell_threshold': None}
        
        try:
            # Başarılı alım/satım noktalarını bul
            successful_buys = df[df['Future_Return_3d'] > 0.03][indicator_name].values
            successful_sells = df[df['Future_Return_3d'] < -0.03][indicator_name].values
            
            if len(successful_buys) > 5:
                buy_threshold = np.percentile(successful_buys, 25)
            else:
                buy_threshold = None
                
            if len(successful_sells) > 5:
                sell_threshold = np.percentile(successful_sells, 75)
            else:
                sell_threshold = None
            
            return {
                'buy_threshold': buy_threshold,
                'sell_threshold': sell_threshold,
                'successful_buys': len(successful_buys),
                'successful_sells': len(successful_sells)
            }
            
        except Exception as e:
            self.logger.error(f"Eşik değer hesaplama hatası {indicator_name}: {e}")
            return {'buy_threshold': None, 'sell_threshold': None}
    
    def create_stock_profile(self, symbol, period="2y"):
        """
        Hisse senedi için detaylı profil oluştur
        """
        self.logger.info(f"{symbol} için profil oluşturuluyor...")
        
        try:
            # Veri çek
            ticker = yf.Ticker(f"{symbol}.IS")
            data = ticker.history(period=period)
            
            if len(data) < 100:
                self.logger.warning(f"{symbol} için yeterli veri yok")
                return None
            
            # Göstergeleri hesapla
            data_with_indicators = self.calculate_all_indicators(data)
            
            # Önemli hareketleri tespit et
            data_with_moves = self.detect_significant_moves(data_with_indicators)
            
            # Her gösterge için etkinlik analizi
            indicator_results = {}
            
            # Tüm gösterge kategorilerini kontrol et
            all_indicators = []
            for category in self.indicators.values():
                all_indicators.extend(category)
            
            # Ek hesaplanan göstergeler
            additional_indicators = [
                'RSI', 'STOCH_K', 'STOCH_D', 'WILLR', 'ROC', 'MOM',
                'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                'OBV', 'MFI', 'UPPER_BB', 'LOWER_BB', 'ATR',
                'MACD', 'MACDSIGNAL', 'MACDHIST', 'SAR', 'PRICE_TO_SMA20',
                'VOLUME_RATIO', 'HIGH_LOW_RATIO', 'PRICE_POSITION'
            ]
            
            for indicator in additional_indicators:
                if indicator in data_with_moves.columns:
                    result = self.analyze_indicator_effectiveness(data_with_moves, indicator)
                    thresholds = self.find_optimal_thresholds(data_with_moves, indicator)
                    
                    indicator_results[indicator] = {
                        **result,
                        **thresholds
                    }
            
            # En iyi göstergeleri sırala
            sorted_indicators = sorted(
                indicator_results.items(),
                key=lambda x: x[1]['score'],
                reverse=True
            )
            
            # Profil oluştur
            profile = {
                'symbol': symbol,
                'created_date': datetime.now().isoformat(),
                'data_period': period,
                'total_days': len(data),
                'analysis_summary': {
                    'total_indicators': len(indicator_results),
                    'top_3_indicators': [item[0] for item in sorted_indicators[:3]],
                    'average_accuracy': np.mean([r['accuracy'] for r in indicator_results.values()]),
                    'best_accuracy': max([r['accuracy'] for r in indicator_results.values()]) if indicator_results else 0
                },
                'indicator_rankings': dict(sorted_indicators),
                'trading_characteristics': self.analyze_trading_characteristics(data_with_moves),
                'volume_profile': self.analyze_volume_patterns(data_with_moves),
                'volatility_profile': self.analyze_volatility_patterns(data_with_moves),
                'success_rates': self.calculate_success_rates(data_with_moves)
            }
            
            # Profili kaydet
            self.save_profile(symbol, profile)
            
            return profile
            
        except Exception as e:
            self.logger.error(f"{symbol} profil oluşturma hatası: {e}")
            return None
    
    def analyze_trading_characteristics(self, data):
        """
        Hisse senedinin ticaret karakteristiklerini analiz et
        """
        df = data.dropna()
        
        return {
            'average_daily_return': df['Return'].mean(),
            'volatility': df['Return'].std(),
            'max_drawdown': (df['Close'] / df['Close'].cummax() - 1).min(),
            'trend_strength': self.calculate_trend_strength(df),
            'support_resistance_strength': self.calculate_sr_strength(df),
            'breakout_frequency': self.calculate_breakout_frequency(df),
            'reversal_frequency': self.calculate_reversal_frequency(df)
        }
    
    def analyze_volume_patterns(self, data):
        """
        Hacim desenlerini analiz et
        """
        df = data.dropna()
        
        # Hacim ile fiyat ilişkisi
        volume_price_corr = df['Volume'].corr(df['Return'].abs())
        
        # Yüksek hacimli günlerdeki performans
        high_volume_days = df[df['VOLUME_RATIO'] > 1.5]
        high_volume_performance = high_volume_days['Future_Return_1d'].mean() if len(high_volume_days) > 0 else 0
        
        return {
            'average_volume': df['Volume'].mean(),
            'volume_volatility': df['Volume'].std() / df['Volume'].mean(),
            'volume_price_correlation': volume_price_corr,
            'high_volume_performance': high_volume_performance,
            'volume_trend': np.polyfit(range(len(df)), df['Volume'].values, 1)[0]
        }
    
    def analyze_volatility_patterns(self, data):
        """
        Volatilite desenlerini analiz et
        """
        df = data.dropna()
        
        # Günlük volatilite
        df['Daily_Volatility'] = df['HIGH_LOW_RATIO']
        
        # Volatilite ile gelecek getiri ilişkisi
        volatility_return_corr = df['Daily_Volatility'].corr(df['Future_Return_3d'])
        
        return {
            'average_volatility': df['Daily_Volatility'].mean(),
            'volatility_trend': np.polyfit(range(len(df)), df['Daily_Volatility'].values, 1)[0],
            'volatility_return_correlation': volatility_return_corr,
            'high_volatility_threshold': df['Daily_Volatility'].quantile(0.8),
            'low_volatility_threshold': df['Daily_Volatility'].quantile(0.2)
        }
    
    def calculate_success_rates(self, data):
        """
        Çeşitli senaryolar için başarı oranlarını hesapla
        """
        df = data.dropna()
        
        success_rates = {}
        
        for days in [1, 3, 5]:
            col = f'Future_Return_{days}d'
            if col in df.columns:
                # Genel başarı oranı (pozitif getiri)
                success_rates[f'positive_return_{days}d'] = (df[col] > 0).mean()
                
                # %3+ getiri başarı oranı
                success_rates[f'significant_gain_{days}d'] = (df[col] > 0.03).mean()
                
                # %5+ kayıp riski
                success_rates[f'significant_loss_{days}d'] = (df[col] < -0.05).mean()
        
        return success_rates
    
    def calculate_trend_strength(self, data):
        """Trend gücünü hesapla"""
        if len(data) < 20:
            return 0
        
        # 20 günlük SMA eğimi
        slope = np.polyfit(range(20), data['SMA_20'].tail(20).values, 1)[0]
        return slope / data['Close'].iloc[-1]  # Normalize et
    
    def calculate_sr_strength(self, data):
        """Destek/Direnç gücünü hesapla"""
        if 'PRICE_POSITION' not in data.columns:
            return 0
        
        # Fiyatın destek/direnç aralığındaki pozisyonu
        return data['PRICE_POSITION'].tail(20).std()
    
    def calculate_breakout_frequency(self, data):
        """Kırılma sıklığını hesapla"""
        if len(data) < 50:
            return 0
        
        # Bollinger Band kırılmaları
        upper_breakouts = (data['Close'] > data['UPPER_BB']).sum()
        lower_breakouts = (data['Close'] < data['LOWER_BB']).sum()
        
        return (upper_breakouts + lower_breakouts) / len(data)
    
    def calculate_reversal_frequency(self, data):
        """Geri dönüş sıklığını hesapla"""
        if len(data) < 20:
            return 0
        
        # RSI aşırı alım/satım durumlarından geri dönüşler
        oversold_reversals = ((data['RSI'] < 30) & (data['Future_Return_3d'] > 0)).sum()
        overbought_reversals = ((data['RSI'] > 70) & (data['Future_Return_3d'] < 0)).sum()
        
        total_extreme_conditions = ((data['RSI'] < 30) | (data['RSI'] > 70)).sum()
        
        if total_extreme_conditions == 0:
            return 0
        
        return (oversold_reversals + overbought_reversals) / total_extreme_conditions
    
    def save_profile(self, symbol, profile):
        """Profili dosyaya kaydet"""
        filename = os.path.join(self.profile_dir, f"{symbol}_profile.json")
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"{symbol} profili kaydedildi: {filename}")
            
        except Exception as e:
            self.logger.error(f"Profil kaydetme hatası: {e}")
    
    def load_profile(self, symbol):
        """Kaydedilmiş profili yükle"""
        filename = os.path.join(self.profile_dir, f"{symbol}_profile.json")
        
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Profil yükleme hatası: {e}")
            return None
    
    def get_personalized_signals(self, symbol, current_data):
        """
        Kişiselleştirilmiş alım/satım sinyalleri üret
        """
        profile = self.load_profile(symbol)
        
        if not profile:
            self.logger.warning(f"{symbol} için profil bulunamadı")
            return None
        
        # Mevcut veri için göstergeleri hesapla
        data_with_indicators = self.calculate_all_indicators(current_data)
        latest_data = data_with_indicators.iloc[-1]
        
        signals = {
            'symbol': symbol,
            'date': latest_data.name,
            'signals': [],
            'overall_score': 0,
            'confidence': 0
        }
        
        # En iyi 5 göstergeyi kullan
        top_indicators = list(profile['indicator_rankings'].keys())[:5]
        
        total_score = 0
        total_weight = 0
        
        for indicator in top_indicators:
            if indicator in data_with_indicators.columns:
                indicator_info = profile['indicator_rankings'][indicator]
                current_value = latest_data[indicator]
                
                # Eşik değerleri
                buy_threshold = indicator_info.get('buy_threshold')
                sell_threshold = indicator_info.get('sell_threshold')
                
                # Sinyal üret
                signal_strength = 0
                signal_type = 'hold'
                
                if buy_threshold is not None and current_value >= buy_threshold:
                    signal_type = 'buy'
                    signal_strength = indicator_info['score']
                elif sell_threshold is not None and current_value <= sell_threshold:
                    signal_type = 'sell'
                    signal_strength = -indicator_info['score']
                
                signals['signals'].append({
                    'indicator': indicator,
                    'value': current_value,
                    'signal': signal_type,
                    'strength': signal_strength,
                    'accuracy': indicator_info['accuracy'],
                    'buy_threshold': buy_threshold,
                    'sell_threshold': sell_threshold
                })
                
                # Genel skora ekle
                weight = indicator_info['score']
                total_score += signal_strength * weight
                total_weight += weight
        
        # Genel değerlendirme
        if total_weight > 0:
            signals['overall_score'] = total_score / total_weight
            signals['confidence'] = min(total_weight * 10, 100)  # %0-100 arası
        
        # Sinyal yorumu
        if signals['overall_score'] > 0.3:
            signals['recommendation'] = 'BUY'
        elif signals['overall_score'] < -0.3:
            signals['recommendation'] = 'SELL'
        else:
            signals['recommendation'] = 'HOLD'
        
        return signals

# Test fonksiyonu
def test_stock_profiler():
    """Test fonksiyonu"""
    profiler = StockProfiler()
    
    # Birkaç hisse için profil oluştur
    test_stocks = ['THYAO', 'ASELS', 'BIST']
    
    for stock in test_stocks:
        print(f"\n{stock} için profil oluşturuluyor...")
        profile = profiler.create_stock_profile(stock, period="1y")
        
        if profile:
            print(f"✅ {stock} profili oluşturuldu")
            print(f"En iyi 3 gösterge: {profile['analysis_summary']['top_3_indicators']}")
            print(f"Ortalama doğruluk: {profile['analysis_summary']['average_accuracy']:.3f}")
        else:
            print(f"❌ {stock} profili oluşturulamadı")

if __name__ == "__main__":
    test_stock_profiler() 