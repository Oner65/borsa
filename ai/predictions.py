import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import pickle
from io import BytesIO
import sys
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# İçe aktarma için projenin ana dizinini ekleyebiliriz
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.indicators import calculate_indicators

def validate_and_clean_data(df, symbol):
    """
    Veri kalitesini kontrol eder ve temizler
    
    Args:
        df: Ham veri DataFrame'i
        symbol: Hisse senedi sembolü
        
    Returns:
        Temizlenmiş DataFrame veya None
    """
    try:
        print(f"DEBUG: {symbol} - Veri validasyonu başlatılıyor...")
        
        # Temel kontroller
        if df is None or df.empty:
            print(f"ERROR: {symbol} - Boş veri seti")
            return None
            
        if len(df) < 60:
            print(f"ERROR: {symbol} - Yetersiz veri ({len(df)} < 60 gün)")
            return None
        
        # Gerekli kolonları kontrol et
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"ERROR: {symbol} - Eksik kolonlar: {missing_cols}")
            return None
        
        # Veri tipi kontrolü
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Negatif fiyat kontrolü
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            negative_count = (df[col] <= 0).sum()
            if negative_count > 0:
                print(f"WARNING: {symbol} - {col} kolonunda {negative_count} negatif/sıfır değer bulundu")
                df = df[df[col] > 0]
        
        # Volume kontrolü
        if 'Volume' in df.columns:
            zero_volume = (df['Volume'] == 0).sum()
            if zero_volume > len(df) * 0.1:  # %10'dan fazla sıfır volume varsa
                print(f"WARNING: {symbol} - Volume verilerinde %{(zero_volume/len(df))*100:.1f} sıfır değer")
        
        # OHLC mantık kontrolü
        invalid_ohlc = ((df['High'] < df['Low']) | 
                       (df['High'] < df['Open']) | 
                       (df['High'] < df['Close']) |
                       (df['Low'] > df['Open']) | 
                       (df['Low'] > df['Close'])).sum()
        
        if invalid_ohlc > 0:
            print(f"WARNING: {symbol} - {invalid_ohlc} satırda geçersiz OHLC ilişkisi")
            # Geçersiz satırları kaldır
            valid_mask = ((df['High'] >= df['Low']) & 
                         (df['High'] >= df['Open']) & 
                         (df['High'] >= df['Close']) &
                         (df['Low'] <= df['Open']) & 
                         (df['Low'] <= df['Close']))
            df = df[valid_mask]
        
        # Outlier detection ve temizleme
        df_clean = detect_and_handle_outliers(df, symbol)
        
        # NaN değerleri temizle
        initial_len = len(df_clean)
        df_clean = df_clean.dropna()
        dropped = initial_len - len(df_clean)
        
        if dropped > 0:
            print(f"INFO: {symbol} - {dropped} satır NaN nedeniyle kaldırıldı")
        
        # Final kontrol
        if len(df_clean) < 60:
            print(f"ERROR: {symbol} - Temizleme sonrası yetersiz veri ({len(df_clean)} < 60)")
            return None
            
        print(f"SUCCESS: {symbol} - Veri validasyonu tamamlandı. Final veri uzunluğu: {len(df_clean)}")
        return df_clean
        
    except Exception as e:
        print(f"ERROR: {symbol} - Veri validasyonu hatası: {str(e)}")
        return None

def detect_and_handle_outliers(df, symbol, method='iqr'):
    """
    Outlier tespiti ve düzeltmesi
    
    Args:
        df: DataFrame
        symbol: Hisse senedi sembolü
        method: 'iqr' veya 'zscore'
        
    Returns:
        Temizlenmiş DataFrame
    """
    try:
        df_clean = df.copy()
        outlier_count = 0
        
        # Fiyat kolonları için outlier tespiti
        price_cols = ['Open', 'High', 'Low', 'Close']
        
        for col in price_cols:
            if col not in df_clean.columns:
                continue
                
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR  # 3*IQR daha sıkı kontrol
                upper_bound = Q3 + 3 * IQR
                
                outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df_clean[col]))
                outliers = z_scores > 3  # 3 sigma kuralı
            
            outlier_indices = df_clean[outliers].index
            outlier_count += len(outlier_indices)
            
            # Outlier'ları interpolasyon ile düzelt
            if len(outlier_indices) > 0:
                df_clean.loc[outlier_indices, col] = np.nan
                df_clean[col] = df_clean[col].interpolate(method='linear')
        
        # Volume için özel outlier kontrolü
        if 'Volume' in df_clean.columns and len(df_clean) > 10:
            volume_median = df_clean['Volume'].median()
            volume_mad = df_clean['Volume'].mad()  # Median Absolute Deviation
            
            # Volume için daha esnek sınırlar (çok yüksek volume'lar normal olabilir)
            volume_threshold = volume_median + 10 * volume_mad
            extreme_volume = df_clean['Volume'] > volume_threshold
            
            if extreme_volume.sum() > 0:
                print(f"INFO: {symbol} - {extreme_volume.sum()} aşırı yüksek volume değeri düzeltildi")
                df_clean.loc[extreme_volume, 'Volume'] = volume_median
        
        if outlier_count > 0:
            print(f"INFO: {symbol} - {outlier_count} outlier tespit edildi ve düzeltildi")
            
        return df_clean
        
    except Exception as e:
        print(f"ERROR: {symbol} - Outlier temizleme hatası: {str(e)}")
        return df

def create_deterministic_seed(symbol, model_type):
    """
    Sembole ve model tipine dayalı deterministik seed oluşturur
    """
    # Sembol ve model tipini birleştir
    combined = f"{symbol}_{model_type}"
    
    # Hash değeri hesapla
    hash_value = hash(combined)
    
    # Pozitif bir değer garanti et ve sınırla
    seed = abs(hash_value) % 10000
    
    return seed

def calculate_confidence_score(y_true, y_pred, model_type, data_quality_score, volatility):
    """
    Kapsamlı güven skoru hesaplama
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin değerleri
        model_type: Model tipi
        data_quality_score: Veri kalitesi skoru (0-1)
        volatility: Volatilite değeri
        
    Returns:
        Güven skoru (0-100)
    """
    try:
        # Temel metrikler
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Yön tahmini doğruluğu
        direction_accuracy = 0
        if len(y_true) > 1:
            for i in range(1, len(y_true)):
                actual_change = y_true[i] - y_true[i-1]
                predicted_change = y_pred[i] - y_pred[i-1]
                
                if (actual_change >= 0 and predicted_change >= 0) or (actual_change < 0 and predicted_change < 0):
                    direction_accuracy += 1
                    
            direction_accuracy = direction_accuracy / (len(y_true) - 1)
        
        # R2 bileşeni (ağırlık: 30%)
        r2_component = max(0, min(100, (r2 + 1) * 50))  # R2'yi 0-100 aralığına dönüştür
        
        # Yön doğruluğu bileşeni (ağırlık: 25%)
        direction_component = direction_accuracy * 100
        
        # MAPE bileşeni (ağırlık: 20%) - düşük MAPE iyi
        mape_component = max(0, 100 - mape * 2)  # MAPE ne kadar düşükse o kadar iyi
        
        # Veri kalitesi bileşeni (ağırlık: 15%)
        data_quality_component = data_quality_score * 100
        
        # Volatilite düzeltmesi (ağırlık: 10%) - yüksek volatilite güveni azaltır
        volatility_component = max(0, 100 - volatility * 5)
        
        # Model tipi bonus/malus
        model_bonus = 0
        if model_type == "Ensemble":
            model_bonus = 5
        elif model_type == "Hibrit Model":
            model_bonus = 3
        elif model_type in ["XGBoost", "LightGBM"]:
            model_bonus = 2
        elif model_type == "RandomForest":
            model_bonus = 1
        
        # Ağırlıklı toplam
        confidence = (
            r2_component * 0.30 +
            direction_component * 0.25 +
            mape_component * 0.20 +
            data_quality_component * 0.15 +
            volatility_component * 0.10 +
            model_bonus
        )
        
        # Sınırla (30-95 arası)
        confidence = max(30, min(95, confidence))
        
        return confidence
        
    except Exception as e:
        print(f"Güven skoru hesaplama hatası: {str(e)}")
        return 50.0  # Varsayılan değer

def calculate_data_quality_score(df):
    """
    Veri kalitesi skorunu hesaplar (0-1 arası)
    """
    try:
        # Veri uzunluğu faktörü
        length_factor = min(1.0, len(df) / 252)  # 1 yıl = 252 gün
        
        # NaN oranı faktörü
        nan_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        nan_factor = 1.0 - min(0.5, nan_ratio)  # %50'den fazla NaN varsa 0.5'e düşür
        
        # Volume tutarlılığı
        volume_factor = 1.0
        if 'Volume' in df.columns:
            zero_volume_ratio = (df['Volume'] == 0).sum() / len(df)
            volume_factor = 1.0 - min(0.3, zero_volume_ratio)  # %30'dan fazla sıfır volume varsa azalt
        
        # OHLC tutarlılığı
        ohlc_factor = 1.0
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            invalid_ohlc_ratio = ((df['High'] < df['Low']) | 
                                 (df['High'] < df['Open']) | 
                                 (df['High'] < df['Close']) |
                                 (df['Low'] > df['Open']) | 
                                 (df['Low'] > df['Close'])).sum() / len(df)
            ohlc_factor = 1.0 - min(0.2, invalid_ohlc_ratio)
        
        # Genel kalite skoru
        quality_score = (length_factor * 0.4 + 
                        nan_factor * 0.3 + 
                        volume_factor * 0.2 + 
                        ohlc_factor * 0.1)
        
        return max(0.0, min(1.0, quality_score))
        
    except Exception as e:
        print(f"Veri kalitesi hesaplama hatası: {str(e)}")
        return 0.5  # Varsayılan değer

def create_advanced_features(df):
    """
    Gelişmiş feature engineering - daha fazla teknik gösterge ve gecikmeli değerler
    """
    try:
        df = df.copy()
        
        # Gecikmeli fiyatlar (lag features)
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
            
        # Fiyat değişim oranları
        for period in [1, 3, 5, 10, 20]:
            df[f'price_change_{period}d'] = df['Close'].pct_change(period)
            df[f'volume_change_{period}d'] = df['Volume'].pct_change(period)
            
        # Hareketli ortalama çaprazlamaları
        df['sma5_sma20_ratio'] = df['SMA5'] / df['SMA20']
        df['price_sma20_ratio'] = df['Close'] / df['SMA20']
        
        # Volatilite göstergeleri
        for window in [5, 10, 20]:
            df[f'volatility_{window}d'] = df['Close'].rolling(window).std()
            df[f'price_range_{window}d'] = (df['High'].rolling(window).max() - df['Low'].rolling(window).min()) / df['Close']
            
        # Momentum göstergeleri
        for period in [5, 10, 20]:
            df[f'momentum_{period}d'] = df['Close'] / df['Close'].shift(period) - 1
            
        # Volume Price Trend (VPT)
        df['vpt'] = (df['Close'].pct_change() * df['Volume']).cumsum()
        
        # True Range ve Average True Range
        df['tr1'] = df['High'] - df['Low']
        df['tr2'] = (df['High'] - df['Close'].shift()).abs()
        df['tr3'] = (df['Low'] - df['Close'].shift()).abs()
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr_14'] = df['true_range'].rolling(14).mean()
        
        # Price Position (BB içindeki pozisyon)
        if all(col in df.columns for col in ['Upper_Band', 'Lower_Band']):
            df['bb_position'] = (df['Close'] - df['Lower_Band']) / (df['Upper_Band'] - df['Lower_Band'])
            
        # Stochastic %K ve %D farkları
        if all(col in df.columns for col in ['Stoch_%K', 'Stoch_%D']):
            df['stoch_k_d_diff'] = df['Stoch_%K'] - df['Stoch_%D']
            
        # MACD histogram trend
        if 'MACD_Hist' in df.columns:
            df['macd_hist_change'] = df['MACD_Hist'] - df['MACD_Hist'].shift(1)
            
        # Hacim moving averageları
        for period in [5, 10, 20]:
            df[f'volume_ma_{period}'] = df['Volume'].rolling(period).mean()
            df[f'volume_ma_ratio_{period}'] = df['Volume'] / df[f'volume_ma_{period}']
        
        # Temizlik
        df = df.drop(['tr1', 'tr2', 'tr3'], axis=1, errors='ignore')
        
        return df
        
    except Exception as e:
        print(f"Advanced feature engineering hatası: {str(e)}")
        return df

def optimize_model_hyperparameters(X_train, y_train, model_type, cv_folds=3):
    """
    Grid search ile model hiperparametre optimizasyonu
    """
    try:
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        
        print(f"DEBUG: {model_type} için hiperparametre optimizasyonu başlatılıyor...")
        
        if model_type == "RandomForest":
            from sklearn.ensemble import RandomForestRegressor
            
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [8, 12, 16, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
            
        elif model_type == "XGBoost":
            import xgboost as xgb
            
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
            
        elif model_type == "LightGBM":
            import lightgbm as lgb
            
            param_grid = {
                'n_estimators': [100, 200, 300],
                'num_leaves': [31, 50, 100],
                'learning_rate': [0.01, 0.1, 0.2],
                'feature_fraction': [0.8, 0.9, 1.0],
                'bagging_fraction': [0.8, 0.9, 1.0]
            }
            
            model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
        
        elif model_type == "Ensemble":
            # Ensemble modeli için basit parametre optimizasyonu
            # Her alt model için temel parametreleri optimize ederiz
            from sklearn.ensemble import RandomForestRegressor
            
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [8, 12, 16],
                'min_samples_split': [2, 5, 10],
                'learning_rate': [0.01, 0.1, 0.2]  # XGBoost/LightGBM için
            }
            
            # Ensemble için RandomForest parametrelerini optimize et
            # Sonuçları diğer modeller için de kullanacağız
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        else:
            return None
        
        # RandomizedSearchCV ile daha hızlı optimizasyon
        search = RandomizedSearchCV(
            model, 
            param_grid, 
            n_iter=20,  # Daha az iterasyon ile hızlandırma
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=42
        )
        
        search.fit(X_train, y_train)
        
        print(f"DEBUG: En iyi parametreler: {search.best_params_}")
        print(f"DEBUG: En iyi CV skoru: {-search.best_score_:.4f}")
        
        return search.best_estimator_
        
    except Exception as e:
        print(f"Hiperparametre optimizasyonu hatası: {str(e)}")
        return None

def walk_forward_validation(X, y, model_class, model_params, n_splits=5, test_size=0.2):
    """
    Walk-forward validation ile zaman serisi modeli değerlendirme
    """
    try:
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        n_samples = len(X)
        step_size = int(n_samples * test_size / n_splits)
        min_train_size = int(n_samples * 0.6)  # Minimum eğitim verisi
        
        scores = {'r2': [], 'mse': [], 'mae': [], 'direction_accuracy': []}
        
        for i in range(n_splits):
            # Test setinin başlangıç ve bitiş indeksleri
            test_start = min_train_size + i * step_size
            test_end = min(test_start + step_size, n_samples)
            
            if test_end - test_start < 10:  # Minimum test boyutu
                break
                
            # Train/test split
            X_train = X[:test_start]
            y_train = y[:test_start]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            
            # Model eğitimi
            if model_class == "RandomForest":
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(**model_params)
            elif model_class == "XGBoost":
                import xgboost as xgb
                model = xgb.XGBRegressor(**model_params)
            elif model_class == "LightGBM":
                import lightgbm as lgb
                model = lgb.LGBMRegressor(**model_params)
            elif model_class == "Ensemble":
                # Ensemble için basit bir yaklaşım - sadece RandomForest kullan
                from sklearn.ensemble import RandomForestRegressor
                rf_params = {k: v for k, v in model_params.items() if k in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 'random_state', 'n_jobs']}
                model = RandomForestRegressor(**rf_params)
            else:
                continue
                
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Skorları hesapla
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Yön doğruluğu hesapla
            actual_direction = (y_test[1:] > y_test[:-1]).astype(int)
            pred_direction = (y_pred[1:] > y_pred[:-1]).astype(int)
            direction_accuracy = (actual_direction == pred_direction).mean()
            
            scores['r2'].append(r2)
            scores['mse'].append(mse)
            scores['mae'].append(mae)
            scores['direction_accuracy'].append(direction_accuracy)
            
        # Ortalama skorları döndür
        avg_scores = {metric: np.mean(values) for metric, values in scores.items()}
        return avg_scores
        
    except Exception as e:
        print(f"Walk-forward validation hatası: {str(e)}")
        return {'r2': -1, 'mse': float('inf'), 'mae': float('inf'), 'direction_accuracy': 0.5}

def enhanced_confidence_score(y_true, y_pred, model_type, data_quality_score, volatility, walk_forward_scores=None):
    """
    Gelişmiş güven skoru hesaplama
    """
    try:
        confidence_factors = {}
        
        # 1. Model performansı (R² skoru)
        r2 = r2_score(y_true, y_pred)
        if r2 > 0.8:
            r2_score_normalized = 1.0
        elif r2 > 0.5:
            r2_score_normalized = 0.7 + (r2 - 0.5) * 0.6  # 0.5-0.8 arası için 0.7-0.88
        elif r2 > 0:
            r2_score_normalized = 0.4 + (r2 * 0.6)  # 0-0.5 arası için 0.4-0.7
        else:
            r2_score_normalized = max(0.1, 0.4 + r2 * 0.2)  # Negatif R² için 0.1-0.4
            
        confidence_factors['r2_factor'] = r2_score_normalized
        
        # 2. Yön doğruluğu
        actual_direction = (y_true[1:] > y_true[:-1]).astype(int)
        pred_direction = (y_pred[1:] > y_pred[:-1]).astype(int)
        direction_accuracy = (actual_direction == pred_direction).mean()
        confidence_factors['direction_factor'] = direction_accuracy
        
        # 3. Tahmin tutarlılığı (düşük varyans daha iyi)
        prediction_variance = np.var(y_pred - y_true) / np.var(y_true)
        consistency_factor = max(0.1, 1 - min(prediction_variance, 2))
        confidence_factors['consistency_factor'] = consistency_factor
        
        # 4. Veri kalitesi faktörü
        confidence_factors['data_quality_factor'] = data_quality_score
        
        # 5. Volatilite ayarlaması (düşük volatilite daha güvenilir)
        volatility_factor = max(0.5, 1 - (volatility / 100))
        confidence_factors['volatility_factor'] = volatility_factor
        
        # 6. Walk-forward validation skorları (varsa)
        if walk_forward_scores:
            wf_r2 = walk_forward_scores.get('r2', 0)
            wf_direction = walk_forward_scores.get('direction_accuracy', 0.5)
            wf_factor = (wf_r2 * 0.6 + wf_direction * 0.4)
            confidence_factors['walk_forward_factor'] = max(0.1, wf_factor)
        else:
            confidence_factors['walk_forward_factor'] = 0.6  # Varsayılan
            
        # 7. Model tipi güvenilirlik faktörü
        model_reliability = {
            "RandomForest": 0.85,
            "XGBoost": 0.90,
            "LightGBM": 0.88,
            "Ensemble": 0.92,
            "Hibrit Model": 0.80
        }
        confidence_factors['model_factor'] = model_reliability.get(model_type, 0.75)
        
        # Ağırlıklı ortalama ile final güven skoru
        weights = {
            'r2_factor': 0.25,
            'direction_factor': 0.20,
            'consistency_factor': 0.15,
            'data_quality_factor': 0.15,
            'volatility_factor': 0.10,
            'walk_forward_factor': 0.10,
            'model_factor': 0.05
        }
        
        final_confidence = sum(confidence_factors[factor] * weights[factor] 
                             for factor in confidence_factors)
        
        # 0-100 arasına ölçekle ve sınırla
        final_confidence = max(25, min(95, final_confidence * 100))
        
        print(f"DEBUG: Güven faktörleri: {confidence_factors}")
        print(f"DEBUG: Final güven skoru: {final_confidence:.1f}%")
        
        return final_confidence
        
    except Exception as e:
        print(f"Güven skoru hesaplama hatası: {str(e)}")
        return 45.0  # Güvenli varsayılan değer

def ml_price_prediction(stock_symbol, stock_data, days_to_predict=30, threshold=0.03, model_type="RandomForest", model_params=None, prediction_params=None):
    """
    Geliştirilmiş makine öğrenimi ile hisse senedi fiyat tahmini
    
    Args:
        stock_symbol: Hisse senedi sembolü
        stock_data: Fiyat verileri (DataFrame)
        days_to_predict: Tahmin edilecek gün sayısı
        threshold: Trend belirleme eşiği
        model_type: Kullanılacak model türü
        model_params: Model parametreleri
        prediction_params: Tahmin parametreleri
    
    Returns:
        dict: Tahmin sonuçları veya None (başarısız durumda)
    """
    
    try:
        print(f"DEBUG: {stock_symbol} için geliştirilmiş ML tahmin başlatılıyor - Model: {model_type}")
        
        # Deterministik seed kurulumu
        seed_value = create_deterministic_seed(stock_symbol, model_type)
        np.random.seed(seed_value)
        random.seed(seed_value)
        
        print(f"DEBUG: Deterministik seed kullanılıyor: {seed_value}")
        
        # Veriyi doğrula ve temizle
        df = validate_and_clean_data(stock_data.copy(), stock_symbol)
        if df is None:
            print(f"ERROR: {stock_symbol} - Veri validasyonu başarısız")
            return None
        
        # Veri kalitesi skorunu hesapla
        data_quality_score = calculate_data_quality_score(df)
        print(f"DEBUG: Veri kalitesi skoru: {data_quality_score:.3f}")
        
        # Teknik göstergeleri hesapla (eğer yoksa)
        try:
            if 'SMA20' not in df.columns:
                print(f"DEBUG: {stock_symbol} için teknik göstergeler hesaplanıyor...")
                df = calculate_indicators(df)
                print(f"DEBUG: Teknik göstergeler hesaplandı, toplam sütun sayısı: {len(df.columns)}")
        except Exception as indicator_error:
            print(f"Teknik gösterge hesaplama hatası: {str(indicator_error)}")
            # Temel göstergeleri kendimiz hesaplayalım
            for period in [5, 10, 20]:
                df[f'SMA{period}'] = df['Close'].rolling(window=period).mean()
        
        # Gelişmiş feature engineering
        print("DEBUG: Gelişmiş feature engineering uygulanıyor...")
        df = create_advanced_features(df)
        
        # Final temizlik - daha az agresif
        initial_length = len(df)
        df = df.dropna()
        removed_rows = initial_length - len(df)
        print(f"DEBUG: {removed_rows} satır NaN nedeniyle kaldırıldı ({initial_length} -> {len(df)})")
        
        if len(df) < 100:  # Minimum veri gereksinimi artırıldı
            print(f"ERROR: {stock_symbol} için yeterli veri yok: {len(df)} < 100")
            return None
        
        # Model parametrelerini kontrol et
        if model_params is None:
            model_params = {}
        
        if prediction_params is None:
            prediction_params = {}
        
        # Gelişmiş özellik seçimi
        base_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        technical_features = [col for col in df.columns if any(indicator in col.lower() 
                            for indicator in ['sma', 'ema', 'rsi', 'macd', 'band', 'stoch', 'adx'])]
        advanced_features = [col for col in df.columns if any(pattern in col.lower() 
                           for pattern in ['lag', 'change', 'ratio', 'momentum', 'volatility', 'vpt', 'atr'])]
        
        all_potential_features = base_features + technical_features + advanced_features
        
        # Kullanılabilir özellikleri filtrele - daha gevşek kriterler
        available_features = []
        for feature in all_potential_features:
            if feature in df.columns:
                non_na_ratio = df[feature].notna().sum() / len(df)
                if non_na_ratio > 0.7:  # %70 non-NA yeterli
                    available_features.append(feature)
        
        print(f"DEBUG: Kullanılabilir özellikler ({len(available_features)}): {available_features[:10]}...")
        
        if len(available_features) < 10:  # Minimum özellik sayısı artırıldı
            print(f"ERROR: Yeterli özellik yok. En az 10 gerekli, mevcut: {len(available_features)}")
            return None
        
        # Son fiyat ve volatilite hesapla
        last_price = df['Close'].iloc[-1]
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100  # Yıllık volatilite
        
        if pd.isna(volatility) or volatility <= 0:
            volatility = 3.0  # Varsayılan volatilite
            
        print(f"DEBUG: Son fiyat: {last_price:.2f}, Yıllık volatilite: {volatility:.2f}%")
        
        # Veri hazırlığı
        X = df[available_features].values
        y = df['Close'].values
            
        # Veri ölçeklendirme
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Walk-forward validation kullan
        use_walk_forward = prediction_params.get('use_walk_forward_validation', True)
        
        if use_walk_forward:
            print("DEBUG: Walk-forward validation ile model değerlendiriliyor...")
            
            # Model hiperparametre optimizasyonu
            print("DEBUG: Hiperparametre optimizasyonu yapılıyor...")
            optimized_model = optimize_model_hyperparameters(
                X_scaled[:-int(len(X_scaled)*0.2)], 
                y[:-int(len(y)*0.2)], 
                model_type, 
                cv_folds=3
            )
            
            if optimized_model is not None:
                # Optimized model parametrelerini al
                model_params_optimized = optimized_model.get_params()
                
                # Walk-forward validation skorları
                walk_forward_scores = walk_forward_validation(
                    X_scaled, y, model_type, model_params_optimized, n_splits=5
                )
                print(f"DEBUG: Walk-forward validation skorları: {walk_forward_scores}")
            else:
                walk_forward_scores = None
                model_params_optimized = model_params
        else:
            # Geleneksel train/test split
            split_index = int(len(X_scaled) * 0.8)
            walk_forward_scores = None
            model_params_optimized = model_params
        
        # Final model eğitimi
        print(f"DEBUG: Final {model_type} modeli eğitiliyor...")
        
        if model_type == "RandomForest":
            from sklearn.ensemble import RandomForestRegressor
            
            final_params = {
                'n_estimators': model_params_optimized.get('n_estimators', 200),
                'max_depth': model_params_optimized.get('max_depth', 15),
                'min_samples_split': model_params_optimized.get('min_samples_split', 5),
                'min_samples_leaf': model_params_optimized.get('min_samples_leaf', 2),
                'max_features': model_params_optimized.get('max_features', 'sqrt'),
                'random_state': seed_value,
                'n_jobs': -1
            }
            
            model = RandomForestRegressor(**final_params)
            
        elif model_type == "XGBoost":
            try:
                import xgboost as xgb
                
                final_params = {
                    'n_estimators': model_params_optimized.get('n_estimators', 200),
                    'max_depth': model_params_optimized.get('max_depth', 8),
                    'learning_rate': model_params_optimized.get('learning_rate', 0.1),
                    'subsample': model_params_optimized.get('subsample', 0.8),
                    'colsample_bytree': model_params_optimized.get('colsample_bytree', 0.8),
                    'random_state': seed_value,
                    'n_jobs': -1
                }
                
                model = xgb.XGBRegressor(**final_params)
                
            except ImportError:
                print("XGBoost kütüphanesi bulunamadı, RandomForest'a geçiliyor...")
                return ml_price_prediction(stock_symbol, stock_data, days_to_predict, threshold, "RandomForest", model_params, prediction_params)
            
        elif model_type == "LightGBM":
            try:
                import lightgbm as lgb
                
                final_params = {
                    'n_estimators': model_params_optimized.get('n_estimators', 200),
                    'num_leaves': model_params_optimized.get('num_leaves', 50),
                    'learning_rate': model_params_optimized.get('learning_rate', 0.1),
                    'feature_fraction': model_params_optimized.get('feature_fraction', 0.8),
                    'bagging_fraction': model_params_optimized.get('bagging_fraction', 0.8),
                    'random_state': seed_value,
                    'n_jobs': -1,
                    'verbose': -1
                }
                
                model = lgb.LGBMRegressor(**final_params)
                
            except ImportError:
                print("LightGBM kütüphanesi bulunamadı, RandomForest'a geçiliyor...")
                return ml_price_prediction(stock_symbol, stock_data, days_to_predict, threshold, "RandomForest", model_params, prediction_params)
        
        elif model_type == "Ensemble":
            try:
                # Ensemble modeli - çoklu model kombinasyonu
                from sklearn.ensemble import VotingRegressor, GradientBoostingRegressor
                
                # Model listesini hazırla
                models = []
                
                # RandomForest - her zaman ekle
                rf_params = {
                    'n_estimators': model_params_optimized.get('n_estimators', 200),
                    'max_depth': model_params_optimized.get('max_depth', 15),
                    'min_samples_split': model_params_optimized.get('min_samples_split', 5),
                    'min_samples_leaf': model_params_optimized.get('min_samples_leaf', 2),
                    'max_features': model_params_optimized.get('max_features', 'sqrt'),
                    'random_state': seed_value,
                    'n_jobs': -1
                }
                models.append(('rf', RandomForestRegressor(**rf_params)))
                
                # GradientBoosting - her zaman ekle
                gb_params = {
                    'n_estimators': model_params_optimized.get('n_estimators', 200),
                    'max_depth': model_params_optimized.get('max_depth', 8),
                    'learning_rate': model_params_optimized.get('learning_rate', 0.1),
                    'random_state': seed_value
                }
                models.append(('gb', GradientBoostingRegressor(**gb_params)))
                
                # XGBoost - eğer mevcut ise ekle
                try:
                    import xgboost as xgb
                    xgb_params = {
                        'n_estimators': model_params_optimized.get('n_estimators', 200),
                        'max_depth': model_params_optimized.get('max_depth', 8),
                        'learning_rate': model_params_optimized.get('learning_rate', 0.1),
                        'subsample': model_params_optimized.get('subsample', 0.8),
                        'colsample_bytree': model_params_optimized.get('colsample_bytree', 0.8),
                        'random_state': seed_value,
                        'n_jobs': -1
                    }
                    models.append(('xgb', xgb.XGBRegressor(**xgb_params)))
                    print("DEBUG: XGBoost ensemble'a eklendi")
                except ImportError:
                    print("DEBUG: XGBoost mevcut değil, ensemble'a eklenmiyor")
                
                # LightGBM - eğer mevcut ise ekle
                try:
                    import lightgbm as lgb
                    lgb_params = {
                        'n_estimators': model_params_optimized.get('n_estimators', 200),
                        'num_leaves': model_params_optimized.get('num_leaves', 50),
                        'learning_rate': model_params_optimized.get('learning_rate', 0.1),
                        'feature_fraction': model_params_optimized.get('feature_fraction', 0.8),
                        'bagging_fraction': model_params_optimized.get('bagging_fraction', 0.8),
                        'random_state': seed_value,
                        'n_jobs': -1,
                        'verbose': -1
                    }
                    models.append(('lgb', lgb.LGBMRegressor(**lgb_params)))
                    print("DEBUG: LightGBM ensemble'a eklendi")
                except ImportError:
                    print("DEBUG: LightGBM mevcut değil, ensemble'a eklenmiyor")
                
                print(f"DEBUG: Ensemble modeli {len(models)} alt model ile oluşturuluyor: {[name for name, _ in models]}")
                
                # VotingRegressor ile ensemble oluştur
                model = VotingRegressor(estimators=models, n_jobs=-1)
                
            except Exception as e:
                print(f"Ensemble model oluşturma hatası: {str(e)}, RandomForest'a geçiliyor...")
                return ml_price_prediction(stock_symbol, stock_data, days_to_predict, threshold, "RandomForest", model_params, prediction_params)
        
        else:
            # Varsayılan olarak RandomForest kullan
            print(f"WARNING: Bilinmeyen model tipi '{model_type}', RandomForest kullanılıyor...")
            
            final_params = {
                'n_estimators': model_params_optimized.get('n_estimators', 200),
                'max_depth': model_params_optimized.get('max_depth', 15),
                'min_samples_split': model_params_optimized.get('min_samples_split', 5),
                'min_samples_leaf': model_params_optimized.get('min_samples_leaf', 2),
                'max_features': model_params_optimized.get('max_features', 'sqrt'),
                'random_state': seed_value,
                'n_jobs': -1
            }
            
            model = RandomForestRegressor(**final_params)
        
        # Son %20 ile test için train/test ayırma
        split_index = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        print(f"DEBUG: Final eğitim veri boyutu: {len(X_train)}, Test veri boyutu: {len(X_test)}")
        
        # Model eğitimi
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        
        # Model performansını hesapla
        r2 = r2_score(y_test, y_pred_test)
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred_test)
        
        print(f"DEBUG: {model_type} performansı - R²: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        
        # Gelişmiş güven skorunu hesapla
        confidence = enhanced_confidence_score(
            y_test, y_pred_test, model_type, data_quality_score, 
            volatility, walk_forward_scores
        )
        
        # Gelecek tahminleri - sequence prediction
        future_dates = []
        future_pred_prices = []
        
        # Son veriyi al
        last_data = X_scaled[-1].reshape(1, -1)
        current_price = y[-1]
        
        # Sequence prediction için sliding window
        sequence_length = min(10, len(X_scaled))
        prediction_sequence = X_scaled[-sequence_length:]
        
        for i in range(1, days_to_predict + 1):
            # Bir sonraki iş gününü bul
            next_date = df.index[-1] + timedelta(days=i)
            # Hafta sonlarını atla
            while next_date.weekday() >= 5:  # 5 = Cumartesi, 6 = Pazar
                next_date = next_date + timedelta(days=1)
        
            future_dates.append(next_date)
            
            # Tahmin yap
            prediction = model.predict(last_data)[0]
            
            # Volatilite ayarlaması ekle
            volatility_adjustment = np.random.normal(0, volatility/100/np.sqrt(252)) if prediction_params.get('add_volatility', False) else 0
            adjusted_prediction = prediction * (1 + volatility_adjustment)
            
            future_pred_prices.append(adjusted_prediction)
            
            # Son veriyi güncelle (sadece fiyat değişikliği ile)
            # Bu basit bir yaklaşım, gerçekte yeni teknik göstergeler hesaplanmalı
            price_change_ratio = adjusted_prediction / current_price
            last_data = last_data.copy()
            # Close price feature'ını güncelle (eğer varsa)
            close_feature_idx = None
            for idx, feature in enumerate(available_features):
                if feature == 'Close':
                    close_feature_idx = idx
                    break
            if close_feature_idx is not None:
                last_data[0, close_feature_idx] *= price_change_ratio
    
        # Tahminleri DataFrame'e dönüştür
        predictions_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Price': future_pred_prices
        })
        predictions_df.set_index('Date', inplace=True)
    
        # Trend belirleme
        trend = "Yükseliş" if predictions_df['Predicted Price'].iloc[-1] > last_price else "Düşüş"
        
        # Sonuç dictionary'si oluştur
        result = {
            'symbol': stock_symbol,
            'model_type': model_type,
            'last_price': last_price,
            'current_price': last_price,
            'prediction_7d': predictions_df['Predicted Price'].iloc[min(6, len(predictions_df)-1)] if len(predictions_df) >= 7 else predictions_df['Predicted Price'].iloc[-1],
            'prediction_30d': predictions_df['Predicted Price'].iloc[-1] if days_to_predict >= 30 else None,
            'trend': trend,
            'confidence': confidence,
            'r2_score': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'volatility': volatility,
            'predictions_df': predictions_df,
            'features_used': available_features,
            'data_length': len(df),
            'data_quality_score': data_quality_score,
            'threshold_used': threshold,
            'seed_used': seed_value,
            'walk_forward_scores': walk_forward_scores,
            'optimized_params': model_params_optimized,
            'features_count': len(available_features)
        }
        
        print(f"SUCCESS: {stock_symbol} - {model_type} başarılı! R²={r2:.4f}, Güven=%{confidence:.1f}, Features={len(available_features)}")
        return result
        
    except Exception as e:
        import traceback
        print(f"ML fiyat tahmini hatası ({stock_symbol} - {model_type}): {str(e)}")
        print(traceback.format_exc())
        return None

def ai_price_prediction(stock_symbol, stock_data):
    """
    Fiyat tahmini yapar (yapay sonuçlar üretir)
    """
    # Tutarlı sonuçlar için sabit seed değerini sembole göre ayarla
    import random
    seed = sum(ord(c) for c in stock_symbol)
    random.seed(seed)
    
    current_price = stock_data['Close'].iloc[-1]
    
    # Fiyat değişim aralığını belirle
    price_change_range = current_price * 0.20  # Maksimum %20 değişim
    
    # Rastgele tahmin oluştur (iyimser bias ile)
    price_change = random.uniform(-price_change_range, price_change_range * 1.2)
    predicted_price = current_price + price_change
    
    trend = "Yükseliş" if predicted_price > current_price else "Düşüş"
    confidence = random.randint(60, 90)
    
    prediction = {
        'current_price': current_price,
        'predicted_price_7d': current_price * random.uniform(0.95, 1.05),
        'predicted_price_30d': predicted_price,
        'trend': trend,
        'confidence': confidence,
        'strength': random.randint(3, 9),
        'support_level': current_price * random.uniform(0.85, 0.98),
        'resistance_level': current_price * random.uniform(1.02, 1.15),
    }
    
    return prediction

def ai_technical_interpretation(stock_data, indicators=None):
    """
    Teknik göstergeleri yorumlar (yapay sonuçlar üretir)
    """
    if indicators is None:
        # Göstergeleri direkt veri çerçevesinden çıkarmaya çalış
        indicators = {}
        last_row = stock_data.iloc[-1]
        technical_indicators = [
            'RSI', 'MACD', 'MACD_Signal', 'SMA20', 'SMA50', 'SMA200',
            'Bollinger_High', 'Bollinger_Middle', 'Bollinger_Low',
            'Stoch_%K', 'Stoch_%D', 'Williams_R', 'ADX', 'CCI'
        ]
        
        for indicator in technical_indicators:
            if indicator in last_row:
                indicators[indicator] = round(float(last_row[indicator]), 2) if not pd.isna(last_row[indicator]) else "Mevcut değil"
            else:
                indicators[indicator] = "Mevcut değil"
    
    # Basit teknik yorumları
    interpretations = {}
    
    # RSI Yorumu
    rsi = indicators.get('RSI', 50)
    if rsi != "Mevcut değil":
        if rsi < 30:
            rsi_comment = "Aşırı satım bölgesinde, tepki yükselişi gelebilir"
            rsi_recommendation = "Al"
        elif rsi > 70:
            rsi_comment = "Aşırı alım bölgesinde, kar satışları gelebilir"
            rsi_recommendation = "Sat"
        elif rsi > 50:
            rsi_comment = "Yükseliş trendinde, momentum devam ediyor"
            rsi_recommendation = "Tut/Al"
        else:
            rsi_comment = "Düşüş trendinde, zayıf momentum"
            rsi_recommendation = "Tut/Sat"
    else:
        rsi_comment = "RSI verisi mevcut değil"
        rsi_recommendation = "Belirsiz"
        
    interpretations['RSI'] = {
        'value': rsi,
        'comment': rsi_comment,
        'recommendation': rsi_recommendation
    }
    
    # MACD Yorumu
    macd = indicators.get('MACD', 0)
    macd_signal = indicators.get('MACD_Signal', 0)
    
    if macd != "Mevcut değil" and macd_signal != "Mevcut değil":
        if macd > macd_signal:
            if macd > 0:
                macd_comment = "MACD, sinyal çizgisinin üstünde ve pozitif, güçlü yükseliş sinyali"
                macd_recommendation = "Al"
            else:
                macd_comment = "MACD, sinyal çizgisinin üstünde ama negatif, toparlanma sinyali"
                macd_recommendation = "Al/Tut"
        else:
            if macd < 0:
                macd_comment = "MACD, sinyal çizgisinin altında ve negatif, güçlü düşüş sinyali"
                macd_recommendation = "Sat"
            else:
                macd_comment = "MACD, sinyal çizgisinin altında ama pozitif, zayıflama sinyali"
                macd_recommendation = "Tut/Sat"
    else:
        macd_comment = "MACD verisi mevcut değil"
        macd_recommendation = "Belirsiz"
        
    interpretations['MACD'] = {
        'value': macd,
        'signal': macd_signal,
        'comment': macd_comment,
        'recommendation': macd_recommendation
    }
    
    # SMA Yorumu (20-50-200)
    sma_20 = indicators.get('SMA20', "Mevcut değil")
    sma_50 = indicators.get('SMA50', "Mevcut değil") 
    sma_200 = indicators.get('SMA200', "Mevcut değil")
    
    if sma_20 != "Mevcut değil" and sma_50 != "Mevcut değil" and sma_200 != "Mevcut değil":
        current_price = stock_data['Close'].iloc[-1]
        
        sma_comment = "Hareketli ortalamalar: "
        
        # Fiyat-SMA ilişkisi
        if current_price > sma_20 > sma_50 > sma_200:
            sma_comment += "Tüm HO'ların üzerinde, güçlü yükseliş trendi."
            sma_recommendation = "Al"
        elif current_price < sma_20 < sma_50 < sma_200:
            sma_comment += "Tüm HO'ların altında, güçlü düşüş trendi."
            sma_recommendation = "Sat"
        elif current_price > sma_20 and current_price > sma_50 and current_price < sma_200:
            sma_comment += "Kısa vadeli yükseliş, uzun vadeli direnç var."
            sma_recommendation = "Tut"
        elif current_price < sma_20 and current_price < sma_50 and current_price > sma_200:
            sma_comment += "Kısa vadeli düşüş, uzun vadeli destek var."
            sma_recommendation = "Tut"
        elif current_price > sma_20 and current_price < sma_50:
            sma_comment += "Kısa vadeli toparlanma sinyali."
            sma_recommendation = "Al/Tut"
        elif current_price < sma_20 and current_price > sma_50:
            sma_comment += "Kısa vadeli zayıflama sinyali."
            sma_recommendation = "Tut/Sat"
        else:
            sma_comment += "Karışık sinyaller gösteriyor."
            sma_recommendation = "Tut"
    else:
        sma_comment = "SMA verileri tam olarak mevcut değil"
        sma_recommendation = "Belirsiz"
        
    interpretations['SMA'] = {
        'SMA20': sma_20,
        'SMA50': sma_50,
        'SMA200': sma_200,
        'comment': sma_comment,
        'recommendation': sma_recommendation
    }
    
    # Bollinger Bantları Yorumu
    bb_high = indicators.get('Bollinger_High', "Mevcut değil")
    bb_middle = indicators.get('Bollinger_Middle', "Mevcut değil")
    bb_low = indicators.get('Bollinger_Low', "Mevcut değil")
    
    if bb_high != "Mevcut değil" and bb_middle != "Mevcut değil" and bb_low != "Mevcut değil":
        current_price = stock_data['Close'].iloc[-1]
        
        if current_price > bb_high:
            bb_comment = "Fiyat üst bandın üzerinde, aşırı alım bölgesinde"
            bb_recommendation = "Sat"
        elif current_price < bb_low:
            bb_comment = "Fiyat alt bandın altında, aşırı satım bölgesinde"
            bb_recommendation = "Al"
        elif current_price > bb_middle:
            bb_comment = "Fiyat orta bandın üzerinde, yükseliş eğiliminde"
            bb_recommendation = "Tut/Al"
        else:
            bb_comment = "Fiyat orta bandın altında, düşüş eğiliminde"
            bb_recommendation = "Tut/Sat"
    else:
        bb_comment = "Bollinger Bantları verisi mevcut değil"
        bb_recommendation = "Belirsiz"
        
    interpretations['Bollinger'] = {
        'High': bb_high,
        'Middle': bb_middle,
        'Low': bb_low,
        'comment': bb_comment,
        'recommendation': bb_recommendation
    }
    
    # Genel teknik yorum ve öneri
    recommendations = [info['recommendation'] for info in interpretations.values() if 'recommendation' in info]
    
    # Öneri eğilimini belirle
    buy_signals = recommendations.count('Al') + recommendations.count('Al/Tut') * 0.5
    sell_signals = recommendations.count('Sat') + recommendations.count('Tut/Sat') * 0.5
    hold_signals = recommendations.count('Tut')
    
    total_signals = len(recommendations)
    if total_signals == 0:
        overall_recommendation = "Yeterli veri yok"
    else:
        buy_ratio = buy_signals / total_signals
        sell_ratio = sell_signals / total_signals
        
        if buy_ratio > 0.6:
            overall_recommendation = "GÜÇLÜ AL"
        elif buy_ratio > 0.4:
            overall_recommendation = "AL"
        elif sell_ratio > 0.6:
            overall_recommendation = "GÜÇLÜ SAT"
        elif sell_ratio > 0.4:
            overall_recommendation = "SAT"
        else:
            overall_recommendation = "TUT"
    
    return {
        'indicators': indicators,
        'interpretations': interpretations,
        'overall_recommendation': overall_recommendation
    }

def ai_sector_analysis(sector="Bilinmeyen"):
    """
    Sektör analizi yapar (yapay sonuçlar üretir)
    """
    analysis = {
        "sector_name": sector,
        "outlook": random.choice(["Olumlu", "Nötr", "Olumsuz", "Karışık"]),
        "strength": random.randint(30, 90),
        "trend": random.choice(["Yükseliş", "Yatay", "Düşüş"]),
        "recommendation": random.choice(["Sektöre Ağırlık Ver", "Nötr Kal", "Sektörden Kaçın"]),
        "comments": f"{sector} sektörü için rastgele üretilmiş analiz yorumu."
    }
    
    return analysis

def detect_chart_patterns(stock_data):
    """
    Hisse senedi grafiğindeki formasyonları tespit eder (basit ve yapay)
    """
    # Tespit edilen formasyonları tutacak liste
    detected_patterns = []
    
    # Basit bir örnek: Çift dip tespit et
    try:
        close = stock_data['Close'].values
        
        # Son 40 günü incele
        window = min(40, len(close))
        last_prices = close[-window:]
        
        # Fiyat minimumlarını bul
        local_mins = []
        for i in range(1, len(last_prices)-1):
            if last_prices[i] < last_prices[i-1] and last_prices[i] < last_prices[i+1]:
                local_mins.append((i, last_prices[i]))
        
        # En az 2 minimum varsa
        if len(local_mins) >= 2:
            # Son iki miniumu al
            last_two_mins = sorted(local_mins, key=lambda x: x[0])[-2:]
            
            # Çift dip için kriterler
            min1_idx, min1_val = last_two_mins[0]
            min2_idx, min2_val = last_two_mins[1]
            
            # Minimum fiyatlar arasındaki fark %5'ten az ve aralarında en az 5 gün varsa
            price_diff_pct = abs(min1_val - min2_val) / min1_val * 100
            day_diff = min2_idx - min1_idx
            
            if price_diff_pct < 5 and day_diff >= 5:
                detected_patterns.append({
                    'type': 'double_bottom',
                    'description': 'Çift Dip Formasyonu',
                    'start_idx': min1_idx,
                    'end_idx': min2_idx,
                    'strength': random.uniform(0.6, 0.9),
                    'signal': 'bullish'
                })
    except Exception:
        pass  # Hata durumunda sessizce devam et
    
    # Rastgele bir formasyon daha ekle (demo amaçlı)
    if random.random() < 0.3:  # %30 olasılıkla
        pattern_types = [
            {'type': 'head_shoulders', 'description': 'Omuz Baş Omuz', 'signal': 'bearish'},
            {'type': 'inverse_head_shoulders', 'description': 'Ters Omuz Baş Omuz', 'signal': 'bullish'},
            {'type': 'ascending_triangle', 'description': 'Yükselen Üçgen', 'signal': 'bullish'},
            {'type': 'descending_triangle', 'description': 'Alçalan Üçgen', 'signal': 'bearish'},
            {'type': 'cup_and_handle', 'description': 'Fincan ve Kulp', 'signal': 'bullish'}
        ]
        
        pattern = random.choice(pattern_types)
        pattern['strength'] = random.uniform(0.5, 0.8)
        detected_patterns.append(pattern)
    
    return detected_patterns

def backtest_models(df_with_indicators, period=30, model_type="RandomForest", test_size=180):
    """
    Modellerin geçmiş performansını test etmek için backtest fonksiyonu
    
    Args:
        df_with_indicators: İndikatörleri içeren DataFrame
        period: Tahmin dönemi (gün)
        model_type: Kullanılacak model tipi
        test_size: Test edilecek gün sayısı
        
    Returns:
        Backtest sonuçları içeren dictionary
    """
    # Veriler yeterli mi kontrol et
    if len(df_with_indicators) <= test_size + period:
        return {
            'error': 'Backtest için yeterli veri yok',
            'mae': np.nan,
            'rmse': np.nan,
            'accuracy': np.nan,
            'predictions': None
        }
    
    try:
        # Tutarlı sonuçlar için sabit seed değeri kullan
        import numpy as np
        import random
        
        # Sabit seed oluştur - model tipi ve dönem kombinasyonu ile
        seed_value = sum(ord(c) for c in f"{model_type}_{period}") % 10000
        np.random.seed(seed_value)
        random.seed(seed_value)
        
        # Veri setini hazırla
        df = df_with_indicators.copy()
        
        # Özellik sütunları ve hedef sütunu - Hata veren yerde features değişkenini tanımlayalım
        features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA5', 'SMA10', 'SMA20', 'EMA5', 'EMA10', 'EMA20', 
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 
            'Upper_Band', 'Middle_Band', 'Lower_Band',
            'Stoch_%K', 'Stoch_%D', 'ADX'
        ]
        
        # Eksik feature'lar varsa kaldır
        features_to_use = [f for f in features if f in df.columns]
        
        # Veri ön işleme iyileştirmesi
        # NaN değerlerini daha akıllı bir şekilde doldur
        for col in features_to_use:
            # Teknik göstergeler
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                # İlk interpolasyon dene
                df[col] = df[col].interpolate(method='linear').bfill()
            else:
                # Fiyat verileri için ffill daha uygun
                df[col] = df[col].ffill().bfill()
        
        # Kalan NaN değerleri temizle
        df = df.dropna()
        
        # Yeterli veri kontrolü
        if len(df) <= test_size + period:
            return {
                'error': 'Ön işleme sonrası backtest için yeterli veri kalmadı',
                'mae': np.nan,
                'rmse': np.nan,
                'accuracy': np.nan,
                'predictions': None
            }
        
        # Tahmin sonuçlarını saklamak için DataFrame
        predictions_df = pd.DataFrame()
        
        # Test günleri için döngü
        all_actual_prices = []
        all_predicted_prices = []
        
        # Zaman serisi çapraz doğrulama
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Modelin ilerleme kontrolü için değişkenler
        last_percent = 0
        
        for i in range(test_size):
            # İlerleme durumunu yazdır (her %10'da bir)
            current_percent = int((i / test_size) * 100)
            if current_percent % 10 == 0 and current_percent != last_percent:
                print(f"Backtest ilerleme: %{current_percent} tamamlandı...")
                last_percent = current_percent
            train_end_idx = len(df) - test_size + i - period
            if train_end_idx <= 0:
                continue
            if train_end_idx + period - 1 >= len(df):
                continue
            train_data = df.iloc[:train_end_idx]
            actual_date = df.index[train_end_idx + period - 1]
            actual_price = df.iloc[train_end_idx + period - 1]['Close']
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_data[features_to_use])
            y_train = train_data['Close'].values
            X_last = scaler.transform(df.iloc[train_end_idx - 1:train_end_idx][features_to_use])
            try:
                # Model seçimi ve hiperparametre optimizasyonu
                if model_type == "RandomForest":
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.model_selection import RandomizedSearchCV
                    rf_param_grid = {
                        'n_estimators': [50, 100, 150, 200],
                        'max_depth': [5, 8, 10, 15, None],
                        'min_samples_split': [2, 5, 8],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': ['sqrt', 'log2', None]
                    }
                    if len(X_train) > 500:
                        n_iter = 10
                    else:
                        n_iter = 5
                    base_model = RandomForestRegressor(random_state=42)
                    grid_search = RandomizedSearchCV(base_model, rf_param_grid, 
                                                   n_iter=n_iter, cv=tscv, 
                                                   scoring='neg_mean_squared_error',
                                                   n_jobs=-1, random_state=42)
                    grid_search.fit(X_train, y_train)
                    model = grid_search.best_estimator_
                    prediction = model.predict(X_last)[0]
                elif model_type == "XGBoost":
                    try:
                        import xgboost as xgb
                        from sklearn.model_selection import RandomizedSearchCV
                        xgb_param_grid = {
                            'learning_rate': [0.01, 0.05, 0.1, 0.2],
                            'max_depth': [3, 5, 7, 9],
                            'n_estimators': [50, 100, 200],
                            'subsample': [0.7, 0.8, 0.9, 1.0],
                            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                            'min_child_weight': [1, 3, 5],
                            'gamma': [0, 0.1, 0.2, 0.3]
                        }
                        if len(X_train) > 500:
                            n_iter = 10
                        else:
                            n_iter = 5
                        base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
                        grid_search = RandomizedSearchCV(base_model, xgb_param_grid, 
                                                       n_iter=n_iter, cv=tscv, 
                                                       scoring='neg_mean_squared_error',
                                                       n_jobs=-1, random_state=42)
                        grid_search.fit(X_train, y_train)
                        model = grid_search.best_estimator_
                        prediction = model.predict(X_last)[0]
                    except (ImportError, ModuleNotFoundError):
                        print("XGBoost kütüphanesi bulunamadı, RandomForest kullanılıyor...")
                        from sklearn.ensemble import RandomForestRegressor
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                        model.fit(X_train, y_train)
                        prediction = model.predict(X_last)[0]
                elif model_type == "LightGBM":
                    try:
                        import lightgbm as lgb
                        from sklearn.model_selection import RandomizedSearchCV
                        lgb_param_grid = {
                            'learning_rate': [0.01, 0.05, 0.1, 0.2],
                            'num_leaves': [31, 50, 70, 100],
                            'n_estimators': [50, 100, 200],
                            'max_depth': [3, 5, 7, 9, -1],
                            'subsample': [0.7, 0.8, 0.9, 1.0],
                            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                            'reg_alpha': [0, 0.1, 0.5, 1.0],
                            'reg_lambda': [0, 0.1, 0.5, 1.0]
                        }
                        if len(X_train) > 500:
                            n_iter = 10
                        else:
                            n_iter = 5
                        base_model = lgb.LGBMRegressor(random_state=42)
                        grid_search = RandomizedSearchCV(base_model, lgb_param_grid, 
                                                       n_iter=n_iter, cv=tscv, 
                                                       scoring='neg_mean_squared_error',
                                                       n_jobs=-1, random_state=42)
                        grid_search.fit(X_train, y_train)
                        model = grid_search.best_estimator_
                        prediction = model.predict(X_last)[0]
                    except (ImportError, ModuleNotFoundError):
                        print("LightGBM kütüphanesi bulunamadı, RandomForest kullanılıyor...")
                        from sklearn.ensemble import RandomForestRegressor
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                        model.fit(X_train, y_train)
                        prediction = model.predict(X_last)[0]
                elif model_type == "Ensemble":
                    try:
                        from sklearn.ensemble import RandomForestRegressor, VotingRegressor
                        import xgboost as xgb
                        xgb_available = True
                    except ImportError:
                        xgb_available = False
                    
                    try:
                        import lightgbm as lgb
                        lgbm_available = True
                    except ImportError:
                        lgbm_available = False
                    
                    # Veriyi hazırla
                    X = df[features_to_use].values
                    y = df['Close'].values
                    
                    # Min-Max Ölçekleme
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_data = scaler.fit_transform(X)
                    
                    # Modelleri oluştur
                    models = []
                    
                    # RandomForest her zaman ekle
                    models.append(('rf', RandomForestRegressor(n_estimators=100, random_state=42)))
                    
                    # GradientBoosting her zaman ekle
                    models.append(('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)))
                    
                    # XGBoost varsa ekle
                    if xgb_available:
                        models.append(('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42)))
                    
                    # LightGBM varsa ekle
                    if lgbm_available:
                        models.append(('lgbm', lgb.LGBMRegressor(n_estimators=100, random_state=42)))
                    
                    # Ensemble tahmin fonksiyonu
                    def ensemble_predict(X):
                        predictions = []
                        for name, model in models:
                            predictions.append(model.predict(X))
                        
                        # Ortalama tahmin
                        return np.mean(predictions, axis=0)
                    
                    # Zaman serisi çapraz doğrulama
                    tscv = TimeSeriesSplit(n_splits=5)
                    y_true = []
                    y_pred = []
                    
                    for train_index, test_index in tscv.split(scaled_data):
                        X_train, X_test = scaled_data[train_index], scaled_data[test_index]
                        y_train, y_test = y[train_index], y[test_index]
                        
                        # Her modeli eğit
                        for name, model in models:
                            model.fit(X_train, y_train)
                        
                        # Ensemble tahmin
                        ensemble_predictions = ensemble_predict(X_test)
                        y_true.extend(y_test)
                        y_pred.extend(ensemble_predictions)
                    
                    # Son modeli tüm veri ile eğit
                    for name, model in models:
                        model.fit(scaled_data, y)
                    
                    # Performans metrikleri
                    mse = mean_squared_error(y_true, y_pred)
                    r2 = r2_score(y_true, y_pred)
                    
                    # Moving window boyutu - Zaman serisi modellemesi için
                    window_size = 20
                    
                    # Gelecek tahminleri
                    future_dates = []
                    future_pred_prices = []
                    
                    # Son pencereyi al
                    last_window = scaled_data[-window_size:]
                    
                    # Tahmin parametresi period kullanılarak days_to_predict yerine
                    days_to_predict = period
                    
                    # Gelecek günleri tahmin et
                    for i in range(1, days_to_predict + 1):
                        # Bir sonraki iş gününü bul
                        next_date = df.index[-1] + timedelta(days=i)
                        # Hafta sonlarını atla
                        while next_date.weekday() >= 5:  # 5 = Cumartesi, 6 = Pazar
                            next_date = next_date + timedelta(days=1)
                        
                        future_dates.append(next_date)
                        
                        # Son veriyi al
                        last_data = last_window[-1].reshape(1, -1)
                        
                        # Tahmin yap
                        prediction = ensemble_predict(last_data)[0]
                        future_pred_prices.append(prediction)
                    
                    # Tahminleri DataFrame'e dönüştür
                    predictions_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted Price': future_pred_prices
                    })
                    predictions_df.set_index('Date', inplace=True)
                elif model_type == "Hibrit Model":
                    try:
                        from sklearn.ensemble import RandomForestRegressor
                        rf_model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
                        rf_model.fit(X_train, y_train)
                        base_pred = rf_model.predict(X_last)[0]
                        rsi_factor = 0
                        macd_factor = 0
                        bb_factor = 0
                        if 'RSI' in df.columns:
                            last_rsi = df.iloc[train_end_idx - 1]['RSI']
                            if not pd.isna(last_rsi):
                                if last_rsi < 30:
                                    rsi_factor = (30 - last_rsi) / 100
                                elif last_rsi > 70:
                                    rsi_factor = (70 - last_rsi) / 100
                        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
                            last_macd = df.iloc[train_end_idx - 1]['MACD']
                            last_signal = df.iloc[train_end_idx - 1]['MACD_Signal']
                            if not pd.isna(last_macd) and not pd.isna(last_signal):
                                macd_diff = (last_macd - last_signal) / 10
                                macd_factor = max(min(macd_diff, 0.05), -0.05)
                        if all(band in df.columns for band in ['Upper_Band', 'Middle_Band', 'Lower_Band']):
                            last_close = df.iloc[train_end_idx - 1]['Close']
                            upper_band = df.iloc[train_end_idx - 1]['Upper_Band']
                            lower_band = df.iloc[train_end_idx - 1]['Lower_Band']
                            middle_band = df.iloc[train_end_idx - 1]['Middle_Band']
                            if not pd.isna(upper_band) and not pd.isna(lower_band) and not pd.isna(middle_band):
                                if last_close < middle_band:
                                    distance_to_lower = max(0.01, (last_close - lower_band) / (middle_band - lower_band))
                                    bb_factor = 0.03 * (1 - distance_to_lower)
                                else:
                                    distance_to_upper = max(0.01, (upper_band - last_close) / (upper_band - middle_band))
                                    bb_factor = -0.03 * (1 - distance_to_upper)
                        combined_factor = rsi_factor + macd_factor + bb_factor
                        combined_factor = max(min(combined_factor, 0.08), -0.08)
                        prediction = base_pred * (1 + combined_factor)
                    except Exception as e:
                        print(f"Hibrit model hatası: {str(e)}, RandomForest kullanılıyor...")
                        from sklearn.ensemble import RandomForestRegressor
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                        model.fit(X_train, y_train)
                        prediction = model.predict(X_last)[0]
                else:
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    prediction = model.predict(X_last)[0]
                all_actual_prices.append(actual_price)
                all_predicted_prices.append(prediction)
                predictions_df = pd.concat([predictions_df, pd.DataFrame({
                    'Date': [actual_date],
                    'Actual': [actual_price],
                    'Predicted': [prediction]
                })])
            except Exception as e:
                import traceback
                print(f"Backtest iterasyon hatası (i={i}): {str(e)}")
                print(traceback.format_exc())
                continue
        
        # Performans metrikleri
        if len(all_actual_prices) > 0:
            # Temel metrikler
            mae = mean_absolute_error(all_actual_prices, all_predicted_prices)
            rmse = np.sqrt(mean_squared_error(all_actual_prices, all_predicted_prices))
            
            # R2 skoru güvenlik kontrolü
            if len(set(all_actual_prices)) > 1:  # Veri çeşitliliği kontrolü
                r2 = r2_score(all_actual_prices, all_predicted_prices)
                if r2 < -1:  # Çok kötü R2 değerlerini sınırla
                    r2 = -1
            else:
                r2 = 0  # Tek bir değer varsa R2 hesaplanamaz
            
            # Trend doğruluğu (yön tahmini)
            correct_direction = 0
            for i in range(1, len(all_actual_prices)):
                actual_change = all_actual_prices[i] - all_actual_prices[i-1]
                predicted_change = all_predicted_prices[i] - all_predicted_prices[i-1]
                
                if (actual_change >= 0 and predicted_change >= 0) or (actual_change < 0 and predicted_change < 0):
                    correct_direction += 1
            
            direction_accuracy = correct_direction / (len(all_actual_prices) - 1) if len(all_actual_prices) > 1 else 0
            
            # MAPE hesapla (Ortalama Mutlak Yüzde Hata)
            mape = np.mean(np.abs((np.array(all_actual_prices) - np.array(all_predicted_prices)) / np.array(all_actual_prices))) * 100
            
            # Sonuçları indexle birlikte döndür
            predictions_df = predictions_df.set_index('Date')
            
            return {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'accuracy': direction_accuracy,
                'predictions': predictions_df,
                'total_predictions': len(all_actual_prices)
            }
        else:
            return {
                'error': 'Backtest sırasında bir hata oluştu',
                'mae': np.nan,
                'rmse': np.nan,
                'accuracy': np.nan,
                'predictions': None
            }
            
    except Exception as e:
        import traceback
        print(f"Backtest ana fonksiyon hatası: {str(e)}")
        print(traceback.format_exc())
        
        return {
            'error': str(e),
            'mae': np.nan,
            'rmse': np.nan,
            'accuracy': np.nan,
            'predictions': None
        } 

def train_and_save_model(symbol, stock_data, model_type="RandomForest", force_update=False, model_version=None):
    """
    Belirli bir hisse için model eğitir ve veritabanına kaydeder.
    
    Args:
        symbol (str): Hisse sembolü
        stock_data (DataFrame): Hisse verileri
        model_type (str): Model tipi (RandomForest, XGBoost, LightGBM, Ensemble, Hibrit)
        force_update (bool): True ise, mevcut model olsa bile yeniden eğitir
        model_version (str): Kaydedilecek model versiyonu. Belirtilmezse otomatik atanır.
        
    Returns:
        bool: İşlem başarılıysa True, değilse False
    """
    # data.db_utils içe aktarmasını projenin ana dizinine göre düzeltiyoruz
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from data.db_utils import save_ml_model, load_ml_model, get_model_update_status
    
    try:
        # Sembolü düzenle
        symbol = symbol.upper().strip()
        
        # Eğer force_update değilse, mevcut modelin güncelliğini kontrol et
        if not force_update:
            update_status = get_model_update_status(symbol)
            if model_type in update_status and not update_status[model_type]['needs_update']:
                print(f"{symbol} için {model_type} modeli zaten güncel. Son güncelleme: {update_status[model_type]['last_update']}")
                return True
                
        # Veriyi hazırla
        df = stock_data.copy()
        
        # Teknik göstergeleri hesapla
        df = calculate_indicators(df)
        
        # NaN değerleri temizle
        df = df.dropna()
        
        # Özellik sütunları
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA5', 'SMA10', 'SMA20', 'EMA5', 'EMA10', 'EMA20', 
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 
            'Upper_Band', 'Middle_Band', 'Lower_Band',
            'Stoch_%K', 'Stoch_%D', 'ADX'
        ]
        
        # Eksik feature'lar varsa kaldır
        features_to_use = [f for f in feature_columns if f in df.columns]
        
        # Veri setini eğitim ve test olarak ayır
        X = df[features_to_use].values
        y = df['Close'].values
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Zaman serisi çapraz doğrulama için split oluştur
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Hiperparametre optimizasyonu için RandomizedSearchCV kullan
        from sklearn.model_selection import RandomizedSearchCV
        
        # Model eğitim parametrelerini sakla (versiyonlama için)
        model_params = {}
        
        # Modeli eğit
        if model_type == "RandomForest":
            from sklearn.ensemble import RandomForestRegressor
            
            # RandomForest için geliştirilmiş parametreler
            rf_param_grid = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            # Parametre optimizasyonu
            rf_base_model = RandomForestRegressor(random_state=42)
            grid_search = RandomizedSearchCV(
                rf_base_model, 
                rf_param_grid, 
                n_iter=15, 
                cv=tscv, 
                n_jobs=-1, 
                scoring='neg_mean_squared_error',
                random_state=42
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            
            # Model parametrelerini kaydet
            model_params = {
                'type': 'RandomForest',
                'best_params': grid_search.best_params_,
                'train_size': train_size,
                'feature_count': len(features_to_use)
            }
            
            print(f"RandomForest en iyi parametreler: {grid_search.best_params_}")
            
        elif model_type == "XGBoost":
            try:
                import xgboost as xgb
                # XGBoost için geliştirilmiş parametreler
                xgb_param_grid = {
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7, 9],
                    'n_estimators': [50, 100, 200, 300],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                    'min_child_weight': [1, 3, 5, 7],
                    'gamma': [0, 0.1, 0.2, 0.3]
                }
                # Parametre optimizasyonu
                xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
                grid_search = RandomizedSearchCV(
                    xgb_model, 
                    xgb_param_grid, 
                    n_iter=15, 
                    cv=tscv, 
                    n_jobs=-1, 
                    scoring='neg_mean_squared_error',
                    random_state=42
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                model.fit(X_train, y_train)
                # Model parametrelerini kaydet
                model_params = {
                    'type': 'XGBoost',
                    'best_params': grid_search.best_params_,
                    'train_size': train_size,
                    'feature_count': len(features_to_use)
                }
                print(f"XGBoost en iyi parametreler: {grid_search.best_params_}")
            except ImportError:
                # XGBoost yüklü değilse RandomForest kullan
                print("XGBoost kütüphanesi bulunamadı, RandomForest kullanılıyor.")
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                # Model parametrelerini kaydet
                model_params = {
                    'type': 'RandomForest (XGBoost fallback)',
                    'params': {'n_estimators': 100, 'random_state': 42},
                    'train_size': train_size,
                    'feature_count': len(features_to_use)
                }
        elif model_type == "LightGBM":
            try:
                import lightgbm as lgb
                # LightGBM için geliştirilmiş parametreler
                lgb_param_grid = {
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'num_leaves': [31, 50, 70, 100, 150],
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 7, 9, -1],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.1, 0.5, 1.0],
                    'reg_lambda': [0, 0.1, 0.5, 1.0]
                }
                # Parametre optimizasyonu
                lgb_model = lgb.LGBMRegressor(random_state=42)
                grid_search = RandomizedSearchCV(
                    lgb_model, 
                    lgb_param_grid, 
                    n_iter=15, 
                    cv=tscv, 
                    n_jobs=-1, 
                    scoring='neg_mean_squared_error',
                    random_state=42
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                model.fit(X_train, y_train)
                # Model parametrelerini kaydet
                model_params = {
                    'type': 'LightGBM',
                    'best_params': grid_search.best_params_,
                    'train_size': train_size,
                    'feature_count': len(features_to_use)
                }
                print(f"LightGBM en iyi parametreler: {grid_search.best_params_}")
            except ImportError:
                # LightGBM yüklü değilse RandomForest kullan
                print("LightGBM kütüphanesi bulunamadı, RandomForest kullanılıyor.")
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                # Model parametrelerini kaydet
                model_params = {
                    'type': 'RandomForest (LightGBM fallback)',
                    'params': {'n_estimators': 100, 'random_state': 42},
                    'train_size': train_size,
                    'feature_count': len(features_to_use)
                }
        elif model_type == "Ensemble":
            # Ensemble için birden fazla model oluştur ve optimize et
            from sklearn.ensemble import RandomForestRegressor, VotingRegressor
            import numpy as np
            
            models = []
            weights = []
            
            # RandomForest
            rf_param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None]
            }
            
            rf_model = RandomForestRegressor(random_state=42)
            rf_search = RandomizedSearchCV(
                rf_model, 
                rf_param_grid, 
                n_iter=10, 
                cv=tscv, 
                n_jobs=-1, 
                scoring='neg_mean_squared_error',
                random_state=42
            )
            rf_search.fit(X_train, y_train)
            rf_best = rf_search.best_estimator_
            models.append(('rf', rf_best))
            
            # XGBoost
            try:
                import xgboost as xgb
                xgb_param_grid = {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'n_estimators': [50, 100, 200]
                }
                
                xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
                xgb_search = RandomizedSearchCV(
                    xgb_model, 
                    xgb_param_grid, 
                    n_iter=10, 
                    cv=tscv, 
                    n_jobs=-1, 
                    scoring='neg_mean_squared_error',
                    random_state=42
                )
                xgb_search.fit(X_train, y_train)
                xgb_best = xgb_search.best_estimator_
                models.append(('xgb', xgb_best))
                
                # Modellerin performansına göre ağırlık belirle
                # Negatif MSE skorları pozitife çevir
                rf_score = -rf_search.best_score_
                xgb_score = -xgb_search.best_score_
                
                # Ağırlıklandırma: daha düşük hata daha yüksek ağırlık
                total_score = rf_score + xgb_score
                rf_weight = 1 - (rf_score / total_score)
                xgb_weight = 1 - (xgb_score / total_score)
                
                weights = [rf_weight, xgb_weight]
                
            except ImportError:
                # XGBoost yoksa sadece RandomForest kullan
                weights = [1.0]
                
            # LightGBM
            try:
                import lightgbm as lgb
                lgb_param_grid = {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'num_leaves': [31, 50, 70],
                    'n_estimators': [50, 100, 200]
                }
                
                lgb_model = lgb.LGBMRegressor(random_state=42)
                lgb_search = RandomizedSearchCV(
                    lgb_model, 
                    lgb_param_grid, 
                    n_iter=10, 
                    cv=tscv, 
                    n_jobs=-1, 
                    scoring='neg_mean_squared_error',
                    random_state=42
                )
                lgb_search.fit(X_train, y_train)
                lgb_best = lgb_search.best_estimator_
                models.append(('lgb', lgb_best))
                
                # Ağırlıkları güncelle (LightGBM eklendiyse)
                if len(weights) > 0:
                    # Negatif MSE skorunu pozitife çevir
                    lgb_score = -lgb_search.best_score_
                    
                    # Yeni toplam skor hesapla
                    if 'xgb_score' in locals():
                        total_score = rf_score + xgb_score + lgb_score
                        rf_weight = 1 - (rf_score / total_score)
                        xgb_weight = 1 - (xgb_score / total_score)
                        lgb_weight = 1 - (lgb_score / total_score)
                        
                        weights = [rf_weight, xgb_weight, lgb_weight]
                    else:
                        total_score = rf_score + lgb_score
                        rf_weight = 1 - (rf_score / total_score)
                        lgb_weight = 1 - (lgb_score / total_score)
                        
                        weights = [rf_weight, lgb_weight]
                
            except ImportError:
                # LightGBM yoksa devam et
                pass
            
            # Ağırlıklı Voting Regressor oluştur
            model = VotingRegressor(estimators=models, weights=weights)
            model.fit(X_train, y_train)
            
            # Model parametrelerini kaydet
            ensemble_params = {
                'type': 'Ensemble',
                'models': [m[0] for m in models],
                'weights': weights,
                'rf_params': rf_search.best_params_ if 'rf_search' in locals() else None,
                'xgb_params': xgb_search.best_params_ if 'xgb_search' in locals() else None,
                'lgb_params': lgb_search.best_params_ if 'lgb_search' in locals() else None,
                'train_size': train_size,
                'feature_count': len(features_to_use)
            }
            model_params = ensemble_params
            
            # Optimizasyon sonuçlarını göster
            print(f"Ensemble model oluşturuldu:")
            print(f"- RandomForest en iyi parametreler: {rf_search.best_params_}")
            if len(models) > 1:
                print(f"- XGBoost en iyi parametreler: {xgb_search.best_params_ if 'xgb_search' in locals() else 'Kullanılmadı'}")
            if len(models) > 2:
                print(f"- LightGBM en iyi parametreler: {lgb_search.best_params_ if 'lgb_search' in locals() else 'Kullanılmadı'}")
            print(f"- Model ağırlıkları: {weights}")
            
        elif model_type == "Hibrit Model":
            # Hibrit model: ML + teknik göstergeler
            from sklearn.ensemble import RandomForestRegressor
            
            # RandomForest için geliştirilmiş parametreler
            rf_param_grid = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 8, 12, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            # Parametre optimizasyonu
            rf_base_model = RandomForestRegressor(random_state=42)
            grid_search = RandomizedSearchCV(
                rf_base_model, 
                rf_param_grid, 
                n_iter=15, 
                cv=tscv, 
                n_jobs=-1, 
                scoring='neg_mean_squared_error',
                random_state=42
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            
            # Model parametrelerini kaydet
            model_params = {
                'type': 'Hibrit Model',
                'base_model': 'RandomForest',
                'best_params': grid_search.best_params_,
                'train_size': train_size,
                'feature_count': len(features_to_use),
                'technical_indicators': [f for f in features_to_use if f not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            }
            
            print(f"Hibrit Model için RandomForest en iyi parametreler: {grid_search.best_params_}")
            
        else:
            # Bilinmeyen model tipi, varsayılan olarak RandomForest
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Model parametrelerini kaydet
            model_params = {
                'type': 'RandomForest (fallback)',
                'params': {'n_estimators': 100, 'random_state': 42},
                'train_size': train_size,
                'feature_count': len(features_to_use)
            }
        
        # Model performansını ölç
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Yön tahmini doğruluğu
        correct_direction = 0
        for i in range(1, len(y_test)):
            actual_change = y_test[i] - y_test[i-1]
            predicted_change = y_pred[i] - y_pred[i-1]
            
            if (actual_change >= 0 and predicted_change >= 0) or (actual_change < 0 and predicted_change < 0):
                correct_direction += 1
        
        direction_accuracy = correct_direction / (len(y_test) - 1) if len(y_test) > 1 else 0
        
        # Performans metriklerini kaydet
        performance_metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'direction_accuracy': float(direction_accuracy),
            'test_size': len(y_test)
        }
        
        # Modeli serialize et
        model_buffer = BytesIO()
        pickle.dump(model, model_buffer)
        model_bytes = model_buffer.getvalue()
        
        # Model parametrelerini performans metriklerine ekle
        performance_metrics['model_params'] = model_params
        
        # Versiyonlama için veri hazırlama tarihini ekle
        date_info = {
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data_start_date': stock_data.index[0].strftime("%Y-%m-%d") if len(stock_data) > 0 else None,
            'data_end_date': stock_data.index[-1].strftime("%Y-%m-%d") if len(stock_data) > 0 else None,
            'data_length': len(stock_data)
        }
        performance_metrics['date_info'] = date_info
        
        # Modeli veritabanına kaydet
        result = save_ml_model(
            symbol=symbol,
            model_type=model_type,
            model_data=model_bytes,
            features_used=features_to_use,
            performance_metrics=performance_metrics,
            model_version=model_version
        )
        
        if result:
            print(f"{symbol} için {model_type} modeli başarıyla eğitildi ve kaydedildi.")
            print(f"Performans Metrikleri: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}, Yön Doğruluğu={direction_accuracy:.2f}")
        
        return result
        
    except Exception as e:
        print(f"Model eğitimi ve kaydı sırasında hata: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def ml_price_prediction_with_db(symbol, stock_data, days_to_predict=30, threshold=0.03, model_type="RandomForest", use_existing_model=True, model_params=None, model_version=None):
    """
    Veritabanında kayıtlı ML modelini kullanarak fiyat tahmini yapar
    
    Args:
        symbol (str): Hisse sembolü
        stock_data (DataFrame): Hisse verileri
        days_to_predict (int): Tahmin edilecek gün sayısı
        threshold (float): Yükseliş/düşüş eşik değeri
        model_type (str): Model tipi
        use_existing_model (bool): True ise veritabanındaki modeli kullanır, False ise yeni model eğitir
        model_params (dict): Model parametreleri
        model_version (str): Kullanılacak model versiyonu. Belirtilmezse aktif versiyon kullanılır.
        
    Returns:
        dict: Tahmin sonuçları
    """
    # data.db_utils içe aktarmasını projenin ana dizinine göre düzeltiyoruz
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from data.db_utils import save_ml_model, load_ml_model, get_model_update_status
    
    try:
        # Sembolü düzenle
        symbol = symbol.upper().strip()
        model = None
        features_to_use = None
        
        # Veritabanından model yükleme veya yeni eğitme
        if use_existing_model:
            model_info = load_ml_model(symbol, model_type, version=model_version)
            
            if model_info:
                # Modeli yükle
                model_bytes = model_info['model_data']
                model = pickle.loads(model_bytes)
                features_to_use = model_info['features']
                
                # Versiyon bilgisini kontrol et
                version_str = f", Versiyon: {model_info['model_version']}" if 'model_version' in model_info else ""
                print(f"{symbol} için {model_type} modeli veritabanından yüklendi{version_str}.")
                print(f"Son güncelleme: {model_info['last_update_date']}")
                
                if 'metrics' in model_info and model_info['metrics']:
                    metrics = model_info['metrics']
                    print(f"Model performansı: R²={metrics.get('r2', 'N/A')}, Yön doğruluğu={metrics.get('direction_accuracy', 'N/A')}")
            else:
                print(f"{symbol} için {model_type} modeli bulunamadı. Yeni model eğitiliyor...")
                # Model bulunamadıysa yeni eğit
                train_result = train_and_save_model(symbol, stock_data, model_type)
                
                if train_result:
                    # Yeni eğitilen modeli yükle
                    model_info = load_ml_model(symbol, model_type)
                    model_bytes = model_info['model_data']
                    model = pickle.loads(model_bytes)
                    features_to_use = model_info['features']
        else:
            # Her zaman yeni model eğit
            train_result = train_and_save_model(symbol, stock_data, model_type, force_update=True, model_version=model_version)
            
            if train_result:
                # Yeni eğitilen modeli yükle
                model_info = load_ml_model(symbol, model_type)
                model_bytes = model_info['model_data']
                model = pickle.loads(model_bytes)
                features_to_use = model_info['features']
        
        if model is None:
            # Modeli yükleme/eğitme başarısız olursa standart tahmin
            return ml_price_prediction(symbol, stock_data, days_to_predict, threshold, model_type, model_params)
        
        # Veriyi hazırla
        df = stock_data.copy()
        
        # Teknik göstergeleri hesapla (gerekliyse)
        if not all(feature in df.columns for feature in features_to_use):
            df = calculate_indicators(df)
        
        # Eksik değerleri doldur
        df = df.ffill()
        
        # Tahminler için son veriyi al
        last_data = df[features_to_use].iloc[-1:].values
        
        # Tahmin yap
        prediction = model.predict(last_data)[0]
        
        # Son fiyat
        last_price = df['Close'].iloc[-1]
        
        # Tahmin edilen değişim
        predicted_change = (prediction - last_price) / last_price
        
        # Trend belirle
        trend = "Yükseliş" if predicted_change > threshold else ("Düşüş" if predicted_change < -threshold else "Yatay")
        
        # Tahmin güven aralığı
        confidence = 0.0
        if 'metrics' in model_info and model_info['metrics']:
            # Model metriklerine göre güven skoru
            r2 = model_info['metrics'].get('r2', 0)
            direction_accuracy = model_info['metrics'].get('direction_accuracy', 0)
            
            # R2 ve yön doğruluğunun ağırlıklı ortalaması
            confidence = (r2 * 0.4 + direction_accuracy * 0.6) * 100
            confidence = max(min(confidence, 95), 30)  # 30-95 arasında sınırla
        else:
            # Varsayılan güven skoru
            confidence = 65.0
        
        # Sonuç dictionary'si
        prediction_info = {
            'symbol': symbol,
            'last_price': last_price,
            'prediction_30d': prediction if days_to_predict >= 30 else None,
            'prediction_7d': last_price * (1 + predicted_change / 4) if days_to_predict >= 7 else None,
            'trend': trend,
            'confidence': confidence,
            'r2_score': r2,
            'predictions_df': None,
            'model_from_db': True,
            'last_update': model_info['last_update_date'] if 'last_update_date' in model_info else 'Yeni',
            'model_version': model_info.get('model_version', 'v1')
        }
        
        return prediction_info
        
    except Exception as e:
        print(f"ML veritabanı tahmini sırasında hata: {str(e)}")
        import traceback
        traceback.print_exc()
        # Hata durumunda standart tahmin metoduna dön
        return ml_price_prediction(symbol, stock_data, days_to_predict, threshold, model_type, model_params) 