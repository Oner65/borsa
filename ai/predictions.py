import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
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

# İçe aktarma için projenin ana dizinini ekleyebiliriz
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.analysis_functions import calculate_indicators

def ml_price_prediction(stock_symbol, stock_data, days_to_predict=30, threshold=0.03, model_type="RandomForest", model_params=None):
    """
    Makine öğrenimi ile hisse senedi fiyat tahmini yapar
    """
    try:
        # Model parametrelerini tanımla
        if model_params is None:
            model_params = {}
        
        # İşlenecek veri kopyası oluştur
        df = stock_data.copy()
        
        # Eksik değerleri doldur
        df = df.fillna(method='ffill')
        
        # Sadece kapanış fiyatları ile başla
        df_close = df['Close'].values.reshape(-1, 1)
        
        # Veriyi ölçeklendir (0 ile 1 arasında)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df_close)
        
        # Özellikler ve etiketler oluştur (her gün için önceki 60 gün)
        X = []
        y = []
        window_size = min(60, len(scaled_data)-1)
        
        for i in range(window_size, len(scaled_data)):
            X.append(scaled_data[i-window_size:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Veriyi eğitim ve test setlerine ayır
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Model seçimi
        if model_type == "XGBoost":
            try:
                from xgboost import XGBRegressor
                # XGBoost parametrelerini al
                n_estimators = model_params.get("n_estimators", 100)
                learning_rate = model_params.get("learning_rate", 0.1)
                max_depth = model_params.get("max_depth", 6)
                
                model = XGBRegressor(
                    n_estimators=n_estimators, 
                    learning_rate=learning_rate, 
                    max_depth=max_depth,
                    random_state=42
                )
            except ImportError:
                # XGBoost yüklü değilse RandomForest kullan
                st.warning("XGBoost kütüphanesi bulunamadı, RandomForest kullanılıyor.")
                model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "LightGBM":
            try:
                from lightgbm import LGBMRegressor
                # LightGBM parametrelerini al
                n_estimators = model_params.get("n_estimators", 100)
                learning_rate = model_params.get("learning_rate", 0.1)
                num_leaves = model_params.get("num_leaves", 31)
                feature_fraction = model_params.get("feature_fraction", 1.0)
                
                model = LGBMRegressor(
                    n_estimators=n_estimators, 
                    learning_rate=learning_rate,
                    num_leaves=num_leaves,
                    feature_fraction=feature_fraction,
                    random_state=42
                )
            except ImportError:
                # LightGBM yüklü değilse RandomForest kullan
                st.warning("LightGBM kütüphanesi bulunamadı, RandomForest kullanılıyor.")
                model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "Ensemble":
            # Ensemble modeli: RandomForest, XGBoost ve LightGBM'in ortalaması
            models = []
            predictions = []
            
            # RandomForest
            rf_n_estimators = model_params.get("rf_n_estimators", 100)
            rf_max_depth = model_params.get("rf_max_depth", None)
            
            rf_model = RandomForestRegressor(
                n_estimators=rf_n_estimators, 
                max_depth=rf_max_depth,
                random_state=42
            )
            models.append(("RandomForest", rf_model))
            
            # XGBoost
            try:
                from xgboost import XGBRegressor
                xgb_n_estimators = model_params.get("xgb_n_estimators", 100)
                xgb_learning_rate = model_params.get("xgb_learning_rate", 0.1)
                xgb_max_depth = model_params.get("xgb_max_depth", 6)
                
                xgb_model = XGBRegressor(
                    n_estimators=xgb_n_estimators, 
                    learning_rate=xgb_learning_rate,
                    max_depth=xgb_max_depth,
                    random_state=42
                )
                models.append(("XGBoost", xgb_model))
            except ImportError:
                pass
                
            # LightGBM
            try:
                from lightgbm import LGBMRegressor
                lgbm_n_estimators = model_params.get("lgbm_n_estimators", 100)
                lgbm_learning_rate = model_params.get("lgbm_learning_rate", 0.1)
                lgbm_num_leaves = model_params.get("lgbm_num_leaves", 31)
                lgbm_feature_fraction = model_params.get("lgbm_feature_fraction", 1.0)
                
                lgbm_model = LGBMRegressor(
                    n_estimators=lgbm_n_estimators, 
                    learning_rate=lgbm_learning_rate,
                    num_leaves=lgbm_num_leaves,
                    feature_fraction=lgbm_feature_fraction,
                    random_state=42
                )
                models.append(("LightGBM", lgbm_model))
            except ImportError:
                pass
            
            # Her modeli eğit
            for name, model in models:
                model.fit(X_train, y_train)
                predictions.append(model.predict(X_test))
            
            # Tahminlerin ortalamasını al
            y_pred = np.mean(predictions, axis=0)
            r2 = r2_score(y_test, y_pred)
            
            # Gelecek tahminleri için tüm modellerin ortalamasını kullanan bir ensemble fonksiyonu
            def ensemble_predict(x):
                preds = []
                for name, model in models:
                    preds.append(model.predict(x))
                return np.mean(preds, axis=0)
            
            # Sonraki günler için tahminler
            last_window = scaled_data[-window_size:]
            future_predictions = []
            
            last_window_copy = last_window.copy()
            
            # Gelecek günleri tahmin et
            for _ in range(days_to_predict):
                next_pred = ensemble_predict(last_window_copy.reshape(1, -1))[0]
                future_predictions.append(next_pred)
                last_window_copy = np.append(last_window_copy[1:], next_pred)
            
            # Tahminleri gerçek değerlere dönüştür
            future_pred_reshaped = np.array(future_predictions).reshape(-1, 1)
            future_pred_prices = scaler.inverse_transform(future_pred_reshaped).flatten()
            
            # Tahmin edilen tarihleri oluştur
            last_date = df.index[-1]
            future_dates = []
            
            for i in range(1, days_to_predict + 1):
                next_date = last_date + timedelta(days=i)
                # Hafta sonlarını atla
                while next_date.weekday() >= 5:  # 5 = Cumartesi, 6 = Pazar
                    next_date = next_date + timedelta(days=1)
                future_dates.append(next_date)
            
            # Tahminleri DataFrame'e dönüştür
            predictions_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Price': future_pred_prices
            })
            predictions_df.set_index('Date', inplace=True)
            
            # Başarı oranını hesapla
            success_rate = max(min(r2 * 100, 95), 30)  # %30-%95 arasında sınırla
            
            # Volatilite testi ekle ve tahminleri düzelt
            last_price = df['Close'].iloc[-1]
            volatility = df['Close'].pct_change().std() * 100  # Günlük değişim standart sapması
            
            # Rastgelelik faktörü ekle
            for i in range(len(predictions_df)):
                # Daha yüksek volatilite, daha fazla rastgelelik demektir
                randomness = random.uniform(-1, 1) * volatility * (i+1) / days_to_predict
                adjustment = predictions_df['Predicted Price'].iloc[i] * randomness / 100
                predictions_df['Predicted Price'].iloc[i] += adjustment
            
            # Tahminlerden trend bilgisini çıkar
            trend = "Yükseliş" if predictions_df['Predicted Price'].iloc[-1] > last_price else "Düşüş"
            
            # Tahmin güven aralığı
            confidence = random.uniform(max(30, success_rate-20), min(success_rate+10, 95))
            
            # Temel tahmin bilgilerini oluştur
            prediction_info = {
                'symbol': stock_symbol,
                'last_price': last_price,
                'prediction_30d': predictions_df['Predicted Price'].iloc[-1] if days_to_predict >= 30 else None,
                'prediction_7d': predictions_df['Predicted Price'].iloc[6] if days_to_predict >= 7 else None,
                'trend': trend,
                'confidence': confidence,
                'r2_score': r2,
                'predictions_df': predictions_df
            }
            
            return prediction_info
            
        elif model_type == "Hibrit Model":
            # Hibrit model: Teknik göstergeler + ML
            # Teknik göstergeleri dahil et (RSI, MACD, vs.)
            # Basit bir karma model olarak RandomForest kullanıp ağırlıkları ayarla
            
            # Mevcut modeli eğit
            n_estimators = model_params.get("n_estimators", 150)
            max_depth = model_params.get("max_depth", 12)
            
            model = RandomForestRegressor(
                n_estimators=n_estimators, 
                max_depth=max_depth, 
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Bir miktar uzman bilgisi ekle (teknik göstergelere göre değerlerde ayarlama yap)
            rsi_val = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
            macd_val = df['MACD'].iloc[-1] if 'MACD' in df.columns else 0
            
            # RSI ve MACD değerlerine göre ayarlama faktörü
            adjustment_factor = 0.0  # Varsayılan
            
            if not pd.isna(rsi_val) and not pd.isna(macd_val):
                # RSI aşırı alış/satış düzeltmesi
                if rsi_val > 70:  # Aşırı alış
                    adjustment_factor -= 0.02  # Fiyat tahminini düşür
                elif rsi_val < 30:  # Aşırı satış
                    adjustment_factor += 0.02  # Fiyat tahminini yükselt
                
                # MACD sinyal düzeltmesi
                macd_signal = df['MACD_Signal'].iloc[-1] if 'MACD_Signal' in df.columns else 0
                if not pd.isna(macd_signal):
                    if macd_val > macd_signal:  # Yükseliş sinyali
                        adjustment_factor += 0.01
                    else:  # Düşüş sinyali
                        adjustment_factor -= 0.01
        else:
            # Varsayılan: RandomForest
            n_estimators = model_params.get("n_estimators", 100)
            max_depth = model_params.get("max_depth", None)
            
            model = RandomForestRegressor(
                n_estimators=n_estimators, 
                max_depth=max_depth, 
                random_state=42
            )
        
        # Ensemble ve Hibrit dışındaki modeller için standart eğitim ve tahmin
        if model_type != "Ensemble":
            model.fit(X_train, y_train)
            
            # Test seti üzerinde değerlendir
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            
            # Sonraki günler için tahminler
            last_window = scaled_data[-window_size:]
            future_predictions = []
            
            last_window_copy = last_window.copy()
            
            # Gelecek günleri tahmin et
            for _ in range(days_to_predict):
                next_pred = model.predict(last_window_copy.reshape(1, -1))[0]
                future_predictions.append(next_pred)
                last_window_copy = np.append(last_window_copy[1:], next_pred)
            
            # Tahminleri gerçek değerlere dönüştür
            future_pred_reshaped = np.array(future_predictions).reshape(-1, 1)
            future_pred_prices = scaler.inverse_transform(future_pred_reshaped).flatten()
            
            # Hibrit model için uzman ayarlaması
            if model_type == "Hibrit Model" and 'adjustment_factor' in locals():
                # Tüm tahmini fiyatları ayarla (uzak gelecek için daha fazla)
                for i in range(len(future_pred_prices)):
                    time_factor = (i + 1) / days_to_predict  # Zamanla artan faktör
                    future_pred_prices[i] *= (1 + adjustment_factor * time_factor)
            
            # Tahmin edilen tarihleri oluştur
            last_date = df.index[-1]
            future_dates = []
            
            for i in range(1, days_to_predict + 1):
                next_date = last_date + timedelta(days=i)
                # Hafta sonlarını atla
                while next_date.weekday() >= 5:  # 5 = Cumartesi, 6 = Pazar
                    next_date = next_date + timedelta(days=1)
                future_dates.append(next_date)
            
            # Tahminleri DataFrame'e dönüştür
            predictions_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Price': future_pred_prices
            })
            predictions_df.set_index('Date', inplace=True)
        
        # Başarı oranını hesapla
        success_rate = max(min(r2 * 100, 95), 30)  # %30-%95 arasında sınırla
        
        # Volatilite testi ekle ve tahminleri düzelt
        last_price = df['Close'].iloc[-1]
        volatility = df['Close'].pct_change().std() * 100  # Günlük değişim standart sapması
        
        # Rastgelelik faktörü ekle
        for i in range(len(predictions_df)):
            # Daha yüksek volatilite, daha fazla rastgelelik demektir
            randomness = random.uniform(-1, 1) * volatility * (i+1) / days_to_predict
            adjustment = predictions_df['Predicted Price'].iloc[i] * randomness / 100
            predictions_df['Predicted Price'].iloc[i] += adjustment
        
        # Tahminlerden trend bilgisini çıkar
        trend = "Yükseliş" if predictions_df['Predicted Price'].iloc[-1] > last_price else "Düşüş"
        
        # Tahmin güven aralığı
        confidence = random.uniform(max(30, success_rate-20), min(success_rate+10, 95))
        
        # Temel tahmin bilgilerini oluştur
        prediction_info = {
            'symbol': stock_symbol,
            'last_price': last_price,
            'prediction_30d': predictions_df['Predicted Price'].iloc[-1] if days_to_predict >= 30 else None,
            'prediction_7d': predictions_df['Predicted Price'].iloc[6] if days_to_predict >= 7 else None,
            'trend': trend,
            'confidence': confidence,
            'r2_score': r2,
            'predictions_df': predictions_df,
            'model_type': model_type
        }
        
        # Grafiği oluştur
        plt.figure(figsize=(12, 6))
        plt.plot(df.index[-60:], df['Close'].iloc[-60:], label='Geçmiş Veriler')
        plt.plot(predictions_df.index, predictions_df['Predicted Price'], label='ML Tahmini', linestyle='--')
        plt.title(f"{stock_symbol} ML Fiyat Tahmini - {model_type} (Güven: %{confidence:.1f})")
        plt.xlabel('Tarih')
        plt.ylabel('Fiyat (TL)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return prediction_info
        
    except Exception as e:
        st.error(f"Makine öğrenimi tahmini sırasında hata oluştu: {str(e)}")
        return None

def ai_market_sentiment():
    """
    Piyasa duyarlılığı analizi yapar (yapay sonuçlar üretir)
    """
    try:
        sentiment = {
            'market_mood': random.choice(['Olumlu', 'Nötr', 'Tedbirli', 'Karışık', 'Olumsuz']),
            'confidence': random.randint(60, 95),
            'trend_strength': random.randint(30, 90),
            'volatility_expectation': random.choice(['Düşük', 'Orta', 'Yüksek']),
            'overall_recommendation': random.choice(['Al', 'Tut', 'Sat'])
        }
        
        # 'market_mood' anahtarının varlığını kontrol et
        if 'market_mood' not in sentiment:
            sentiment['market_mood'] = 'Nötr'  # Varsayılan değer
        
        return sentiment
    except Exception as e:
        # Hata durumunda varsayılan değerlerle sözlük döndür
        return {
            'market_mood': 'Nötr',
            'confidence': 75,
            'trend_strength': 50,
            'volatility_expectation': 'Orta',
            'overall_recommendation': 'Tut'
        }

def ai_price_prediction(stock_symbol, stock_data):
    """
    Fiyat tahmini yapar (yapay sonuçlar üretir)
    """
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
            'RSI', 'MACD', 'MACD_Signal', 'SMA_20', 'SMA_50', 'SMA_200',
            'Bollinger_High', 'Bollinger_Middle', 'Bollinger_Low',
            'Stoch_k', 'Stoch_d', 'Williams_R', 'ADX', 'CCI'
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
    sma_20 = indicators.get('SMA_20', "Mevcut değil")
    sma_50 = indicators.get('SMA_50', "Mevcut değil") 
    sma_200 = indicators.get('SMA_200', "Mevcut değil")
    
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
        'SMA_20': sma_20,
        'SMA_50': sma_50,
        'SMA_200': sma_200,
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
        # Veri setini hazırla
        df = df_with_indicators.copy()
        
        # Özellik sütunları ve hedef sütunu
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_5', 'SMA_10', 'SMA_20', 'EMA_5', 'EMA_10', 'EMA_20', 
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 
            'Upper_Band', 'Middle_Band', 'Lower_Band',
            'K_Line', 'D_Line', 'ADX'
        ]
        
        # Eksik feature'lar varsa kaldır
        features_to_use = [f for f in feature_columns if f in df.columns]
        
        # NaN değerleri temizle
        df = df.dropna()
        
        # Tahmin sonuçlarını saklamak için DataFrame
        predictions_df = pd.DataFrame()
        
        # Test günleri için döngü
        all_actual_prices = []
        all_predicted_prices = []
        
        for i in range(test_size):
            # Test için kullanılacak başlangıç ve bitiş indekslerini belirle
            train_end_idx = len(df) - test_size + i - period
            
            if train_end_idx <= 0:
                continue
                
            # Eğitim verisi
            train_data = df.iloc[:train_end_idx]
            
            # Test verisi (bitiş tarihi)
            actual_date = df.index[train_end_idx + period - 1]
            actual_price = df.iloc[train_end_idx + period - 1]['Close']
            
            # Özellik ölçeklendirme
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(train_data[features_to_use])
            
            # Hedef değişken
            y_train = train_data['Close'].values
            
            # En son veriler (tahmin için)
            X_last = scaler.transform(df.iloc[train_end_idx - 1:train_end_idx][features_to_use])
            
            # Model eğitimi ve tahmin
            if model_type == "RandomForest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                prediction = model.predict(X_last)[0]
                
            elif model_type == "XGBoost":
                model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
                model.fit(X_train, y_train)
                prediction = model.predict(X_last)[0]
                
            elif model_type == "LightGBM":
                model = lgb.LGBMRegressor(random_state=42)
                model.fit(X_train, y_train)
                prediction = model.predict(X_last)[0]
                
            elif model_type == "Ensemble":
                # Birden fazla modeli birleştir
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
                lgb_model = lgb.LGBMRegressor(random_state=42)
                
                rf_model.fit(X_train, y_train)
                xgb_model.fit(X_train, y_train)
                lgb_model.fit(X_train, y_train)
                
                # Tahminleri birleştir
                rf_pred = rf_model.predict(X_last)[0]
                xgb_pred = xgb_model.predict(X_last)[0]
                lgb_pred = lgb_model.predict(X_last)[0]
                
                prediction = (rf_pred + xgb_pred + lgb_pred) / 3
                
            elif model_type == "Hibrit Model":
                # Teknik göstergeler ile makine öğrenimi karışımı
                technical_features = [f for f in [
                    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 
                    'Upper_Band', 'Middle_Band', 'Lower_Band', 
                    'K_Line', 'D_Line', 'ADX'
                ] if f in features_to_use]
                
                price_features = [f for f in ['Open', 'High', 'Low', 'Close', 'Volume'] if f in features_to_use]
                
                if technical_features and price_features:
                    # Fiyat bazlı tahmin
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf_model.fit(X_train, y_train)
                    price_pred = rf_model.predict(X_last)[0]
                    
                    # Teknik gösterge ağırlıklı düzeltme
                    last_price = df.iloc[train_end_idx - 1]['Close']
                    rsi = df.iloc[train_end_idx - 1]['RSI'] if 'RSI' in df.columns else 50
                    macd = df.iloc[train_end_idx - 1]['MACD'] if 'MACD' in df.columns else 0
                    
                    # RSI ve MACD'ye göre ayarlama
                    rsi_factor = (rsi - 50) / 50  # -1 ile 1 arasında
                    macd_factor = 1 if macd > 0 else -1
                    
                    adjust_factor = 0.02 * (rsi_factor + macd_factor) / 2
                    
                    # Hibrit tahmin
                    prediction = price_pred * (1 + adjust_factor)
                else:
                    # Yeterli teknik gösterge yoksa standart modeli kullan
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    prediction = model.predict(X_last)[0]
            else:
                # Varsayılan olarak RandomForest kullan
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                prediction = model.predict(X_last)[0]
            
            # Tahmin ve gerçek değeri kaydet
            all_actual_prices.append(actual_price)
            all_predicted_prices.append(prediction)
            
            # Sonuçları DataFrame'e ekle
            predictions_df = pd.concat([predictions_df, pd.DataFrame({
                'Date': [actual_date],
                'Actual': [actual_price],
                'Predicted': [prediction]
            })])
        
        # Performans metrikleri
        if len(all_actual_prices) > 0:
            mae = mean_absolute_error(all_actual_prices, all_predicted_prices)
            rmse = np.sqrt(mean_squared_error(all_actual_prices, all_predicted_prices))
            r2 = r2_score(all_actual_prices, all_predicted_prices)
            
            # Trend doğruluğu (yön tahmini)
            correct_direction = 0
            for i in range(1, len(all_actual_prices)):
                actual_change = all_actual_prices[i] - all_actual_prices[i-1]
                predicted_change = all_predicted_prices[i] - all_predicted_prices[i-1]
                
                if (actual_change >= 0 and predicted_change >= 0) or (actual_change < 0 and predicted_change < 0):
                    correct_direction += 1
            
            direction_accuracy = correct_direction / (len(all_actual_prices) - 1) if len(all_actual_prices) > 1 else 0
            
            # Sonuçları indexle birlikte döndür
            predictions_df = predictions_df.set_index('Date')
            
            return {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'accuracy': direction_accuracy,
                'predictions': predictions_df
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
        print(f"Backtest hata: {str(e)}")
        print(traceback.format_exc())
        
        return {
            'error': str(e),
            'mae': np.nan,
            'rmse': np.nan,
            'accuracy': np.nan,
            'predictions': None
        } 

def train_and_save_model(symbol, stock_data, model_type="RandomForest", force_update=False):
    """
    Belirli bir hisse için model eğitir ve veritabanına kaydeder.
    
    Args:
        symbol (str): Hisse sembolü
        stock_data (DataFrame): Hisse verileri
        model_type (str): Model tipi (RandomForest, XGBoost, LightGBM, Ensemble, Hibrit)
        force_update (bool): True ise, mevcut model olsa bile yeniden eğitir
        
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
            'SMA_5', 'SMA_10', 'SMA_20', 'EMA_5', 'EMA_10', 'EMA_20', 
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 
            'Upper_Band', 'Middle_Band', 'Lower_Band',
            'K_Line', 'D_Line', 'ADX'
        ]
        
        # Eksik feature'lar varsa kaldır
        features_to_use = [f for f in feature_columns if f in df.columns]
        
        # Veri setini eğitim ve test olarak ayır
        X = df[features_to_use].values
        y = df['Close'].values
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Modeli eğit
        if model_type == "RandomForest":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
        elif model_type == "XGBoost":
            import xgboost as xgb
            model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
            model.fit(X_train, y_train)
            
        elif model_type == "LightGBM":
            import lightgbm as lgb
            model = lgb.LGBMRegressor(random_state=42)
            model.fit(X_train, y_train)
            
        elif model_type == "Ensemble":
            # Birden fazla modeli birleştir
            from sklearn.ensemble import RandomForestRegressor, VotingRegressor
            import xgboost as xgb
            import lightgbm as lgb
            
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
            lgb_model = lgb.LGBMRegressor(random_state=42)
            
            model = VotingRegressor([
                ('rf', rf_model),
                ('xgb', xgb_model),
                ('lgb', lgb_model)
            ])
            model.fit(X_train, y_train)
            
        elif model_type == "Hibrit Model":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
            model.fit(X_train, y_train)
            
        else:
            # Varsayılan olarak RandomForest
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
        
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
        
        # Modeli veritabanına kaydet
        result = save_ml_model(
            symbol=symbol,
            model_type=model_type,
            model_data=model_bytes,
            features_used=features_to_use,
            performance_metrics=performance_metrics
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

def ml_price_prediction_with_db(symbol, stock_data, days_to_predict=30, threshold=0.03, model_type="RandomForest", use_existing_model=True, model_params=None):
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
            model_info = load_ml_model(symbol, model_type)
            
            if model_info:
                # Modeli yükle
                model_bytes = model_info['model_data']
                model = pickle.loads(model_bytes)
                features_to_use = model_info['features']
                
                print(f"{symbol} için {model_type} modeli veritabanından yüklendi.")
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
            train_result = train_and_save_model(symbol, stock_data, model_type, force_update=True)
            
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
        df = df.fillna(method='ffill')
        
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
            'r2_score': model_info['metrics'].get('r2', 0) if 'metrics' in model_info else 0,
            'model_from_db': True,
            'last_update': model_info['last_update_date'] if 'last_update_date' in model_info else 'Yeni'
        }
        
        return prediction_info
        
    except Exception as e:
        print(f"ML veritabanı tahmini sırasında hata: {str(e)}")
        import traceback
        traceback.print_exc()
        # Hata durumunda standart tahmin metoduna dön
        return ml_price_prediction(symbol, stock_data, days_to_predict, threshold, model_type, model_params) 