"""
Bu modül, ensemble model stratejilerini uygulayan fonksiyonları içerir.
"""

import numpy as np
import pandas as pd

def ml_ensemble_simple_average(predictions_dict):
    """
    Birden fazla makine öğrenmesi modelinin tahminlerini basit ortalama ile birleştirir.
    
    Args:
        predictions_dict (dict): Model adı ve tahmin sonuçlarını içeren sözlük
        
    Returns:
        dict: Birleştirilmiş tahmin sonuçları
    """
    if not predictions_dict or len(predictions_dict) < 2:
        return next(iter(predictions_dict.values())) if predictions_dict else None
    
    # İlk model üzerinden şablon oluştur
    first_model = next(iter(predictions_dict.values()))
    combined_prediction = first_model.copy()
    
    # Tüm modellerin tahminlerini topla
    prediction_values = {}
    for model_name, prediction in predictions_dict.items():
        # 7 günlük tahmin
        if 'prediction_7d' in prediction and prediction['prediction_7d'] is not None:
            prediction_values.setdefault('prediction_7d', []).append(prediction['prediction_7d'])
        
        # 30 günlük tahmin
        if 'prediction_30d' in prediction and prediction['prediction_30d'] is not None:
            prediction_values.setdefault('prediction_30d', []).append(prediction['prediction_30d'])
        
        # Tahmin veri çerçevesi
        if 'predictions_df' in prediction and prediction['predictions_df'] is not None:
            df = prediction['predictions_df']
            for i, row in df.iterrows():
                date_str = i.strftime('%Y-%m-%d')
                prediction_values.setdefault(f'date_{date_str}', []).append(row['Predicted Price'])
    
    # Ortalama değerleri hesapla
    for key, values in prediction_values.items():
        if key == 'prediction_7d':
            combined_prediction['prediction_7d'] = np.mean(values)
        elif key == 'prediction_30d':
            combined_prediction['prediction_30d'] = np.mean(values)
        elif key.startswith('date_'):
            date_str = key[5:]  # "date_" önekini kaldır
            date = pd.to_datetime(date_str)
            # Şablonda predictions_df yoksa oluştur
            if 'predictions_df' not in combined_prediction or combined_prediction['predictions_df'] is None:
                # İlk modelden tahmin çerçevesini kopyala
                combined_prediction['predictions_df'] = first_model.get('predictions_df').copy()
            
            # Tarih predictions_df'de varsa değeri güncelle
            if date in combined_prediction['predictions_df'].index:
                combined_prediction['predictions_df'].loc[date, 'Predicted Price'] = np.mean(values)
    
    # Trend ve güven bilgisini güncelle
    trend_votes = {}
    for model_name, prediction in predictions_dict.items():
        trend = prediction.get('trend', 'Belirsiz')
        trend_votes[trend] = trend_votes.get(trend, 0) + 1
    
    # En çok oy alan trendi seç
    combined_prediction['trend'] = max(trend_votes.items(), key=lambda x: x[1])[0]
    
    # Modellerin ortalama güven değerini kullan
    confidence_values = [pred.get('confidence', 0) for pred in predictions_dict.values() if 'confidence' in pred]
    if confidence_values:
        combined_prediction['confidence'] = np.mean(confidence_values)
    
    # R² değerini de güncelle
    r2_values = [pred.get('r2_score', 0) for pred in predictions_dict.values() if 'r2_score' in pred]
    if r2_values:
        combined_prediction['r2_score'] = np.mean(r2_values)
    
    # Meta bilgileri güncelle
    combined_prediction['model_type'] = 'Ensemble (Simple Average)'
    
    return combined_prediction

def ml_ensemble_weighted_average(predictions_dict, weights_dict=None):
    """
    Birden fazla makine öğrenmesi modelinin tahminlerini ağırlıklı ortalama ile birleştirir.
    
    Args:
        predictions_dict (dict): Model adı ve tahmin sonuçlarını içeren sözlük
        weights_dict (dict, optional): Her model için ağırlıkları içeren sözlük
                                      Belirtilmezse eşit ağırlıklar kullanılır
                                      
    Returns:
        dict: Ağırlıklı ortalama kullanılarak birleştirilmiş tahmin sonuçları
    """
    if not predictions_dict or len(predictions_dict) < 2:
        return next(iter(predictions_dict.values())) if predictions_dict else None
    
    # Ağırlıkları kontrol et ve normalleştir
    if weights_dict is None:
        # Eşit ağırlıklar kullan
        weight = 1.0 / len(predictions_dict)
        weights_dict = {model: weight for model in predictions_dict}
    else:
        # Sadece tahmin bulunan modeller için ağırlıkları normalleştir
        total_weight = sum(weights_dict.get(model, 0) for model in predictions_dict)
        if total_weight > 0:
            weights_dict = {model: weights_dict.get(model, 0) / total_weight 
                           for model in predictions_dict}
        else:
            # Toplam ağırlık 0 ise eşit ağırlıklar kullan
            weight = 1.0 / len(predictions_dict)
            weights_dict = {model: weight for model in predictions_dict}
    
    # İlk model üzerinden şablon oluştur
    first_model = next(iter(predictions_dict.values()))
    combined_prediction = first_model.copy()
    
    # Tüm modellerin ağırlıklı tahminlerini topla
    weighted_prediction_values = {}
    for model_name, prediction in predictions_dict.items():
        weight = weights_dict.get(model_name, 0)
        
        # 7 günlük tahmin
        if 'prediction_7d' in prediction and prediction['prediction_7d'] is not None:
            weighted_prediction_values.setdefault('prediction_7d', []).append((prediction['prediction_7d'], weight))
        
        # 30 günlük tahmin
        if 'prediction_30d' in prediction and prediction['prediction_30d'] is not None:
            weighted_prediction_values.setdefault('prediction_30d', []).append((prediction['prediction_30d'], weight))
        
        # Tahmin veri çerçevesi
        if 'predictions_df' in prediction and prediction['predictions_df'] is not None:
            df = prediction['predictions_df']
            for i, row in df.iterrows():
                date_str = i.strftime('%Y-%m-%d')
                weighted_prediction_values.setdefault(f'date_{date_str}', []).append((row['Predicted Price'], weight))
    
    # Ağırlıklı ortalama değerleri hesapla
    for key, values in weighted_prediction_values.items():
        if len(values) > 0:
            weighted_sum = sum(value * weight for value, weight in values)
            total_weight = sum(weight for _, weight in values)
            
            # Ağırlıklı ortalama
            weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0
            
            if key == 'prediction_7d':
                combined_prediction['prediction_7d'] = weighted_avg
            elif key == 'prediction_30d':
                combined_prediction['prediction_30d'] = weighted_avg
            elif key.startswith('date_'):
                date_str = key[5:]  # "date_" önekini kaldır
                date = pd.to_datetime(date_str)
                # Şablonda predictions_df yoksa oluştur
                if 'predictions_df' not in combined_prediction or combined_prediction['predictions_df'] is None:
                    # İlk modelden tahmin çerçevesini kopyala
                    combined_prediction['predictions_df'] = first_model.get('predictions_df').copy()
                
                # Tarih predictions_df'de varsa değeri güncelle
                if date in combined_prediction['predictions_df'].index:
                    combined_prediction['predictions_df'].loc[date, 'Predicted Price'] = weighted_avg
    
    # Trend ve güven bilgisini güncelle
    trend_weighted_votes = {}
    for model_name, prediction in predictions_dict.items():
        weight = weights_dict.get(model_name, 0)
        trend = prediction.get('trend', 'Belirsiz')
        trend_weighted_votes[trend] = trend_weighted_votes.get(trend, 0) + weight
    
    # En yüksek ağırlıklı oyu alan trendi seç
    combined_prediction['trend'] = max(trend_weighted_votes.items(), key=lambda x: x[1])[0]
    
    # Ağırlıklı ortalama güven değerini hesapla
    combined_confidence = 0
    total_confidence_weight = 0
    for model_name, prediction in predictions_dict.items():
        if 'confidence' in prediction:
            weight = weights_dict.get(model_name, 0)
            combined_confidence += prediction['confidence'] * weight
            total_confidence_weight += weight
    
    if total_confidence_weight > 0:
        combined_prediction['confidence'] = combined_confidence / total_confidence_weight
    
    # Ağırlıklı ortalama R² değerini hesapla
    combined_r2 = 0
    total_r2_weight = 0
    for model_name, prediction in predictions_dict.items():
        if 'r2_score' in prediction:
            weight = weights_dict.get(model_name, 0)
            combined_r2 += prediction['r2_score'] * weight
            total_r2_weight += weight
    
    if total_r2_weight > 0:
        combined_prediction['r2_score'] = combined_r2 / total_r2_weight
    
    # Meta bilgileri güncelle
    combined_prediction['model_type'] = 'Ensemble (Weighted Average)'
    
    return combined_prediction

def ml_ensemble_adaptive(predictions_dict, market_conditions=None):
    """
    Piyasa koşullarına göre adaptif olarak model seçimi yapar.
    
    Args:
        predictions_dict (dict): Model adı ve tahmin sonuçlarını içeren sözlük
        market_conditions (dict, optional): Piyasa koşullarını içeren sözlük
                                          - volatility: yüksek, orta, düşük
                                          - trend: yükseliş, düşüş, yatay
                                          - volume: yüksek, orta, düşük
                                          
    Returns:
        dict: Seçilen model tahmini
    """
    if not predictions_dict:
        return None
    
    if market_conditions is None:
        # Basit seçim: en yüksek güven skoruna sahip modeli seç
        best_model = max(
            predictions_dict.items(),
            key=lambda x: x[1].get('confidence', 0) * x[1].get('r2_score', 0)
        )[0]
        return predictions_dict[best_model]
    
    # Piyasa koşullarına göre model seçimi
    volatility = market_conditions.get('volatility', 'orta')
    trend = market_conditions.get('trend', 'yatay')
    volume = market_conditions.get('volume', 'orta')
    
    # Model seçim kuralları
    if volatility == 'yüksek':
        if trend == 'yükseliş':
            # Yüksek volatiliteli yükseliş trendinde XGBoost veya Ensemble tercih edilir
            if 'Ensemble' in predictions_dict:
                return predictions_dict['Ensemble']
            elif 'XGBoost' in predictions_dict:
                return predictions_dict['XGBoost']
        else:
            # Yüksek volatiliteli düşüş trendinde RandomForest daha iyi olabilir
            if 'RandomForest' in predictions_dict:
                return predictions_dict['RandomForest']
    elif volatility == 'düşük':
        # Düşük volatilitede LightGBM genellikle daha istikrarlıdır
        if 'LightGBM' in predictions_dict:
            return predictions_dict['LightGBM']
    
    # Hiçbir kural uygulanmadıysa, en yüksek güven skoruna sahip modeli seç
    best_model = max(
        predictions_dict.items(),
        key=lambda x: x[1].get('confidence', 0) * x[1].get('r2_score', 0)
    )[0]
    
    return predictions_dict[best_model] 