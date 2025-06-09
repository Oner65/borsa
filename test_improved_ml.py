#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
İyileştirilmiş ML Prediction Test Scripti
Bu script, yeni geliştirilen ML prediction fonksiyonlarını test eder.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Proje path'ini ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.stock_data import get_stock_data
from ai.predictions import ml_price_prediction
from analysis.indicators import calculate_indicators

def create_sample_data():
    """
    Test için örnek veri oluştur
    """
    # 1 yıllık örnek veri
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    dates = [d for d in dates if d.weekday() < 5]  # Sadece iş günleri
    
    # Rastgele ama gerçekçi fiyat verisi
    np.random.seed(42)
    n = len(dates)
    
    # Base price around 50 TL
    base_price = 50
    returns = np.random.normal(0.001, 0.02, n)  # Günlük %0.1 ortalama, %2 volatilite
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # OHLCV data
    data = {
        'Open': [],
        'High': [],
        'Low': [],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n)
    }
    
    # Open, High, Low hesapla
    for i, close in enumerate(prices):
        open_price = close * np.random.uniform(0.99, 1.01)
        high_price = max(open_price, close) * np.random.uniform(1.0, 1.02)
        low_price = min(open_price, close) * np.random.uniform(0.98, 1.0)
        
        data['Open'].append(open_price)
        data['High'].append(high_price)
        data['Low'].append(low_price)
    
    df = pd.DataFrame(data, index=dates)
    return df

def test_ml_improvements():
    """
    İyileştirilmiş ML fonksiyonlarını test et
    """
    print("=" * 60)
    print("İYİLEŞTİRİLMİŞ ML PREDİCTİON TEST")
    print("=" * 60)
    
    # Test verisini oluştur
    print("1. Test verisi oluşturuluyor...")
    try:
        # Gerçek veri çekmeyi dene
        df = get_stock_data("THYAO", period="1y")
        data_source = "Gerçek veri (THYAO)"
    except:
        # Gerçek veri çekilemezse sample data kullan
        df = create_sample_data()
        data_source = "Örnek test verisi"
    
    print(f"   ✓ {data_source} yüklendi: {len(df)} gün")
    
    # Teknik göstergeleri hesapla
    print("2. Teknik göstergeler hesaplanıyor...")
    df_with_indicators = calculate_indicators(df)
    print(f"   ✓ Toplam {len(df_with_indicators.columns)} özellik hazırlandı")
    
    # Test parametreleri
    test_configs = [
        {
            'model_type': 'RandomForest',
            'prediction_params': {'use_walk_forward_validation': True, 'add_volatility': False}
        },
        {
            'model_type': 'XGBoost',
            'prediction_params': {'use_walk_forward_validation': True, 'add_volatility': False}
        },
        {
            'model_type': 'LightGBM',
            'prediction_params': {'use_walk_forward_validation': False, 'add_volatility': True}
        }
    ]
    
    print("\n3. ML modellerini test ediliyor...")
    print("-" * 60)
    
    results = []
    
    for i, config in enumerate(test_configs, 1):
        model_type = config['model_type']
        prediction_params = config['prediction_params']
        
        print(f"\n{i}. {model_type} Modeli Test Ediliyor...")
        print(f"   Walk-forward: {prediction_params.get('use_walk_forward_validation', False)}")
        print(f"   Volatilite ekleme: {prediction_params.get('add_volatility', False)}")
        
        try:
            # ML tahmin yap
            result = ml_price_prediction(
                stock_symbol="TEST",
                stock_data=df_with_indicators,
                days_to_predict=30,
                threshold=0.03,
                model_type=model_type,
                model_params=None,
                prediction_params=prediction_params
            )
            
            if result:
                results.append({
                    'model': model_type,
                    'r2_score': result.get('r2_score', 0),
                    'confidence': result.get('confidence', 0),
                    'mae': result.get('mae', 0),
                    'rmse': result.get('rmse', 0),
                    'features_count': result.get('features_count', 0),
                    'data_quality': result.get('data_quality_score', 0),
                    'walk_forward_scores': result.get('walk_forward_scores'),
                    'success': True
                })
                
                print(f"   ✓ Başarılı!")
                print(f"     R² skoru: {result.get('r2_score', 0):.4f}")
                print(f"     Güven: %{result.get('confidence', 0):.1f}")
                print(f"     MAE: {result.get('mae', 0):.2f}")
                print(f"     RMSE: {result.get('rmse', 0):.2f}")
                print(f"     Özellik sayısı: {result.get('features_count', 0)}")
                print(f"     Veri kalitesi: {result.get('data_quality_score', 0):.3f}")
                
                # Walk-forward scores varsa göster
                wf_scores = result.get('walk_forward_scores')
                if wf_scores:
                    print(f"     Walk-forward R²: {wf_scores.get('r2', 0):.4f}")
                    print(f"     Walk-forward yön doğruluğu: {wf_scores.get('direction_accuracy', 0):.3f}")
                
            else:
                results.append({
                    'model': model_type,
                    'success': False,
                    'error': 'Tahmin başarısız'
                })
                print(f"   ✗ Başarısız!")
                
        except Exception as e:
            results.append({
                'model': model_type,
                'success': False,
                'error': str(e)
            })
            print(f"   ✗ Hata: {str(e)}")
    
    # Sonuçları özetle
    print("\n" + "=" * 60)
    print("TEST SONUÇLARI ÖZETİ")
    print("=" * 60)
    
    successful_tests = [r for r in results if r.get('success', False)]
    failed_tests = [r for r in results if not r.get('success', False)]
    
    print(f"Başarılı testler: {len(successful_tests)}/{len(results)}")
    print(f"Başarısız testler: {len(failed_tests)}")
    
    if successful_tests:
        print("\nBAŞARILI MODELLER:")
        print("-" * 40)
        
        # R² skoruna göre sırala
        successful_tests.sort(key=lambda x: x.get('r2_score', 0), reverse=True)
        
        for i, result in enumerate(successful_tests, 1):
            print(f"{i}. {result['model']}")
            print(f"   R² Skoru: {result.get('r2_score', 0):.4f}")
            print(f"   Güven: %{result.get('confidence', 0):.1f}")
            print(f"   MAE: {result.get('mae', 0):.2f}")
            print(f"   Özellik Sayısı: {result.get('features_count', 0)}")
            print()
        
        # En iyi modeli belirle
        best_model = successful_tests[0]
        print(f"🏆 EN İYİ MODEL: {best_model['model']}")
        print(f"   R² Skoru: {best_model.get('r2_score', 0):.4f}")
        print(f"   Güven Skoru: %{best_model.get('confidence', 0):.1f}")
        
        # İyileştirme analizi
        print("\n" + "=" * 60)
        print("İYİLEŞTİRME ANALİZİ")
        print("=" * 60)
        
        # R² skorları kontrol et
        avg_r2 = np.mean([r.get('r2_score', 0) for r in successful_tests])
        positive_r2_count = len([r for r in successful_tests if r.get('r2_score', 0) > 0])
        
        print(f"Ortalama R² skoru: {avg_r2:.4f}")
        print(f"Pozitif R² skoru olan modeller: {positive_r2_count}/{len(successful_tests)}")
        
        if avg_r2 > 0:
            print("✅ İYİLEŞTİRME BAŞARILI: Negatif R² problemi çözüldü!")
        else:
            print("⚠️  İYİLEŞTİRME KISMEN BAŞARILI: Daha fazla optimizasyon gerekebilir.")
            
        # Güven skorları analizi
        avg_confidence = np.mean([r.get('confidence', 0) for r in successful_tests])
        high_confidence_count = len([r for r in successful_tests if r.get('confidence', 0) > 70])
        
        print(f"Ortalama güven skoru: %{avg_confidence:.1f}")
        print(f"Yüksek güven (>%70) olan modeller: {high_confidence_count}/{len(successful_tests)}")
        
        # Feature engineering analizi
        avg_features = np.mean([r.get('features_count', 0) for r in successful_tests])
        print(f"Ortalama özellik sayısı: {avg_features:.0f}")
        
        if avg_features > 15:
            print("✅ Gelişmiş feature engineering başarılı!")
        else:
            print("⚠️  Feature engineering daha da geliştirilebilir.")
            
    if failed_tests:
        print(f"\nBAŞARISIZ TESTLER:")
        print("-" * 40)
        for result in failed_tests:
            print(f"❌ {result['model']}: {result.get('error', 'Bilinmeyen hata')}")
    
    print("\n" + "=" * 60)
    print("TEST TAMAMLANDI")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    # Performans ölçümü
    start_time = datetime.now()
    
    try:
        results = test_ml_improvements()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\nToplam test süresi: {duration:.1f} saniye")
        
    except Exception as e:
        print(f"❌ Test sırasında kritik hata: {str(e)}")
        import traceback
        traceback.print_exc() 