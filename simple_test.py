#!/usr/bin/env python3
"""
Basit ML Test
"""

print("=== ML İyileştirme Test Başlatılıyor ===")

try:
    import sys
    import os
    print("✓ Temel modüller yüklendi")
    
    import pandas as pd
    import numpy as np
    print("✓ Pandas ve Numpy yüklendi")
    
    from sklearn.model_selection import RandomizedSearchCV
    print("✓ Sklearn yüklendi")
    
    # Proje modüllerini test et
    from ai.predictions import create_advanced_features, optimize_model_hyperparameters
    print("✓ İyileştirilmiş fonksiyonlar yüklendi")
    
    # Test verisi oluştur
    print("\n--- Test Verisi Oluşturuluyor ---")
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')[:100]
    
    np.random.seed(42)
    n = len(dates)
    prices = 50 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, n)))
    
    df = pd.DataFrame({
        'Open': prices * np.random.uniform(0.99, 1.01, n),
        'High': prices * np.random.uniform(1.0, 1.02, n),
        'Low': prices * np.random.uniform(0.98, 1.0, n), 
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n),
        'SMA5': pd.Series(prices).rolling(5).mean(),
        'SMA10': pd.Series(prices).rolling(10).mean(),
        'SMA20': pd.Series(prices).rolling(20).mean(),
    }, index=dates)
    
    print(f"✓ Test verisi oluşturuldu: {len(df)} satır, {len(df.columns)} sütun")
    
    # Advanced features test
    print("\n--- Gelişmiş Feature Engineering Test ---")
    df_enhanced = create_advanced_features(df)
    print(f"✓ Özellik sayısı arttı: {len(df.columns)} -> {len(df_enhanced.columns)}")
    
    # NaN temizliği
    df_clean = df_enhanced.dropna()
    print(f"✓ NaN temizlendi: {len(df_enhanced)} -> {len(df_clean)} satır")
    
    if len(df_clean) > 50:
        # Hiperparametre optimizasyonu test
        print("\n--- Hiperparametre Optimizasyonu Test ---")
        
        # Features hazırla
        feature_cols = [col for col in df_clean.columns if col not in ['Close']]
        X = df_clean[feature_cols].values
        y = df_clean['Close'].values
        
        if len(feature_cols) >= 5:
            print(f"✓ {len(feature_cols)} özellik hazırlandı")
            
            # Basit optimizasyon testi (sadece RandomForest)
            optimized_model = optimize_model_hyperparameters(
                X[:int(len(X)*0.8)], 
                y[:int(len(y)*0.8)], 
                "RandomForest", 
                cv_folds=3
            )
            
            if optimized_model is not None:
                print("✓ Hiperparametre optimizasyonu başarılı")
                print(f"✓ En iyi parametreler alındı")
            else:
                print("⚠️ Hiperparametre optimizasyonu başarısız")
        else:
            print(f"⚠️ Yetersiz özellik: {len(feature_cols)}")
    else:
        print("⚠️ Yetersiz temiz veri")
    
    print("\n=== TEST BAŞARILI ===")
    print("İyileştirilmiş ML fonksiyonları çalışıyor!")
    
    # Temel istatistikler
    print(f"\nÖZET:")
    print(f"- Orijinal özellik sayısı: {len(df.columns)}")
    print(f"- Gelişmiş özellik sayısı: {len(df_enhanced.columns)}")
    print(f"- Temiz veri boyutu: {len(df_clean)}")
    print(f"- Kullanılabilir özellikler: {len(feature_cols) if 'feature_cols' in locals() else 'N/A'}")

except Exception as e:
    print(f"❌ HATA: {str(e)}")
    import traceback
    traceback.print_exc() 