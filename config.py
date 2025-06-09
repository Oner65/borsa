"""
Borsa Uygulaması için Yapılandırma Dosyası
Bu dosya, uygulamada kullanılan sabit değerleri ve eşik değerlerini içerir.
"""

# API Anahtarları
API_KEYS = {
    "GEMINI_API_KEY": "AIzaSyANEpZjZCV9zYtUsMJ5BBgMzkrf8yu8kM8",  # Google Gemini API anahtarı
    "OPENAI_API_KEY": "",  # OpenAI API anahtarı (kullanılmıyor)
    "ALPHA_VANTAGE_API_KEY": "",  # Alpha Vantage API anahtarı (kullanılmıyor)
}

# Tahmin süreleri ve parametreleri
FORECAST_PERIODS = {
    "1g": {"period": "1mo", "interval": "1d"},
    "1w": {"period": "3mo", "interval": "1d"},
    "1ay": {"period": "6mo", "interval": "1d"},
    "3ay": {"period": "1y", "interval": "1d"},
    "6ay": {"period": "2y", "interval": "1wk"},
    "1yıl": {"period": "5y", "interval": "1wk"}
}

# Varsayılan tahmin süresi
DEFAULT_FORECAST_PERIOD = "1w"

# Risk seviyesi eşik değerleri
RISK_THRESHOLDS = {
    "low": 1.5,   # Düşük risk eşiği - Bu değerden düşük volatilite düşük risk olarak kabul edilir
    "medium": 3.0 # Orta risk eşiği - Bu değerden yüksek volatilite yüksek risk olarak kabul edilir
                  # Bu iki değer arasında ise orta risk olarak kabul edilir
}

# Öneri hesaplama için eşik değerleri
RECOMMENDATION_THRESHOLDS = {
    "strong_buy": 0.6,   # Toplam sinyal skorunun toplam gösterge sayısına oranı bu değerden büyükse "GÜÇLÜ AL"
    "strong_sell": -0.6  # Toplam sinyal skorunun toplam gösterge sayısına oranı bu değerden küçükse "GÜÇLÜ SAT"
    # Bu değerler arasında pozitif ise "AL", negatif ise "SAT", sıfır ise "NÖTR"
}

# Teknik gösterge parametreleri
INDICATOR_PARAMS = {
    "sma_periods": [5, 10, 20, 50, 100, 200],
    "ema_periods": [5, 10, 20, 50, 100, 200],
    "rsi_period": 14,
    "macd_params": {"fast": 12, "slow": 26, "signal": 9},
    "stoch_params": {"k": 14, "d": 3, "slowing": 3},
    "williams_r_period": 14,
    "cci_period": 20,
    "atr_period": 14
}

# ML Modeli parametreleri
ML_MODEL_PARAMS = {
    "default_data_period": "5y",  # Varsayılan veri çekme süresi
    "minimum_data_days": 60,      # Minimum veri günü
    "trend_threshold": 0.03,      # %3 trend eşiği
    "default_volatility": 3.0,    # Varsayılan volatilite değeri
    "chart_history_days": 30,     # Grafik geçmiş gün sayısı
    "default_stock": "THYAO",     # Varsayılan hisse kodu
    "chart_height": 600,          # Grafik yüksekliği (pixel)
    "chart_width": 800,           # Grafik genişliği (pixel)
    "sector_chart_height": 230,   # Sektör grafiği yüksekliği
    "summary_chart_height": 130,  # Özet grafiği yüksekliği
    
    # Model hiperparametreleri
    "model_hyperparams": {
        "rf_n_estimators": 50,
        "rf_max_depth": 5,
        "xgb_n_estimators": 50,
        "xgb_max_depth": 5,
        "lgbm_n_estimators": 50,
        "lgbm_num_leaves": 31
    },
    
    # Tahmin parametreleri
    "prediction_params": {
        "volatility_factor": 0.1,
        "randomize_predictions": False
    },
    
    # Backtest parametreleri
    "backtest_params": {
        "test_size": 0.2,
        "cv_folds": 5,
        "scoring": "neg_mean_squared_error"
    },
    
    # Progress ve retry ayarları
    "max_model_attempts": 2,      # Model başarısız olduğunda kaç kez denenecek
    "supported_models": ["RandomForest", "XGBoost", "LightGBM", "Ensemble", "Hibrit Model"],
    
    # Portföy önerisi parametreleri
    "min_budget": 1000,           # Minimum yatırım bütçesi (TL)
    "max_budget": 10000000,       # Maksimum yatırım bütçesi (TL)
    "default_budget": 10000,      # Varsayılan yatırım bütçesi (TL)
    
    # News sekmesi parametreleri
    "max_news_results": [5, 10, 15, 20, 25, 30],
    "default_max_news": 10,
    "news_sentiment_threshold": {
        "positive": 0.65,
        "negative": 0.35
    },
    "news_trend_threshold": 0.05,
    "news_rolling_window": 5
}

# Tahmin süreleri
PREDICTION_PERIODS = [7, 14, 30, 60, 90]
DEFAULT_PREDICTION_PERIOD = 30

# Hata mesajları
ERROR_MESSAGES = {
    "no_data": "Veri bulunamadı. Lütfen geçerli bir hisse senedi kodu girdiğinizden emin olun.",
    "api_error": "Veri çekilirken bir hata oluştu. Lütfen internet bağlantınızı kontrol edin.",
    "analysis_error": "Analiz sırasında bir hata oluştu. Lütfen daha sonra tekrar deneyin.",
    "indicator_error": "Göstergeler hesaplanırken bir hata oluştu.",
    "chart_error": "Grafik oluşturulurken bir hata oluştu."
}

# Stock analizi için zaman pencereleri (iş günü bazında)
STOCK_ANALYSIS_WINDOWS = {
    "volatility_window": 20,     # Volatilite hesaplama penceresi
    "volume_window": 20,         # Hacim ortalaması penceresi
    "week_1": 5,                 # 1 hafta = 5 iş günü
    "month_1": 21,               # 1 ay = 21 iş günü
    "month_3": 63,               # 3 ay = 63 iş günü
    "year_1": 252,               # 1 yıl = 252 iş günü
    "week_52": 252               # 52 hafta = 252 iş günü
} 