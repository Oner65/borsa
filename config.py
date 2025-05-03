"""
Borsa Uygulaması için Yapılandırma Dosyası
Bu dosya, uygulamada kullanılan sabit değerleri ve eşik değerlerini içerir.
"""

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

# Hata mesajları
ERROR_MESSAGES = {
    "no_data": "Veri bulunamadı. Lütfen geçerli bir hisse senedi kodu girdiğinizden emin olun.",
    "api_error": "Veri çekilirken bir hata oluştu. Lütfen internet bağlantınızı kontrol edin.",
    "analysis_error": "Analiz sırasında bir hata oluştu. Lütfen daha sonra tekrar deneyin.",
    "indicator_error": "Göstergeler hesaplanırken bir hata oluştu.",
    "chart_error": "Grafik oluşturulurken bir hata oluştu."
} 