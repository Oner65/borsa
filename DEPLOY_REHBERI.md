# Streamlit Cloud Deploy Rehberi

## Adım 1: GitHub Repository Hazırlama

1. GitHub'da yeni bir repository oluşturun (örn: `kompakt-borsa-app`)
2. Bu klasördeki tüm dosyaları repository'ye upload edin

## Adım 2: Streamlit Cloud'a Deploy

1. [share.streamlit.io](https://share.streamlit.io) adresine gidin
2. GitHub hesabınızla giriş yapın
3. "New app" butonuna tıklayın
4. Repository'nizi seçin
5. Main file path: `streamlit_app.py`
6. "Deploy!" butonuna tıklayın

## Dosya Yapısı

```
streamlit_deploy/
├── streamlit_app.py          # Ana uygulama dosyası
├── requirements.txt          # Gerekli Python paketleri
├── README.md                 # Proje açıklaması
├── .streamlit/
│   └── config.toml          # Streamlit konfigürasyonu
├── ai/                      # AI modülleri (boş)
├── analysis/                # Analiz modülleri (boş)
├── data/                    # Veri modülleri (boş)
└── ui/                      # UI modülleri (boş)
```

## Özellikler

✅ **Çalışan Özellikler:**
- BIST 100 genel bakış
- Hisse senedi teknik analizi
- Favori hisseler sistemi
- Interaktif grafikler
- Responsive tasarım

❌ **Ana Uygulamadan Farklı Olan Kısımlar:**
- Veritabanı bağlantısı yok (session state kullanılıyor)
- Gelişmiş AI özellikler yok
- Haber entegrasyonu basitleştirilmiş
- ML tahmin özelliği placeholder

## Gerekli Paketler

- streamlit>=1.28.0
- pandas>=1.5.0
- numpy>=1.24.0
- yfinance>=0.2.20
- plotly>=5.15.0

## Deploy Sonrası Kontrol

1. Uygulama başarıyla çalışıyor mu?
2. Grafik gösterimi doğru mu?
3. Hisse analizi çalışıyor mu?
4. Favori ekleme/çıkarma çalışıyor mu?

## Sorun Giderme

**Problem:** `No module named 'xxx'` hatası
**Çözüm:** requirements.txt dosyasına eksik paketi ekleyin

**Problem:** Veri alınamıyor
**Çözüm:** yfinance API'si çalışıyor, internet bağlantısını kontrol edin

**Problem:** Grafik gözükmüyor
**Çözüm:** plotly paketi yüklü mü kontrol edin

## Notlar

- Bu versiyon ana uygulamanızı etkilemez
- Sadece temel özellikler içerir
- Eğitim/demo amaçlıdır
- Gerçek yatırım tavsiyesi değildir
