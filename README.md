# Kompakt Borsa Analiz Uygulaması

Bu uygulama BIST 100 borsa analizi, hisse senedi teknik analizi ve makine öğrenmesi tabanlı tahminler sunar.

## 🚀 Özellikler

- 📊 **BIST100 Genel Bakış**: Piyasa özeti ve performans görünümü
- 🔍 **Hisse Analizi**: Teknik analiz ve grafik görünümü
- 🔎 **ML Tarama**: Makine öğrenmesi tabanlı tahmin ve öneriler

## 🛠️ Teknolojiler

- **Streamlit**: Web uygulaması framework'ü
- **yFinance**: Finansal veri API'si
- **Plotly**: İnteraktif grafik kütüphanesi
- **Scikit-learn, XGBoost, LightGBM**: Makine öğrenmesi modelleri
- **Transformers**: Duygu analizi için AI modelleri

## 📱 Demo

Uygulamayı [Streamlit Cloud](https://share.streamlit.io/) üzerinde deneyebilirsiniz.

## ⚡ Hızlı Başlangıç

```bash
# Repository'yi klonlayın
git clone https://github.com/Oner65/borsa.git
cd borsa

# Bağımlılıkları yükleyin
pip install -r requirements_streamlit.txt

# Uygulamayı çalıştırın
streamlit run compact_borsa_app.py
```

## 📁 Proje Yapısı

```
├── compact_borsa_app.py    # Ana uygulama dosyası
├── ai/                     # AI ve ML modülleri
├── analysis/              # Teknik analiz modülleri
├── data/                  # Veri işleme modülleri
├── ui/                    # UI bileşenleri
└── utils/                 # Yardımcı araçlar
```

## 🔗 API'ler

- **Yahoo Finance**: Gerçek zamanlı hisse senedi verileri
- **Google News**: Finansal haberler
- **Transformers**: Duygu analizi

## 📊 Performans

- **Gerçek zamanlı veri**: Anlık piyasa verileri
- **Hızlı yükleme**: Optimize edilmiş veri işleme
- **Responsive tasarım**: Mobil uyumlu arayüz

## 🤝 Katkıda Bulunma

1. Repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 📞 İletişim

Proje sahibi: [@Oner65](https://github.com/Oner65)

Proje Linki: [https://github.com/Oner65/borsa](https://github.com/Oner65/borsa)

## Kurulum

1. **Gerekli paketleri yükleyin:**

```bash
pip install -r requirements.txt
```

2. **Uygulamayı başlatın:**

```bash
python borsa.py
```

veya

```bash
streamlit run borsa.py
```

3. **Windows kullanıcıları için:**
```bash
borsa_baslat.bat
```

## Modüller

Uygulama modüler bir yapıda tasarlanmıştır:

- **data**: Veri alma işlevleri ve veritabanı yönetimi
- **analysis**: Teknik analiz ve gösterge hesaplama
- **ai**: Yapay zeka ve makine öğrenimi tahmin modelleri
- **ui**: Kullanıcı arayüzü bileşenleri
- **utils**: Yardımcı işlevler ve veritabanı utilities

## Kullanım

### 🔍 Hisse Analizi
1. "Hisse Analizi" sekmesinde istediğiniz hisse senedi kodunu girin
2. Zaman aralığını seçin (1 hafta - 5 yıl)
3. "Analiz Et" butonuna tıklayın
4. Teknik göstergeleri ve sinyalleri inceleyin

### 🚀 Gelişmiş Hisse Tarayıcısı
1. "Gelişmiş Tarayıcı" sekmesine gidin
2. Tarama kapsamını seçin (BIST 30/50/100 veya Tüm Hisseler)
3. Teknik göstergeleri ve filtreleri ayarlayın
4. "Taramayı Başlat" butonuna tıklayın
5. Sonuçları inceleyin ve detaylı analizlere geçin

### 📈 ML Tahminleri
1. "ML Tahminleri" sekmesinde hisse seçin
2. Model parametrelerini ayarlayın
3. Tahmin tipini seçin (fiyat/yön)
4. "Tahmini Başlat" butonuna tıklayın

### 📊 Teknik Tarama
1. "Teknik Tarama" sekmesinde hisse listesini seçin
2. Kullanılacak göstergeleri işaretleyin
3. "Hisseleri Tara" butonuna tıklayın
4. Sinyal verenleri inceleyin

## Önemli Notlar

- ⚠️ **Uygulama yatırım tavsiyesi niteliği taşımaz**
- 📊 Tahminler, geçmiş veriler ve teknik analize dayanır
- 🌐 Bazı özellikler internet bağlantısı gerektirir
- 🔄 Veriler gerçek zamanlı olarak güncellenir
- 💾 Analiz sonuçları veritabanında saklanır

## Teknik Gereksinimler

- **Python**: 3.8+
- **RAM**: Minimum 4GB (8GB önerilir)
- **Disk**: 1GB boş alan
- **İnternet**: Veri çekimi için gerekli

## Desteklenen Göstergeler

### Trend Göstergeleri
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- ADX (Average Directional Index)
- MACD (Moving Average Convergence Divergence)

### Osilatörler
- RSI (Relative Strength Index)
- Stochastic Oscillator
- Williams %R
- CCI (Commodity Channel Index)
- Ultimate Oscillator
- Stochastic RSI

### Volatilite & Diğer
- Bollinger Bands
- ATR (Average True Range)
- Volume Analysis
- Bull/Bear Power
- ROC (Rate of Change)

## Sorun Giderme

### Sık Karşılaşılan Hatalar

1. **"Module not found" hataları:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Streamlit başlatma sorunu:**
   ```bash
   streamlit --version
   streamlit run borsa.py
   ```

3. **Veri çekme hataları:**
   - İnternet bağlantınızı kontrol edin
   - Firewall ayarlarını kontrol edin
   - Birkaç dakika sonra tekrar deneyin

4. **Performans sorunları:**
   - Tarama kapsamını azaltın (BIST 30 ile başlayın)
   - Tarayıcı sekmelerini kapatın
   - Bilgisayarınızı yeniden başlatın

5. **API hataları:**
   - `config.py` dosyasında API anahtarlarını kontrol edin
   - Rate limit hatalarında bekleme süresi uygulayın

### İletişim & Destek

Sorularınız için:
- GitHub Issues bölümünü kullanın
- Hata raporları için detaylı bilgi sağlayın
- Özellik istekleri memnuniyetle karşılanır

## Güncelleme Notları

**v2.0 - Gelişmiş Tarama Özelliği**
- Tüm BIST hisselerini tarama
- Gelişmiş filtreleme seçenekleri
- Skorlama sistemi
- Detaylı analiz görünümü
- Performans iyileştirmeleri

**v1.0 - İlk Sürüm**
- Temel teknik analiz
- ML tahminleri
- BIST100 genel bakış
- Haber analizi 