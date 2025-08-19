# 📊 Kompakt Borsa Analiz Uygulaması

Bu uygulama, ana borsa uygulamanızdan seçilen 3 temel sekmeyi içeren kompakt bir versiyondur.

## 🎯 İçerik

### 1. 📊 BIST 100 Genel Bakış
- BIST 100 endeksinin teknik analizi
- Günlük, haftalık, aylık performans verileri
- Sektör performans analizi
- En çok yükselenler ve düşenler
- Teknik gösterge sinyalleri
- Piyasa genel durumu ve öneriler

### 2. 🔍 Hisse Analizi
- Hisse senedi teknik analizi
- Hareketli ortalamalar (SMA/EMA)
- Osilatörler (RSI, MACD, Stochastic)
- Destek ve direnç seviyeleri
- Fibonacci seviyeleri
- Grafik desenleri
- Yapay zeka değerlendirmesi
- Risk analizi ve yatırım önerileri

### 3. 🧠 ML Yükseliş Tahmini
- Makine öğrenmesi tabanlı yükseliş tahminleri
- RandomForest, XGBoost, LightGBM modelleri
- Gelişmiş teknik göstergeler
- Sentiment analizi (opsiyonel)
- Makroekonomik veriler (opsiyonel)
- Model performans analizi
- Backtesting sonuçları

## 🚀 Nasıl Başlatılır

### Yöntem 1: Batch Dosyası (Önerilen)
1. `kompakt_borsa_baslat.bat` dosyasına çift tıklayın
2. Uygulama otomatik olarak tarayıcınızda açılacaktır
3. Adres: `http://localhost:8502`

### Yöntem 2: Manuel
```bash
# Terminal/Command Prompt'ta
streamlit run compact_borsa_app.py --server.port=8502
```

## ✨ Özellikler

### Kullanıcı Dostu Arayüz
- Modern ve sade tasarım
- Responsive yapı
- Kolay navigasyon
- Sidebar ile hızlı erişim

### Akıllı Özellikler
- Favori hisse yönetimi
- Son analizleri kaydetme
- Otomatik veri güncelleme
- Hata yönetimi

### Performans
- Veri cache'leme
- Hızlı yükleme
- Optimize edilmiş göstergeler

## 📋 Gereksinimler

Bu uygulama ana borsa uygulamanızın tüm bağımlılıklarını kullanır:
- Python 3.8+
- Streamlit
- Pandas, NumPy
- Plotly
- yfinance
- scikit-learn
- XGBoost, LightGBM
- Diğer gerekli kütüphaneler

## 🔧 Yapılandırma

Uygulama ana projenizdeki yapılandırma dosyalarını kullanır:
- `config.py` - Genel ayarlar
- `data/` klasörü - Veri kaynakları
- `ui/` klasörü - Arayüz bileşenleri
- `analysis/` klasörü - Analiz fonksiyonları

## 📊 Veri Kaynakları

- **Hisse Verileri**: Yahoo Finance (yfinance)
- **Teknik Göstergeler**: Kendi hesaplama algoritmaları
- **Haberler**: Yapılandırılmış haber kaynakları
- **ML Modelleri**: Yerel veritabanında saklanan eğitilmiş modeller

## ⚠️ Önemli Notlar

1. **Yatırım Tavsiyesi Değildir**: Bu uygulama sadece analiz amaçlıdır
2. **Veri Güncelliği**: Veriler gerçek zamanlı değildir
3. **Risk**: Yatırım kararlarını kendi riskinizle alın
4. **Ana Uygulama**: Bu uygulama ana borsa uygulamanızı etkilemez

## 🔄 Güncelleme

Ana borsa uygulamanızda yapılan değişiklikler otomatik olarak bu uygulamaya da yansır çünkü aynı modülleri kullanır.

## 📞 Destek

Bu uygulama ana borsa uygulamanızın bir alt kümesi olduğu için, herhangi bir sorun durumunda ana uygulamanızın dokümantasyonunu kontrol edin.

## 🎉 Kullanım İpuçları

1. **Favori Hisseler**: Sık analiz ettiğiniz hisseleri favorilere ekleyin
2. **Sekme Geçişi**: Üç sekme arasında kolayca geçiş yapabilirsiniz
3. **ML Tahminleri**: Gelişmiş ayarlarda parametreleri değiştirerek farklı sonuçlar alabilirsiniz
4. **BIST 100**: Piyasa genel durumunu takip etmek için kullanın
5. **Hisse Analizi**: Detaylı teknik analiz için ideal

---

**Başarılı analizler dileriz! 📈** 