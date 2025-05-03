# Borsa İstanbul Hisse Analiz Uygulaması

Bu uygulama, Borsa İstanbul'daki hisse senetleri için kapsamlı bir teknik analiz ve yapay zeka tabanlı tahmin aracıdır.

## Özellikler

- **Teknik Analiz**: 20+ teknik gösterge (RSI, MACD, Bollinger Bantları vb.)
- **Yapay Zeka Tahminleri**: Hisse senedi fiyat tahminleri ve trend analizi
- **Makine Öğrenimi**: Gerçek verilerle eğitilmiş modeller ile fiyat tahminleri
- **Haber Analizi**: Hisselerle ilgili haberleri getirme ve duyarlılık analizi
- **BIST100 Genel Bakış**: Endeks performansı ve sektör analizleri

## Kurulum

1. Gerekli paketleri yükleyin:

```bash
pip install -r requirements.txt
```

2. Uygulamayı başlatın:

```bash
streamlit run main.py
```

## Modüller

Uygulama modüler bir yapıda tasarlanmıştır:

- **data**: Veri alma işlevleri
- **analysis**: Teknik analiz ve gösterge hesaplama
- **ai**: Yapay zeka ve makine öğrenimi tahmin modelleri
- **ui**: Kullanıcı arayüzü bileşenleri
- **utils**: Yardımcı işlevler

## Kullanım

Uygulamayı başlattıktan sonra:

1. "Hisse Analizi" sekmesinde istediğiniz hisse senedi kodunu girin
2. Zaman aralığını seçin
3. "Analiz Et" butonuna tıklayın
4. Diğer sekmeleri keşfedin: BIST100 Genel Bakış, Yapay Zeka Tahminleri, ML Tahminleri, Hisse Haberleri

## Notlar

- Uygulama yatırım tavsiyesi niteliği taşımaz
- Tahminler, geçmiş veriler ve teknik analize dayanır
- Ekonomik, politik veya şirkete özel gelişmeleri dikkate almayabilir 

## Sorun Giderme

1. "Module not found" hataları için:
   ```bash
   pip install -r requirements.txt
   ```

2. Uygulamayı başlatmak için doğru komut:
   ```bash
   streamlit run main.py
   ```

3. Windows kullanıcıları için:
   - Uygulamayı başlatmak için `borsa_baslat.bat` dosyasını çift tıklayabilirsiniz.

4. Tüm haber kaynakları ve analiz özellikleri kullanılamıyorsa:
   - API anahtarları ve bazı servisler varsayılan olarak devre dışıdır.
   - `ai/api.py` dosyasında API anahtarlarını güncelleyebilirsiniz.

5. İnternet bağlantısı olmadan bazı özellikler çalışmayabilir. 