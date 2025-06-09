# 📈 SmartBorsa - Akıllı Borsa Analiz Sistemi

**Borsa İstanbul (BIST) hisse senetleri için geliştirilmiş AI destekli analiz ve tahmin platformu**

## 🚀 Özellikler

- **Makine Öğrenmesi Tabanlı Tahminler**: RandomForest, XGBoost, LightGBM modelleri
- **Teknik Analiz**: 25+ teknik gösterge ve pattern tanıma
- **Sentiment Analizi**: Haber ve sosyal medya duyarlılık analizi
- **Portföy Yönetimi**: Kişiselleştirilmiş portföy takibi ve optimizasyonu
- **Gerçek Zamanlı Veriler**: Canlı hisse fiyatları ve haberler
- **AI Powered Insights**: Gemini AI ile akıllı yorum ve öneriler

## 🛠️ Teknolojiler

- **Backend**: Python, Streamlit
- **ML/AI**: scikit-learn, XGBoost, LightGBM, Google Gemini
- **Data**: yfinance, BeautifulSoup, NewsAPI
- **Database**: SQLite
- **Visualization**: Plotly, Matplotlib

## 📦 Kurulum

### Gereksinimler
```bash
pip install -r requirements.txt
```

### Çalıştırma
```bash
streamlit run borsa.py
```

## 🌐 Deploy Etme

### Streamlit Cloud (Önerilen)

1. **GitHub'a Push Edin:**
   ```bash
   # Linux/Mac
   chmod +x deploy.sh
   ./deploy.sh
   
   # Windows
   deploy.bat
   ```

2. **Streamlit Cloud'da Deploy Edin:**
   - [https://share.streamlit.io/](https://share.streamlit.io/) adresine gidin
   - GitHub hesabınızla giriş yapın
   - `veteroner/smartborsa` repository'sini seçin
   - Main file: `borsa.py`
   - Deploy'a tıklayın

3. **API Anahtarlarını Ekleyin:**
   - Streamlit Cloud Dashboard > Secrets
   ```toml
   GEMINI_API_KEY = "your_gemini_api_key"
   NEWS_API_KEY = "your_news_api_key"
   ```

### Alternatif Deploy Seçenekleri

#### Heroku
```bash
# Procfile oluşturun
echo "web: streamlit run borsa.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Deploy edin
git push heroku main
```

#### Railway
```bash
railway deploy
```

#### Render
- GitHub repository'nizi connect edin
- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run borsa.py --server.port=10000 --server.address=0.0.0.0`

## 🔧 Konfigürasyon

### API Anahtarları
- **Gemini AI**: [Google AI Studio](https://makersuite.google.com/app/apikey)
- **News API**: [NewsAPI.org](https://newsapi.org/)

### Veritabanı
- SQLite (varsayılan) veya PostgreSQL

## 📁 Proje Yapısı

```
smartborsa/
├── borsa.py                 # Ana uygulama
├── ui/                      # Kullanıcı arayüzü
├── ai/                      # ML modelleri ve AI
├── data/                    # Veri yönetimi
├── analysis/                # Teknik analiz
├── utils/                   # Yardımcı fonksiyonlar
├── .streamlit/config.toml   # Streamlit config
├── requirements.txt         # Python bağımlılıkları
└── README.md               # Bu dosya
```

## 🎯 Kullanım

1. **Hisse Analizi**: Teknik göstergeler ve AI yorumları
2. **ML Tahminleri**: Fiyat ve yön tahminleri
3. **Portföy Takibi**: Kar/zarar ve performans analizi
4. **Haber Takibi**: Sentiment analizi ile haber değerlendirme
5. **Hisse Profilleri**: Kişiselleştirilmiş gösterge analizi

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit yapın (`git commit -m 'Add some AmazingFeature'`)
4. Push edin (`git push origin feature/AmazingFeature`)
5. Pull Request açın

## 📄 Lisans

Bu proje MIT lisansı altında yayınlanmıştır.

## 📞 İletişim

- GitHub: [@veteroner](https://github.com/veteroner)
- Repository: [smartborsa](https://github.com/veteroner/smartborsa)

## ⚠️ Uyarı

Bu uygulama sadece eğitim ve araştırma amaçlıdır. Finansal kararlarınızı almadan önce profesyonel yatırım danışmanınıza başvurun.

---

**⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın!** 