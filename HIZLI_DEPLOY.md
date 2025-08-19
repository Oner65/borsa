# ⚡ Hızlı Deploy Rehberi

Bu rehber, kompakt borsa uygulamanızı 5 dakikada Streamlit Community Cloud'a deploy etmenizi sağlayacak.

## 🚀 Hızlı Adımlar

### 1️⃣ GitHub Repository Oluştur (2 dakika)

```bash
# Terminal'de proje klasörünüzde çalıştırın:

# Git repository başlat
git init

# Tüm dosyaları ekle
git add .

# İlk commit
git commit -m "Kompakt Borsa Uygulaması - İlk Commit"
```

**GitHub'da:**
1. https://github.com adresine gidin
2. "New repository" tıklayın
3. Repository adı: `kompakt-borsa-app`
4. **Public** seçin (önemli!)
5. "Create repository" tıklayın

**Terminal'de devam:**
```bash
# GitHub repository'yi bağla (YOUR_USERNAME'i değiştirin)
git remote add origin https://github.com/YOUR_USERNAME/kompakt-borsa-app.git

# Ana branch'ı main yap
git branch -M main

# GitHub'a yükle
git push -u origin main
```

### 2️⃣ Streamlit Cloud'a Deploy (2 dakika)

1. https://share.streamlit.io adresine gidin
2. GitHub hesabınızla giriş yapın
3. "New app" butonuna tıklayın
4. **Repository:** `YOUR_USERNAME/kompakt-borsa-app`
5. **Branch:** `main`
6. **Main file path:** `compact_borsa_app.py`
7. **App URL:** İsteğe bağlı
8. "Deploy!" butonuna tıklayın

### 3️⃣ Tamamlandı! (1 dakika)

✅ Uygulamanız `https://YOUR_APP_NAME.streamlit.app` adresinde yayında!

---

## 🔧 Olası Sorunlar ve Çözümler

### ❌ Sorun 1: "ModuleNotFoundError"
**Çözüm:** requirements.txt dosyasının proje ana klasöründe olduğundan emin olun.

### ❌ Sorun 2: "FileNotFoundError: data/stock_analysis.db"
**Çözüm:** Normal! Cloud'da veritabanı otomatik oluşturulur.

### ❌ Sorun 3: "Memory limit exceeded"
**Çözüm:** Uygulamada cache kullanın:
```python
@st.cache_data(ttl=300)
def get_data():
    return your_data
```

### ❌ Sorun 4: Repository "private"
**Çözüm:** GitHub'da repository'yi **public** yapın.

---

## 🎯 İpuçları

✅ **Deploy sonrası güncellemeler:**
```bash
git add .
git commit -m "Güncelleme"
git push
# Otomatik olarak yeniden deploy edilir!
```

✅ **API anahtarları için:**
1. Streamlit Cloud dashboard'da uygulamanızı seçin
2. Settings > Secrets
3. Formatı: 
```toml
[api_keys]
GEMINI_API_KEY = "your_key_here"
```

✅ **Hızlı test:**
- Yerel test: `streamlit run compact_borsa_app.py`
- Deploy test: URL'yi ziyaret edin

---

## 📞 Yardım Gerekirse

1. **Streamlit Logs:** Cloud dashboard'da "Manage app" > "Logs"
2. **GitHub Issues:** Repository'nizde issue oluşturun
3. **Streamlit Community:** https://discuss.streamlit.io

---

**🎉 Başarılı deploy dileriz!** 