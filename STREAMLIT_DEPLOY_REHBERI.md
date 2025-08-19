# 🚀 Streamlit Community Cloud Deploy Rehberi

Bu rehber, kompakt borsa uygulamanızı Streamlit Community Cloud'a deploy etmenizi sağlayacak.

## 📋 Gerekli Dosyalar (✅ Hazırlandı)

- ✅ `compact_borsa_app.py` - Ana uygulama
- ✅ `requirements_deploy.txt` - Python kütüphaneleri
- ✅ `.streamlit/config.toml` - Streamlit yapılandırması
- ✅ `packages.txt` - Sistem paketleri
- ✅ `.gitignore` - Git ignore dosyası
- ✅ `secrets.toml.example` - API anahtarları örneği

## 🔧 Adım 1: GitHub Repository Oluşturma

### 1.1 GitHub'da Yeni Repository Oluşturun
1. GitHub hesabınıza giriş yapın
2. "New repository" butonuna tıklayın
3. Repository adını girin: `kompakt-borsa-app`
4. Public olarak ayarlayın (Streamlit Community Cloud için gerekli)
5. "Create repository" butonuna tıklayın

### 1.2 Lokal Git Repository Başlatın
Terminal/Command Prompt'ta projenizin klasöründe:

```bash
# Git repository başlat
git init

# Dosyaları ekle
git add .

# İlk commit
git commit -m "İlk commit: Kompakt Borsa Uygulaması"

# GitHub repository'yi bağla (YOUR_USERNAME yerine kendi kullanıcı adınızı yazın)
git remote add origin https://github.com/YOUR_USERNAME/kompakt-borsa-app.git

# Ana branch'ı main olarak ayarla
git branch -M main

# GitHub'a yükle
git push -u origin main
```

## 🔧 Adım 2: Requirements Dosyasını Yeniden Adlandırın

Deploy için `requirements_deploy.txt` dosyasını `requirements.txt` olarak değiştirin:

```bash
# Windows'ta
ren requirements_deploy.txt requirements.txt

# Linux/Mac'te
mv requirements_deploy.txt requirements.txt
```

## 🔧 Adım 3: Streamlit Community Cloud'a Deploy

### 3.1 Streamlit Community Cloud'a Giriş
1. [share.streamlit.io](https://share.streamlit.io) adresine gidin
2. GitHub hesabınızla giriş yapın
3. "New app" butonuna tıklayın

### 3.2 App Yapılandırması
1. **Repository**: `YOUR_USERNAME/kompakt-borsa-app`
2. **Branch**: `main`
3. **Main file path**: `compact_borsa_app.py`
4. **App URL** (opsiyonel): Özel URL belirleyebilirsiniz

### 3.3 Deploy Butonuna Tıklayın
- "Deploy!" butonuna tıklayın
- Streamlit uygulamanızı otomatik olarak kuracak

## 🔑 Adım 4: API Anahtarlarını Ayarlama (Opsiyonel)

Eğer uygulamanızda API anahtarları kullanıyorsanız:

1. Streamlit Cloud dashboard'da uygulamanızı seçin
2. "Settings" > "Secrets" kısmına gidin  
3. Aşağıdaki formatı kullanın:

```toml
[api_keys]
GEMINI_API_KEY = "your_actual_gemini_api_key"
OPENAI_API_KEY = "your_actual_openai_api_key"
```

## 🔧 Adım 5: Config Dosyasını Güncelle (Gerekirse)

Eğer uygulamanızda yerel dosya yolları varsa, bunları Streamlit Cloud için uyarlayın:

```python
# config.py dosyasında
import os
import streamlit as st

# Yerel veya cloud ortamını algıla
if hasattr(st, 'secrets'):
    # Streamlit Cloud'da çalışıyor
    API_KEYS = {
        "GEMINI_API_KEY": st.secrets["api_keys"]["GEMINI_API_KEY"],
        "OPENAI_API_KEY": st.secrets["api_keys"]["OPENAI_API_KEY"]
    }
else:
    # Yerel ortamda çalışıyor
    API_KEYS = {
        "GEMINI_API_KEY": "your_local_key",
        "OPENAI_API_KEY": "your_local_key"
    }
```

## 🔧 Olası Sorunlar ve Çözümler

### Sorun 1: Import Hatası
```
ModuleNotFoundError: No module named 'xyz'
```
**Çözüm**: `requirements.txt` dosyasına eksik kütüphaneyi ekleyin

### Sorun 2: Dosya Yolu Hatası
```
FileNotFoundError: [Errno 2] No such file or directory
```
**Çözüm**: Dosya yollarını Streamlit Cloud için düzenleyin

### Sorun 3: Database Hatası
```
sqlite3.OperationalError: unable to open database file
```
**Çözüm**: Database dosyasını `.gitignore`'dan çıkarın veya otomatik oluşturma kodu ekleyin

### Sorun 4: Memory Hatası
```
ResourceError: This app has gone over the resource limits
```
**Çözüm**: Veri cache'leme kullanın ve gereksiz hesaplamaları azaltın

## 🎯 Performans Optimizasyonu

### Cache Kullanımı
```python
@st.cache_data(ttl=300)  # 5 dakika cache
def get_stock_data_cached(symbol):
    return get_stock_data(symbol)
```

### Memory Management
```python
# Büyük DataFrame'leri cache'le
@st.cache_data
def load_large_data():
    return pd.read_csv("large_file.csv")
```

## 🔄 Güncelleme

Uygulamanızda değişiklik yaptığınızda:

```bash
git add .
git commit -m "Güncelleme mesajı"
git push origin main
```

Streamlit otomatik olarak uygulamanızı güncelleyecek.

## 📱 Sonuç

Deploy başarılı olduğunda:
- Uygulamanız `https://YOUR_APP_NAME.streamlit.app` adresinde yayında olacak
- Otomatik SSL sertifikası ile güvenli bağlantı
- GitHub'daki her değişiklik otomatik deploy edilecek

## 🆘 Yardım

Deploy sırasında sorun yaşarsanız:
1. Streamlit Community Cloud logs'larını kontrol edin
2. GitHub repository'nizin public olduğundan emin olun
3. `requirements.txt` dosyasındaki versiyonları kontrol edin

---

**Başarılı deploy dileriz! 🚀** 