@echo off
chcp 65001 >nul
echo 🚀 Streamlit uygulamasını GitHub'a deploy ediliyor...

REM Git durumu kontrol et
if not exist .git (
    echo 📦 Git repository başlatılıyor...
    git init
    git remote add origin https://github.com/veteroner/smartborsa.git
)

REM Değişiklikleri ekle
echo 📁 Dosyalar ekleniyor...
git add .

REM Commit mesajı
echo 💾 Commit yapılıyor...
git commit -m "Deploy: Streamlit uygulaması güncellendi %date% %time%"

REM Ana branch'e push et
echo ⬆️ GitHub'a push ediliyor...
git branch -M main
git push -u origin main

echo ✅ Deploy tamamlandı!
echo.
echo 🌐 Streamlit Cloud'da deploy etmek için:
echo 1. https://share.streamlit.io/ adresine gidin
echo 2. GitHub hesabınızla giriş yapın
echo 3. veteroner/smartborsa repository'sini seçin
echo 4. Main file: borsa.py
echo 5. Deploy'a tıklayın
echo.
echo 🔑 API anahtarlarınızı Streamlit Cloud Secrets bölümünden ekleyin:
echo    GEMINI_API_KEY = 'your_key_here'
echo    NEWS_API_KEY = 'your_key_here'

pause 