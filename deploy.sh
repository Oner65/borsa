#!/bin/bash

# Streamlit uygulamasını GitHub'a deploy etme scripti
# Bu scripti çalıştırmadan önce git konfigürasyonunuzun yapıldığından emin olun

echo "🚀 Streamlit uygulamasını GitHub'a deploy ediliyor..."

# Git durumu kontrol et
if [ ! -d .git ]; then
    echo "📦 Git repository başlatılıyor..."
    git init
    git remote add origin https://github.com/veteroner/smartborsa.git
fi

# Değişiklikleri ekle
echo "📁 Dosyalar ekleniyor..."
git add .

# Commit mesajı
echo "💾 Commit yapılıyor..."
git commit -m "Deploy: Streamlit uygulaması güncellendi $(date '+%Y-%m-%d %H:%M:%S')"

# Ana branch'e push et
echo "⬆️ GitHub'a push ediliyor..."
git branch -M main
git push -u origin main

echo "✅ Deploy tamamlandı!"
echo ""
echo "🌐 Streamlit Cloud'da deploy etmek için:"
echo "1. https://share.streamlit.io/ adresine gidin"
echo "2. GitHub hesabınızla giriş yapın"
echo "3. veteroner/smartborsa repository'sini seçin"
echo "4. Main file: borsa.py"
echo "5. Deploy'a tıklayın"
echo ""
echo "🔑 API anahtarlarınızı Streamlit Cloud Secrets bölümünden ekleyin:"
echo "   GEMINI_API_KEY = 'your_key_here'"
echo "   NEWS_API_KEY = 'your_key_here'" 