#!/bin/bash

echo "========================================"
echo "  Kompakt Borsa Analiz Uygulaması"
echo "  BIST 100 - Hisse Analizi - ML Tahmin"
echo "========================================"
echo

# Sanal ortamı etkinleştir
if [ -f "venv/bin/activate" ]; then
    echo "Sanal ortam etkinleştiriliyor..."
    source venv/bin/activate
else
    echo "UYARI: Sanal ortam bulunamadı. Ana Python kullanılacak."
fi

echo
echo "Kompakt borsa uygulaması başlatılıyor..."
echo
echo "Tarayıcınızda http://localhost:8502 adresinde açılacak."
echo "Uygulamayı kapatmak için bu pencerede Ctrl+C tuşlarına basın."
echo

# Environment variables for XGBoost
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"

# Homebrew PATH'ini ayarla
eval "$(/opt/homebrew/bin/brew shellenv)" 2>/dev/null || true

# Streamlit uygulamasını başlat (farklı port kullan)
streamlit run compact_borsa_app.py --server.port=8502 --server.headless=false

echo
echo "Çıkmak için Enter'a basın..."
read
