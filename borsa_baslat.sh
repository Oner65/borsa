#!/bin/bash

echo "Borsa Istanbul Hisse Analiz Uygulaması başlatılıyor..."
echo

# Python venv kontrolü
if [ -d "venv" ]; then
    echo "Virtual environment bulundu, aktive ediliyor..."
    source venv/bin/activate
else
    echo "Virtual environment oluşturuluyor..."
    python3 -m venv venv
    source venv/bin/activate
    
    echo "Gerekli paketler yükleniyor..."
    pip install -r requirements.txt
fi

echo
echo "Uygulama başlatılıyor..."
echo

# Environment variables for XGBoost
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"

# Homebrew PATH'ini ayarla
eval "$(/opt/homebrew/bin/brew shellenv)" 2>/dev/null || true

# Uygulamayı başlat
streamlit run borsa.py

# Hata durumunda bekle
if [ $? -ne 0 ]; then
    echo
    echo "Hata oluştu! Çıkış yapmak için Enter'a basın..."
    read
fi
