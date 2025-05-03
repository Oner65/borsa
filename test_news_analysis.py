import sys
sys.path.append('.')

from ui.improved_news_tab import analyze_news
import streamlit as st

# Test URL
test_url = "https://www.bloomberght.com/borsa"

# Haber analizi yap
print("Haber analizi yapılıyor...")
results = analyze_news(test_url)

# Sonuçları görüntüle
print("\nAnaliz Sonuçları:")
print(f"Başarı: {results.get('success')}")
print(f"Başlık: {results.get('title', '')[:50]}...")
print(f"Özet: {results.get('summary', '')[:100]}...")

# Duyarlılık bilgisi
sentiment = results.get('sentiment', {})
if sentiment:
    print(f"\nDuyarlılık:")
    print(f"Etiket: {sentiment.get('label', '')}")
    print(f"Skor: {sentiment.get('score', 0)}")
    print(f"Açıklama: {sentiment.get('explanation', '')}") 