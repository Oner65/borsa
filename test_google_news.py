import sys
sys.path.append('.')

from ui.improved_news_tab import GoogleNews, get_stock_news, display_log_message
import streamlit as st

# Test sorgusu
symbol = "THYAO"

# Haberleri getir
print(f"{symbol} için haberleri getiriyorum...")
news_list = get_stock_news(symbol, max_results=10)

# Sonuçları görüntüle
print(f"\nToplam {len(news_list)} haber sonucu bulundu.")

# İlk 5 sonucu görüntüle
print("\nSonuçlar:")
for i, news in enumerate(news_list):
    print(f"\n{i+1}. Haber:")
    print(f"Başlık: {news.get('title', 'Başlık yok')}")
    print(f"URL: {news.get('url', '#')}")
    print(f"Kaynak: {news.get('source', 'Kaynak belirtilmemiş')}")
    print(f"Tarih: {news.get('date', '')}")
    print(f"Özet: {news.get('summary', 'Özet yok')[:50]}...")
    print(f"Duyarlılık: {news.get('sentiment', 0.5)}")
    print("-" * 50) 