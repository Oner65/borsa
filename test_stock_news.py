import sys
sys.path.append('.')

from ui.improved_news_tab import GoogleNews, display_log_message

# Test sorgusu
query = '"Türk Hava Yolları" OR "THY" OR "Turkish Airlines" hisse OR borsa OR yatırım OR ekonomi'

# Google News araması yap
print(f"Google News üzerinde arama yapılıyor... Sorgu: {query}")
gnews = GoogleNews(lang='tr', country='TR', period='7d')
gnews.search(query)
results = gnews.result()

# Sonuçları görüntüle
print(f"\nToplam {len(results)} sonuç bulundu.")

# İlk 5 sonucu görüntüle
print("\nİlk 5 sonuç:")
for i, result in enumerate(results[:5]):
    print(f"\n{i+1}. Haber:")
    print(f"Başlık: {result.get('title', 'Başlık yok')}")
    print(f"Link: {result.get('link', '#')}")
    print(f"Kaynak: {result.get('site', 'Kaynak belirtilmemiş')}")
    print(f"Tarih: {result.get('datetime', '')}")
    print("-" * 50) 