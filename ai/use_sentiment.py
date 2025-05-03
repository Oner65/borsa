"""
Eğitilmiş duyarlılık analizi modelinin nasıl kullanılacağını gösteren örnekler
"""
import os
import sys

# Proje dizinini path'e ekle
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

from ai.sentiment_analysis import SentimentAnalyzer

def demonstrate_sentiment_analysis():
    """
    Eğitilmiş duyarlılık analizi modelinin kullanımını gösterir.
    """
    print("Duyarlılık Analizi Örneği")
    print("=" * 40)
    
    # Model nesnesini oluştur (önceden eğitilmiş model yüklenecek)
    analyzer = SentimentAnalyzer()
    
    # Test edilecek finansal haberler
    test_news = [
        "Borsa İstanbul tüm zamanların rekorunu kırdı",
        "BIST 100 endeksi 8500 puanı aştı",
        "Dolar kuru yeni bir rekor kırdı",
        "Bankalar kredi faiz oranlarını yükseltti",
        "İhracat rakamları geçen yıla göre arttı",
        "İşsizlik oranı yükseldi",
        "Şirket karları beklentilerin üzerinde açıklandı",
        "Merkez Bankası faiz kararını değiştirmedi",
        "Türkiye ekonomisi çeyrek bazda büyüdü",
        "Bütçe açığı geçen yıla göre arttı",
    ]
    
    # Her bir haber için duyarlılık analizi yap
    print("\nHaberler ve Duyarlılık Skorları:")
    print("-" * 40)
    
    for news in test_news:
        # Duyarlılık tahmini yap
        sentiment_score = analyzer.predict_proba([news])[0]
        sentiment_label = "Pozitif" if sentiment_score > 0 else "Negatif"
        
        # Güven skorunu -1 ile 1 arasında olacak şekilde formatla
        confidence = abs(sentiment_score)
        
        # Sonuçları göster
        print(f"Haber: {news}")
        print(f"Duyarlılık: {sentiment_label} (skor: {sentiment_score:.2f})")
        
        # Güven seviyesini açıkla
        if confidence > 0.7:
            confidence_text = "Yüksek güven"
        elif confidence > 0.4:
            confidence_text = "Orta güven"
        else:
            confidence_text = "Düşük güven"
        
        print(f"Güven seviyesi: {confidence_text}")
        print("-" * 40)
    
    print("\nKendi Haberlerinizi Analiz Edin:")
    try:
        while True:
            # Kullanıcıdan haber metnini al
            user_news = input("\nBir haber metni girin (çıkmak için 'q' yazın): ")
            if user_news.lower() == 'q':
                break
            
            # Duyarlılık analizi yap
            sentiment_score = analyzer.predict_proba([user_news])[0]
            sentiment_label = "Pozitif" if sentiment_score > 0 else "Negatif"
            
            print(f"Duyarlılık: {sentiment_label} (skor: {sentiment_score:.2f})")
    except KeyboardInterrupt:
        print("\nProgram sonlandırıldı.")
    
    print("\nDuyarlılık analizi örneği tamamlandı.")
    
if __name__ == "__main__":
    demonstrate_sentiment_analysis() 