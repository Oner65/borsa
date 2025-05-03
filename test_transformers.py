import os
import sys

# Proje dizinini ekle
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Proje kök dizini sys.path'e eklendi: {project_root}")

# Transformers'ı import etmeyi dene
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    print("Transformers kütüphanesi başarıyla import edildi.")
    
    # Sentiment model testi
    try:
        # Türkçe BERT modeli için test
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
        model = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased")
        sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        
        # Test et
        test_text = "Bu hisse senedi gelecekte çok iyi performans gösterecek."
        result = sentiment_analyzer(test_text)
        print(f"Sentiment analizi sonucu: {result}")
    except Exception as e:
        print(f"Sentiment model testi başarısız: {str(e)}")

except ImportError as e:
    print(f"Transformers import edilemedi: {str(e)}")

# Kendi sentiment modelimizi test edelim
try:
    from ai.sentiment_analysis import SentimentAnalyzer
    print("SentimentAnalyzer sınıfı başarıyla import edildi.")
    
    try:
        analyzer = SentimentAnalyzer()
        test_text = "Şirket bu yıl yüksek kar beklentisi açıkladı."
        result = analyzer.analyze(test_text)
        print(f"Özel SentimentAnalyzer sonucu: {result}")
    except Exception as e:
        print(f"SentimentAnalyzer test hatası: {str(e)}")
        
except ImportError as e:
    print(f"SentimentAnalyzer import edilemedi: {str(e)}")

# Basit sentiment analizi testi
try:
    def simple_sentiment_analysis(text):
        """Basit kelime tabanlı duyarlılık analizi"""
        if not text:
            return {"label": "POSITIVE", "score": 0.5}
            
        # Türkçe olumlu ve olumsuz kelimeler
        positive_words = {
            'artış', 'yükseliş', 'kazanç', 'kâr', 'rekor', 'başarı', 'pozitif', 'olumlu', 'güçlü', 'büyüme',
            'iyileşme', 'yükseldi', 'arttı', 'çıktı', 'güven', 'istikrar', 'avantaj', 'fırsat', 'yatırım',
            'imzalandı', 'anlaşma', 'destek', 'teşvik', 'ivme', 'fayda', 'gelişti', 'yeni', 'ileri'
        }
        
        negative_words = {
            'düşüş', 'kayıp', 'zarar', 'risk', 'gerileme', 'olumsuz', 'negatif', 'zayıf', 'belirsizlik',
            'endişe', 'azaldı', 'düştü', 'kaybetti', 'gecikme', 'borç', 'iflas', 'kriz', 'tehdit', 'sorun',
            'başarısız', 'yaptırım', 'ceza', 'iptal', 'durgunluk'
        }
        
        # Metin içindeki kelimeleri kontrol et
        text = text.lower()
        words = text.split()
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total = positive_count + negative_count
        if total == 0:
            return {"label": "POSITIVE", "score": 0.5}
        
        score = positive_count / total if total > 0 else 0.5
        label = "POSITIVE" if score >= 0.5 else "NEGATIVE"
        
        return {"label": label, "score": score if label == "POSITIVE" else 1 - score}
    
    test_text = "Şirket karlılığında artış gözlendi, ancak borçlar endişe verici."
    result = simple_sentiment_analysis(test_text)
    print(f"Basit sentiment analizi sonucu: {result}")
    
except Exception as e:
    print(f"Basit sentiment analizi testi başarısız: {str(e)}")

print("Test tamamlandı!") 