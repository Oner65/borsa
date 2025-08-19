import logging
import requests
from textblob import TextBlob
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

# NLTK verilerini indir (sadece ilk çalıştırmada gerekli)
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

class SentimentAnalyzer:
    """Basit sentiment analizi sınıfı"""
    
    def __init__(self):
        try:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        except ImportError:
            # Eğer vaderSentiment yüklenemezse, basit bir alternatif kullan
            self.vader_analyzer = None
        self.logger = logging.getLogger(__name__)
    
    def analyze_text(self, text):
        """
        Verilen metni analiz ederek sentiment skorunu döndürür
        
        Args:
            text (str): Analiz edilecek metin
            
        Returns:
            dict: Sentiment analizi sonuçları
        """
        if not text or not isinstance(text, str):
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'label': 'neutral'
            }
        
        try:
            # VADER analizi
            if self.vader_analyzer:
                vader_scores = self.vader_analyzer.polarity_scores(text)
            else:
                # Basit yedek analiz
                vader_scores = {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}
            
            # TextBlob analizi (yedek olarak)
            try:
                blob = TextBlob(text)
                textblob_sentiment = blob.sentiment.polarity
            except:
                textblob_sentiment = 0.0
            
            # Sonuçları birleştir
            compound_score = (vader_scores['compound'] + textblob_sentiment) / 2
            
            # Label belirle
            if compound_score >= 0.05:
                label = 'positive'
            elif compound_score <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'
            
            return {
                'compound': compound_score,
                'positive': vader_scores['pos'],
                'negative': vader_scores['neg'],
                'neutral': vader_scores['neu'],
                'label': label,
                'vader_compound': vader_scores['compound'],
                'textblob_polarity': textblob_sentiment
            }
            
        except Exception as e:
            self.logger.error(f"Sentiment analizi hatası: {e}")
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'label': 'neutral'
            }
    
    def analyze_news_data(self, news_data):
        """
        Haber verilerini toplu olarak analiz eder
        
        Args:
            news_data (list): Haber listesi
            
        Returns:
            list: Sentiment skorları ile birlikte haber verileri
        """
        analyzed_data = []
        
        for news in news_data:
            if isinstance(news, dict):
                title = news.get('title', '')
                description = news.get('description', '')
                content = news.get('content', '')
                
                # Birleşik metin oluştur
                combined_text = f"{title} {description} {content}".strip()
                
                # Sentiment analizi yap
                sentiment = self.analyze_text(combined_text)
                
                # Orijinal veriyi kopyala ve sentiment ekle
                analyzed_news = news.copy()
                analyzed_news['sentiment'] = sentiment
                analyzed_data.append(analyzed_news)
            else:
                # Eğer string ise direkt analiz et
                sentiment = self.analyze_text(str(news))
                analyzed_data.append({
                    'text': str(news),
                    'sentiment': sentiment
                })
        
        return analyzed_data
    
    def get_market_sentiment_score(self, symbol, days=30):
        """
        Belirli bir hisse senedi için genel piyasa sentiment skoru hesaplar
        
        Args:
            symbol (str): Hisse senedi sembolü
            days (int): Kaç günlük veri kullanılacağı
            
        Returns:
            float: -1 ile 1 arasında sentiment skoru
        """
        try:
            # Basit piyasa sentiment hesaplaması
            # Gerçek uygulamada burada haber API'lerinden veri çekilebilir
            
            # Şimdilik rastgele/basit bir hesaplama yapalım
            # Bu, gerçek sentiment verisi yerine örnek bir implementasyon
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days}d")
            
            if len(hist) < 5:
                return 0.0
            
            # Basit trend analizi ile sentiment tahmini
            recent_prices = hist['Close'].tail(5)
            price_change = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
            
            # Volatilite kontrolü
            volatility = hist['Close'].pct_change().std()
            
            # Basit sentiment skoru (-1 ile 1 arasında)
            sentiment_score = price_change * 2  # -1 ile 1 arasına normalize et
            sentiment_score = max(-1, min(1, sentiment_score))
            
            # Yüksek volatilite sentiment skorunu azaltır
            if volatility > 0.05:  # %5'ten fazla volatilite
                sentiment_score *= 0.7
            
            return sentiment_score
            
        except Exception as e:
            self.logger.error(f"Piyasa sentiment skoru hesaplama hatası: {e}")
            return 0.0

def simple_sentiment_analysis(text):
    """
    Basit sentiment analizi fonksiyonu (geriye uyumluluk için)
    
    Args:
        text (str): Analiz edilecek metin
        
    Returns:
        dict: Sentiment analizi sonuçları
    """
    analyzer = SentimentAnalyzer()
    return analyzer.analyze_text(text)

# Modül test fonksiyonu
if __name__ == "__main__":
    # Test kodları
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "Bu hisse senedi çok iyi performans gösteriyor!",
        "Piyasa çok kötü durumda, büyük kayıplar var.",
        "Normal bir işlem günü, özel bir şey yok."
    ]
    
    print("Sentiment Analizi Testleri:")
    print("-" * 50)
    
    for text in test_texts:
        result = analyzer.analyze_text(text)
        print(f"Metin: {text}")
        print(f"Sonuç: {result['label']} (Skor: {result['compound']:.3f})")
        print() 