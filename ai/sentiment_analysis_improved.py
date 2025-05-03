import os
import re
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline

# Düzenli ifadeler ve özel durum tespiti için sözlükler
POSITIVE_INDICATORS = {
    'yükseliş': 0.7,
    'artış': 0.7,
    'büyüme': 0.7,
    'rekor': 0.8,
    'kar': 0.7,
    'olumlu': 0.8,
    'başarılı': 0.7,
    'yükseldi': 0.7,
    'arttı': 0.7,
    'güçlü': 0.7,
    'pay geri alım': 0.9,
    'temettü': 0.8,
    'birleşme': 0.6,
    'satın alma': 0.6,
    'kazandı': 0.7,
    'aştı': 0.7,
    'üzerine çıktı': 0.7,
    'beklentilerin üzerinde': 0.8
}

NEGATIVE_INDICATORS = {
    'düşüş': 0.7,
    'azalış': 0.7,
    'daralma': 0.7,
    'kayıp': 0.7,
    'zarar': 0.8,
    'olumsuz': 0.8,
    'başarısız': 0.7,
    'düştü': 0.7,
    'azaldı': 0.7,
    'zayıf': 0.7,
    'ceza': 0.8,
    'soruşturma': 0.7,
    'dava': 0.6,
    'iflas': 0.9,
    'kaybetti': 0.7,
    'geriledi': 0.7,
    'altına indi': 0.7,
    'beklentilerin altında': 0.8
}

# Finansal terimlerin ağırlıkları
FINANCIAL_TERM_WEIGHTS = {
    'kar payı': 1.2,
    'net kar': 1.3,
    'faaliyet karı': 1.2,
    'ebitda': 1.2,
    'gelir': 1.1,
    'satış': 1.1,
    'ciro': 1.1,
    'ihracat': 1.2,
    'borçluluk': 0.9,
    'kaldıraç': 0.9,
    'faiz': 0.9,
    'kredi': 0.9,
    'tahvil': 0.8,
    'bono': 0.8,
    'yatırım': 1.1,
    'teşvik': 1.2,
    'vergi': 0.9,
    'teknoloji': 1.1,
    'inovasyon': 1.1,
    'büyüme': 1.1,
    'kapasite': 1.1,
    'üretim': 1.1,
}

class ImprovedSentimentAnalyzer:
    """
    Borsa haberleri ve finansal metinler için gelişmiş duyarlılık analizi yapan sınıf.
    Transformer tabanlı dil modellerini ve özel finans sözlüğünü kullanır.
    """
    
    def __init__(self, model_path=None, use_bert=True):
        """
        Args:
            model_path: Önceden eğitilmiş model dosyasının yolu
            use_bert: BERT modelini kullanmak için True, kelime bazlı analiz için False
        """
        self.use_bert = use_bert
        
        # BERT model yolu veya varsayılan
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), 'sentiment_model_bert')
        
        # Yedek model için dosya yolu
        self.backup_model_path = os.path.join(os.path.dirname(__file__), 'sentiment_model.pkl')
        
        self.bert_model = None
        self.tokenizer = None
        self.backup_model = None
        
        # BERT modelini yükle
        if self.use_bert:
            try:
                # Eğer özel eğitilmiş model varsa onu yükle
                if os.path.exists(self.model_path):
                    self.load_bert_model()
                # Yoksa önceden eğitilmiş Türkçe BERT modelini yükle
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
                    self.bert_model = AutoModelForSequenceClassification.from_pretrained(
                        "dbmdz/bert-base-turkish-cased", 
                        num_labels=3  # Pozitif, Nötr, Negatif
                    )
                print(f"BERT modeli başarıyla yüklendi")
            except Exception as e:
                print(f"BERT model yüklenirken hata oluştu: {e}")
                self.use_bert = False
        
        # Yedek model olarak eski modeli yükle
        try:
            if os.path.exists(self.backup_model_path):
                with open(self.backup_model_path, 'rb') as f:
                    self.backup_model = pickle.load(f)
                print(f"Yedek analiz modeli başarıyla yüklendi: {self.backup_model_path}")
        except Exception as e:
            print(f"Yedek model yüklenirken hata oluştu: {e}")
    
    def _preprocess_text(self, text):
        """Metni ön işleme tabi tutar"""
        if not isinstance(text, str):
            return ""
        
        # Küçük harfe çevir
        text = text.lower()
        
        # HTML tag'lerini temizle
        text = re.sub(r'<.*?>', '', text)
        
        # URL'leri temizle
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def load_bert_model(self):
        """BERT modelini yükler"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            return True
        except Exception as e:
            print(f"BERT model yüklenirken hata oluştu: {e}")
            return False
    
    def _word_based_analysis(self, text):
        """
        Kelime bazlı duyarlılık analizi yapar.
        Özel finansal terimler ve duygu belirten kelimeler için ağırlıklandırma kullanır.
        """
        if not text:
            return 0.5  # Nötr
        
        text = self._preprocess_text(text)
        
        # Hisse kodu tespiti
        stock_code_pattern = r'[A-Z]{3,6}\.[A-Z]{2}'
        contains_stock_code = bool(re.search(stock_code_pattern, text))
        
        # Özel durum tespiti
        positive_score = 0
        negative_score = 0
        detected_terms = []
        
        # Olumlu terimler
        for term, weight in POSITIVE_INDICATORS.items():
            if term in text:
                positive_score += weight
                detected_terms.append(f"Olumlu: {term} ({weight})")
        
        # Olumsuz terimler
        for term, weight in NEGATIVE_INDICATORS.items():
            if term in text:
                negative_score += weight
                detected_terms.append(f"Olumsuz: {term} ({weight})")
        
        # Finansal terimler için ağırlıklandırma
        financial_weight = 1.0
        for term, weight in FINANCIAL_TERM_WEIGHTS.items():
            if term in text:
                financial_weight *= weight
                detected_terms.append(f"Finansal: {term} ({weight})")
        
        # Son skoru hesapla
        if positive_score > 0 or negative_score > 0:
            # Skorları normalize et
            total_score = positive_score - negative_score
            normalized_score = financial_weight * total_score / max(positive_score + negative_score, 1)
            
            # -1 ile 1 arasına sıkıştır
            normalized_score = max(min(normalized_score, 1.0), -1.0)
            
            # 0-1 aralığına dönüştür
            sentiment_score = (normalized_score + 1) / 2
            
            if detected_terms:
                print(f"Tespit edilen terimler: {', '.join(detected_terms)}")
            if contains_stock_code:
                print(f"Hisse kodu içerikte geçiyor. Faydalı bir haber olabilir.")
                # Hisse kodu içeren habere daha fazla ağırlık ver
                sentiment_score = 0.6 * sentiment_score + 0.4 * (0.6 if sentiment_score >= 0.5 else 0.4)
            
            return sentiment_score
        else:
            # Hiçbir terim bulunamadıysa nötr değer döndür
            return 0.5
    
    def _bert_analysis(self, text):
        """
        BERT tabanlı duyarlılık analizi yapar.
        Sonucu 0-1 aralığına normalize eder.
        """
        try:
            # Metni tokenize et
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            
            # Modeli değerlendirme moduna al
            self.bert_model.eval()
            
            # Gradient hesaplamasını kapat
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                logits = outputs.logits
                
                # Softmax uygula
                probs = torch.nn.functional.softmax(logits, dim=1)
                
                # Çıktıları al: [negative, neutral, positive]
                negative_prob = probs[0][0].item()
                neutral_prob = probs[0][1].item()
                positive_prob = probs[0][2].item()
                
                # Sentiment skoru hesapla: negative: -1, neutral: 0, positive: 1
                # Ağırlıklı ortalama alıp -1 ile 1 arasında normalize et
                sentiment_score = -1 * negative_prob + 0 * neutral_prob + 1 * positive_prob
                
                # 0-1 aralığına dönüştür
                normalized_score = (sentiment_score + 1) / 2
                
                return normalized_score
                
        except Exception as e:
            print(f"BERT analizi sırasında hata: {e}")
            # Hata durumunda kelime bazlı analize geri dön
            return self._word_based_analysis(text)
    
    def analyze(self, text):
        """
        Metni analiz eder ve duyarlılık skorunu döndürür.
        Önce BERT modeli ile denenir, başarısız olursa kelime bazlı analiz yapılır.
        
        Args:
            text (str): Analiz edilecek metin
            
        Returns:
            float: 0-1 arasında duyarlılık skoru (0: olumsuz, 0.5: nötr, 1: olumlu)
        """
        text = self._preprocess_text(text)
        
        # Metin boşsa veya çok kısaysa nötr kabul et
        if not text or len(text) < 5:
            return 0.5
        
        # BERT modeli ile analiz dene
        if self.use_bert and self.bert_model and self.tokenizer:
            try:
                return self._bert_analysis(text)
            except Exception as e:
                print(f"BERT analizi başarısız, kelime bazlı analize geçiliyor. Hata: {e}")
        
        # Yedek model ile analiz dene
        if self.backup_model:
            try:
                # Yedek model predict_proba kullanıyorsa
                if hasattr(self.backup_model, 'predict_proba'):
                    probs = self.backup_model.predict_proba([text])
                    # İki sınıflı model (pozitif/negatif) için:
                    if probs.shape[1] == 2:
                        # Pozitif sınıf olasılığını al (0-1 arası)
                        return probs[0][1]
                    else:
                        # Çok sınıflı model için ortalama al
                        return 0.5
                # Decision function varsa
                elif hasattr(self.backup_model, 'decision_function'):
                    decision = self.backup_model.decision_function([text])
                    # -1 ile 1 arasında dönecek şekilde normalize et
                    max_abs = max(abs(decision[0]), 1)
                    normalized = decision[0] / max_abs
                    # 0-1 aralığına dönüştür
                    return (normalized + 1) / 2
            except Exception as e:
                print(f"Yedek model analizi başarısız: {e}")
        
        # Son çare olarak kelime bazlı analiz
        return self._word_based_analysis(text)
        
    def analyze_financial_news(self, news_texts):
        """
        Finansal haber metinlerini analiz eder ve duyarlılık skorlarını döndürür.
        
        Args:
            news_texts (list): Haber metinleri listesi
            
        Returns:
            list: Her haber için duyarlılık skoru (0-1 arası)
            float: Ortalama duyarlılık skoru
        """
        if not news_texts:
            return [], 0.5
        
        scores = []
        successful_analyses = 0
        total_score = 0
        
        for news in news_texts:
            try:
                score = self.analyze(news)
                scores.append(score)
                total_score += score
                successful_analyses += 1
                
                # Sonucu yorumla
                if score > 0.7:
                    sentiment = "Olumlu"
                elif score < 0.3:
                    sentiment = "Olumsuz"
                else:
                    sentiment = "Nötr"
                
                print(f"Gelişmiş analiz sonucu: {sentiment} ({score:.2f})")
                
            except Exception as e:
                print(f"Haber analiz edilirken hata oluştu: {e}")
                scores.append(0.5)  # Hata durumunda nötr değer ekle
        
        # Ortalama duyarlılık hesapla
        avg_sentiment = total_score / max(successful_analyses, 1)
        
        # Duyarlılığı yorumla
        if avg_sentiment > 0.7:
            print(f"Ortalama Duyarlılık: Olumlu ({avg_sentiment:.2f})")
        elif avg_sentiment < 0.3:
            print(f"Ortalama Duyarlılık: Olumsuz ({avg_sentiment:.2f})")
        else:
            print(f"Ortalama Duyarlılık: Nötr ({avg_sentiment:.2f})")
        
        # Normalize Edilmiş Skor: -1 ile 1 arasında
        normalized_score = 2 * avg_sentiment - 1
        print(f"Normalize Edilmiş Skor: {normalized_score:.2f}")
        
        return scores, avg_sentiment
    
    def train(self, texts, labels, epochs=3, batch_size=8):
        """
        BERT modelini eğitir
        
        Args:
            texts (list): Metin listesi
            labels (list): Etiket listesi (0: negatif, 1: nötr, 2: pozitif)
            epochs (int): Eğitim devir sayısı
            batch_size (int): Batch boyutu
        
        Returns:
            dict: Eğitim raporu
        """
        if not self.use_bert:
            print("BERT modeli aktif değil, eğitim yapılamıyor.")
            return None
        
        try:
            # Veri setini hazırla
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=0.2, random_state=42
            )
            
            # Tokenize et
            train_encodings = self.tokenizer(train_texts, truncation=True, padding=True)
            val_encodings = self.tokenizer(val_texts, truncation=True, padding=True)
            
            # Dataset sınıfı
            class FinancialNewsDataset(torch.utils.data.Dataset):
                def __init__(self, encodings, labels):
                    self.encodings = encodings
                    self.labels = labels

                def __getitem__(self, idx):
                    item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
                    item["labels"] = torch.tensor(self.labels[idx])
                    return item

                def __len__(self):
                    return len(self.labels)
            
            # Dataset nesnelerini oluştur
            train_dataset = FinancialNewsDataset(train_encodings, train_labels)
            val_dataset = FinancialNewsDataset(val_encodings, val_labels)
            
            # Eğitim argümanlarını tanımla
            training_args = TrainingArguments(
                output_dir=self.model_path,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir='./logs',
                logging_steps=10,
                evaluation_strategy="epoch",
            )
            
            # Trainer nesnesini oluştur
            trainer = Trainer(
                model=self.bert_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset
            )
            
            # Eğitimi başlat
            trainer.train()
            
            # Modeli kaydet
            self.bert_model.save_pretrained(self.model_path)
            self.tokenizer.save_pretrained(self.model_path)
            
            # Değerlendirme yap
            eval_results = trainer.evaluate()
            
            return eval_results
            
        except Exception as e:
            print(f"BERT model eğitimi sırasında hata: {e}")
            return None

    def predict_sentiment_label(self, text):
        """
        Metin için duyarlılık etiketini tahmin eder
        
        Args:
            text (str): Metin
            
        Returns:
            str: Duyarlılık etiketi (Olumlu, Nötr, Olumsuz)
        """
        score = self.analyze(text)
        
        if score > 0.7:
            return "Olumlu"
        elif score < 0.3:
            return "Olumsuz"
        else:
            return "Nötr"


def example_usage():
    """ImprovedSentimentAnalyzer sınıfının örnek kullanımı"""
    
    analyzer = ImprovedSentimentAnalyzer(use_bert=False)  # BERT modelini yükleme süresi uzun olduğu için kapalı başlatıyoruz
    
    # Örnek haberler
    news_samples = [
        "Pegasus Hava Yolları karını yüzde 15 artırdı ve hisse başına 2.5 TL temettü dağıtacağını açıkladı.",
        "BIST 100 endeksi günü yüzde 0.5 düşüşle kapattı. Bankacılık hisseleri satış baskısı altında kaldı.",
        "X Şirketi'nin borçları arttı ve kredi notu düşürüldü.",
        "Teknoloji şirketi, yeni ürünü için üretim kapasitesini iki katına çıkaracağını duyurdu.",
        "Pay geri alım programı başlatan şirket, 3 ay içinde 100 milyon TL'lik hisse alımı yapacak.",
        "Şirket, beklentilerin üzerinde kar açıkladı ve yıl sonu hedeflerini yukarı yönlü revize etti."
    ]
    
    print("Haber Duyarlılık Analizi Örnekleri:\n")
    
    for i, news in enumerate(news_samples):
        score = analyzer.analyze(news)
        label = analyzer.predict_sentiment_label(news)
        
        print(f"Haber {i+1}: {news}")
        print(f"Duyarlılık: {label} ({score:.2f})")
        print("-" * 80)
    
    # Toplu analiz
    print("\nToplu Analiz Sonucu:")
    scores, avg_score = analyzer.analyze_financial_news(news_samples)
    print(f"Haber skorları: {[round(score, 2) for score in scores]}")
    print(f"Ortalama skor: {avg_score:.2f}")
    
    # Özel terim tespiti örnekleri
    special_terms = [
        "Şirket pay geri alım programı başlattı.",
        "THYAO.IS hissesi yüzde 5 düştü.",
        "Faiz oranlarındaki artış bankaların karlarını olumlu etkiledi.",
        "Temettü dağıtımı beklentilerin altında kaldı."
    ]
    
    print("\nÖzel Terim Tespiti Örnekleri:\n")
    
    for term in special_terms:
        print(f"Metin: {term}")
        score = analyzer._word_based_analysis(term)
        label = "Olumlu" if score > 0.7 else ("Olumsuz" if score < 0.3 else "Nötr")
        print(f"Sonuç: {label} ({score:.2f})")
        print("-" * 80)


if __name__ == "__main__":
    example_usage() 