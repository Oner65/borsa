import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os

class SentimentAnalyzer:
    """
    Borsa haberleri ve finansal metinler için duyarlılık analizi yapan sınıf.
    Scikit-learn kullanarak TF-IDF ve LinearSVC ile modeli oluşturur.
    """
    
    def __init__(self, model_path=None):
        """
        Args:
            model_path: Önceden eğitilmiş model dosyasının yolu
        """
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), 'sentiment_model.pkl')
        self.model = None
        
        # Eğer model dosyası varsa yükle
        if os.path.exists(self.model_path):
            try:
                self.model = self._load_model()
                print(f"Model başarıyla yüklendi: {self.model_path}")
            except Exception as e:
                print(f"Model yüklenirken hata oluştu: {e}")
                self._create_model()
        else:
            self._create_model()
    
    def _create_model(self):
        """Model pipeline'ını oluşturur"""
        self.model = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=5000,
                min_df=5,
                max_df=0.8,
                sublinear_tf=True,
                use_idf=True,
                ngram_range=(1, 2),
                strip_accents='unicode'
            )),
            ('classifier', LinearSVC(
                C=1,
                loss='squared_hinge',
                max_iter=1000
            ))
        ])
    
    def _preprocess_text(self, text):
        """Metni ön işleme tabi tutar"""
        if not isinstance(text, str):
            return ""
        
        # Küçük harfe çevir
        text = text.lower()
        
        # Özel karakterleri temizle
        text = re.sub(r'[^\w\s]', '', text)
        
        # Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def train(self, texts, labels):
        """
        Modeli eğitir
        
        Args:
            texts (list): Metin listesi
            labels (list): Etiket listesi (1: pozitif, 0: negatif)
        
        Returns:
            dict: Eğitim raporu
        """
        # Metinleri ön işleme
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Eğitim ve test veri setlerine ayır
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, labels, test_size=0.2, random_state=42
        )
        
        # Modeli eğit
        self.model.fit(X_train, y_train)
        
        # Test et
        y_pred = self.model.predict(X_test)
        
        # Rapor oluştur
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Modeli kaydet
        self._save_model()
        
        return report
    
    def predict(self, texts):
        """
        Metinlerin duyarlılığını tahmin eder
        
        Args:
            texts (str or list): Tek bir metin veya metin listesi
        
        Returns:
            list: Tahmin sonuçları (1: pozitif, 0: negatif)
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi.")
        
        # Tek metin mi yoksa liste mi kontrol et
        if isinstance(texts, str):
            texts = [texts]
        
        # Metinleri ön işleme
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Tahmin yap
        predictions = self.model.predict(processed_texts)
        
        return predictions
    
    def predict_proba(self, texts):
        """
        Metinlerin duyarlılık olasılıklarını tahmin eder
        
        Args:
            texts (str or list): Tek bir metin veya metin listesi
        
        Returns:
            list: Olasılık değerleri (-1 ile 1 arasında)
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi.")
        
        # Tek metin mi yoksa liste mi kontrol et
        if isinstance(texts, str):
            texts = [texts]
        
        # Metinleri ön işleme
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Model tipine göre prob hesapla
        if hasattr(self.model, 'predict_proba'):
            # CalibratedClassifierCV için
            probs = self.model.predict_proba(processed_texts)
            # Pozitif sınıf olasılıklarını al (ikinci sütun) ve [-1, 1] aralığına dönüştür
            confidences = 2 * probs[:, 1] - 1
        else:
            # LinearSVC için decision_function
            decision_values = self.model.decision_function(processed_texts)
            # Normalize et (-1 ile 1 arasında)
            max_abs = np.max(np.abs(decision_values)) if len(decision_values) > 0 else 1
            if max_abs > 0:
                confidences = decision_values / max_abs
            else:
                confidences = decision_values
        
        return confidences
    
    def _save_model(self):
        """Modeli dosyaya kaydeder"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model başarıyla kaydedildi: {self.model_path}")
        except Exception as e:
            print(f"Model kaydedilirken hata oluştu: {e}")
    
    def _load_model(self):
        """Modeli dosyadan yükler"""
        with open(self.model_path, 'rb') as f:
            return pickle.load(f)


# Örnek veri setiyle eğitim fonksiyonu
def train_with_sample_data():
    """Örnek veri setiyle modeli eğitir"""
    # Örnek finansal haberler ve duyarlılıkları
    # 1: Pozitif, 0: Nötr/Negatif
    sample_texts = [
        "Borsa güne yükselişle başladı",
        "Endeks rekor kırdı",
        "Hisseler değer kazandı",
        "Yatırımcılar kar etti",
        "Ekonomik büyüme beklentilerin üzerinde",
        "Şirket karları arttı",
        "İhracat rakamları yükseldi",
        "Piyasalarda olumlu hava",
        "Faiz indirimi bekleniyor",
        "Ekonomik veriler olumlu sinyal verdi",
        "Borsa sert düştü",
        "Hisseler değer kaybetti",
        "Piyasalar negatif seyrediyor",
        "Yatırımcılar zarar etti",
        "Ekonomik daralma bekleniyor",
        "Şirket zararları arttı",
        "İhracat rakamları geriledi",
        "Piyasalarda olumsuz hava",
        "Faiz artışı bekleniyor",
        "Ekonomik veriler olumsuz sinyal verdi"
    ]
    
    sample_labels = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]
    
    # Analiz nesnesini oluştur
    analyzer = SentimentAnalyzer()
    
    # Eğit
    report = analyzer.train(sample_texts, sample_labels)
    
    print("Model eğitim raporu:")
    print(f"Doğruluk: {report['accuracy']:.2f}")
    print(f"Pozitif F1 skoru: {report['1']['f1-score']:.2f}")
    print(f"Negatif F1 skoru: {report['0']['f1-score']:.2f}")
    
    return analyzer


# Örnek kullanım
if __name__ == "__main__":
    # Örnek model eğitimi
    analyzer = train_with_sample_data()
    
    # Örnek tahmin
    test_texts = [
        "Piyasalar güçlü bir yükseliş gösterdi",
        "Hisseler sert düşüş yaşadı",
        "Ekonomik görünüm belirsiz"
    ]
    
    predictions = analyzer.predict(test_texts)
    probabilities = analyzer.predict_proba(test_texts)
    
    print("\nTahmin Sonuçları:")
    for text, pred, prob in zip(test_texts, predictions, probabilities):
        sentiment = "Pozitif" if pred == 1 else "Negatif"
        print(f"Metin: {text}")
        print(f"Duyarlılık: {sentiment}")
        print(f"Güven skoru: {prob:.2f}")
        print("-" * 50) 