"""
Duyarlılık analizi modelini eğitmek için script.
"""
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Proje dizinini path'e ekle
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

from ai.sentiment_analysis import SentimentAnalyzer
from ai.sentiment_dataset import get_turkish_financial_dataset

def optimize_hyperparameters(X_train, y_train):
    """
    GridSearchCV ile en iyi hiperparametreleri bulur
    """
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.svm import LinearSVC, SVC
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.naive_bayes import MultinomialNB
    
    print("Gelişmiş hiperparametre optimizasyonu başlatılıyor...")
    
    # SVM pipeline
    svm_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', CalibratedClassifierCV(LinearSVC(), cv=5))
    ])
    
    # RandomForest pipeline
    rf_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # GradientBoosting pipeline
    gb_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])
    
    # Naive Bayes pipeline
    nb_pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])
    
    # Modeller ve parametreleri
    pipelines = {
        'svm': {
            'pipeline': svm_pipeline,
            'params': {
                'vectorizer__max_features': [10000, 15000],
                'vectorizer__min_df': [1, 2],
                'vectorizer__ngram_range': [(1, 2), (1, 3)],
                'vectorizer__use_idf': [True],
                'vectorizer__sublinear_tf': [True],
                'classifier__estimator__C': [0.8, 1.0, 2.0],
                'classifier__estimator__class_weight': ['balanced'],
                'classifier__estimator__max_iter': [3000],
            }
        },
        'rf': {
            'pipeline': rf_pipeline,
            'params': {
                'vectorizer__max_features': [10000, 15000],
                'vectorizer__min_df': [2],
                'vectorizer__ngram_range': [(1, 2), (1, 3)],
                'vectorizer__use_idf': [True],
                'vectorizer__sublinear_tf': [True],
                'classifier__n_estimators': [200, 300],
                'classifier__max_depth': [None],
                'classifier__min_samples_split': [2],
                'classifier__class_weight': ['balanced'],
            }
        },
        'gb': {
            'pipeline': gb_pipeline,
            'params': {
                'vectorizer__max_features': [10000, 15000],
                'vectorizer__min_df': [2],
                'vectorizer__ngram_range': [(1, 2)],
                'vectorizer__use_idf': [True],
                'vectorizer__sublinear_tf': [True],
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.05, 0.1],
                'classifier__max_depth': [3, 5],
                'classifier__subsample': [0.8, 1.0],
            }
        },
        'nb': {
            'pipeline': nb_pipeline,
            'params': {
                'vectorizer__max_features': [10000, 15000],
                'vectorizer__min_df': [1, 2],
                'vectorizer__ngram_range': [(1, 2), (1, 3)],
                'classifier__alpha': [0.01, 0.1, 1.0],
                'classifier__fit_prior': [True],
            }
        }
    }
    
    # En iyi model ve skorunu tut
    best_score = 0
    best_estimator = None
    best_model_name = None
    best_params = None
    
    # Her model için grid search yap
    results = {}
    for name, config in pipelines.items():
        print(f"\n{name.upper()} modeli için grid search başlatılıyor...")
        grid_search = GridSearchCV(
            config['pipeline'],
            config['params'],
            cv=5,
            scoring='f1_weighted',
            verbose=1,
            n_jobs=-1
        )
        
        # Grid Search'ü çalıştır
        grid_search.fit(X_train, y_train)
        
        # Sonuçları kaydet
        results[name] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_
        }
        
        print(f"\n{name.upper()} için en iyi parametreler:")
        for param, value in grid_search.best_params_.items():
            print(f"{param}: {value}")
        
        print(f"\n{name.upper()} için en iyi F1 skoru: {grid_search.best_score_:.4f}")
        
        # En iyi modeli güncelle
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_estimator = grid_search.best_estimator_
            best_model_name = name
            best_params = grid_search.best_params_
    
    print(f"\nEn iyi performans gösteren model: {best_model_name.upper()} (F1 skoru: {best_score:.4f})")
    
    # En iyi iki modelle voting ensemble oluştur
    top_models = sorted(results.items(), key=lambda x: x[1]['best_score'], reverse=True)[:3]
    
    if len(top_models) >= 2:
        print("\nEn iyi modellerle Voting Ensemble oluşturuluyor...")
        estimators = [(name, results[name]['best_estimator']) for name, _ in top_models]
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=[1.2, 1.0, 0.8]  # En iyi modele daha fazla ağırlık ver
        )
        
        # Voting modelini eğit
        voting_clf.fit(X_train, y_train)
        
        # Voting model performansını ölç
        print("Voting Ensemble eğitildi.")
        
        return voting_clf
    else:
        # Sadece en iyi modeli döndür
        return best_estimator

def create_custom_analyzer():
    """
    Özelleştirilmiş parametrelerle analiz nesnesini oluşturur
    """
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.svm import LinearSVC, SVC
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.preprocessing import StandardScaler
    
    # Özel özellik çıkarıcı
    class TextFeatureExtractor(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.word_vectorizer = TfidfVectorizer(
                max_features=15000,
                min_df=1,
                max_df=0.9,
                sublinear_tf=True,
                use_idf=True,
                ngram_range=(1, 2),
                strip_accents='unicode'
            )
            self.char_vectorizer = TfidfVectorizer(
                analyzer='char',
                ngram_range=(2, 5),
                max_features=10000,
                min_df=2
            )
        
        def fit(self, X, y=None):
            self.word_vectorizer.fit(X)
            self.char_vectorizer.fit(X)
            return self
        
        def transform(self, X):
            word_features = self.word_vectorizer.transform(X)
            char_features = self.char_vectorizer.transform(X)
            return word_features, char_features
    
    # SVM tabanli siniflandirici
    svm_classifier = CalibratedClassifierCV(
        LinearSVC(
            C=2.0,
            class_weight='balanced',
            loss='squared_hinge',
            max_iter=5000,
            dual=False
        ),
        cv=5
    )
    
    # Random Forest siniflandirici
    rf_classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Gradient Boosting siniflandirici
    gb_classifier = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )
    
    # Özellik birleştirici pipeline
    feature_pipeline = Pipeline([
        ('features', FeatureUnion([
            ('word_tfidf', TfidfVectorizer(
                max_features=15000,
                min_df=1,
                max_df=0.9,
                sublinear_tf=True,
                use_idf=True,
                ngram_range=(1, 3),
                strip_accents='unicode'
            )),
            ('char_tfidf', TfidfVectorizer(
                analyzer='char',
                ngram_range=(2, 5),
                max_features=10000,
                min_df=2
            ))
        ]))
    ])
    
    # Ensemble model - modelleri birleştir
    ensemble_model = VotingClassifier(
        estimators=[
            ('svm', Pipeline([('features', feature_pipeline), ('clf', svm_classifier)])),
            ('rf', Pipeline([('features', feature_pipeline), ('clf', rf_classifier)])),
            ('gb', Pipeline([('features', feature_pipeline), ('clf', gb_classifier)]))
        ],
        voting='soft',
        weights=[1.2, 1.0, 0.8]
    )
    
    # Özel analizci nesnesi oluştur
    analyzer = SentimentAnalyzer()
    analyzer.model = ensemble_model
    
    return analyzer

def train_model(optimize=False):
    """
    Duyarlılık analizi modelini eğitir ve performansını raporlar.
    
    Args:
        optimize (bool): Eğer True ise, hiperparametre optimizasyonu yapar
    """
    print("Duyarlılık analizi modeli eğitiliyor...")
    print("-" * 50)
    
    # Veri setini yükle
    texts, labels = get_turkish_financial_dataset()
    
    print(f"Toplam veri seti büyüklüğü: {len(texts)} örnek")
    print(f"  - Pozitif örnekler: {sum(labels)}")
    print(f"  - Negatif örnekler: {len(labels) - sum(labels)}")
    print("-" * 50)
    
    # Veri ön işleme fonksiyonları
    import re
    import string
    
    def clean_text(text):
        """Metni detaylı şekilde temizle"""
        if not isinstance(text, str):
            return ""
        
        # Küçük harfe çevir
        text = text.lower()
        
        # Noktalama işaretlerini ayrı kelimeler olarak işaretle
        # Böylece !, ? gibi önemli işaretler de özellik olarak kullanılabilir
        for punct in string.punctuation:
            if punct in '.,!?%':  # Önemli noktalama işaretlerini koru
                text = text.replace(punct, f" {punct} ")
            else:
                text = text.replace(punct, ' ')
        
        # Gereksiz boşlukları temizle
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Sayıları özel token ile değiştir
        text = re.sub(r'\b\d+\b', 'SAYI', text)
        
        # Yüzde işaretlerini koru
        text = text.replace('yüzde SAYI', 'YUZDESAYI')
        
        return text
    
    # Veri setini temizle
    cleaned_texts = [clean_text(text) for text in texts]
    
    # Train-test split
    from sklearn.model_selection import train_test_split, StratifiedKFold
    
    # Veri setini eğitim ve test olarak böl
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    print(f"Eğitim seti: {len(X_train)} örnek, Test seti: {len(X_test)} örnek")
    
    # K-fold cross validation için
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Analiz nesnesini oluştur
    if optimize:
        print("Hiperparametre optimizasyonu yapılıyor...")
        best_model = optimize_hyperparameters(X_train, y_train)
        analyzer = SentimentAnalyzer()
        analyzer.model = best_model
    else:
        print("Özelleştirilmiş ensemble model oluşturuluyor...")
        analyzer = create_custom_analyzer()
    
    # Modeli eğit ve cross-validation yap
    if not optimize:
        print("\nCross-validation performans değerlendirmesi yapılıyor...")
        cv_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            # Eğitim ve doğrulama setlerini ayır
            X_train_fold, X_val_fold = [X_train[i] for i in train_idx], [X_train[i] for i in val_idx]
            y_train_fold, y_val_fold = [y_train[i] for i in train_idx], [y_train[i] for i in val_idx]
            
            # Modeli eğit
            analyzer.train(X_train_fold, y_train_fold)
            
            # Doğrulama seti üzerinde tahminler
            y_val_pred = analyzer.predict(X_val_fold)
            
            # Metrikleri hesapla
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            cv_scores['accuracy'].append(accuracy_score(y_val_fold, y_val_pred))
            cv_scores['precision'].append(precision_score(y_val_fold, y_val_pred, average='weighted'))
            cv_scores['recall'].append(recall_score(y_val_fold, y_val_pred, average='weighted'))
            cv_scores['f1'].append(f1_score(y_val_fold, y_val_pred, average='weighted'))
        
        # Cross-validation sonuçlarını raporla
        print("\nCross-validation sonuçları (5-fold):")
        print(f"Ortalama Doğruluk: {sum(cv_scores['accuracy']) / len(cv_scores['accuracy']):.2f}")
        print(f"Ortalama Precision: {sum(cv_scores['precision']) / len(cv_scores['precision']):.2f}")
        print(f"Ortalama Recall: {sum(cv_scores['recall']) / len(cv_scores['recall']):.2f}")
        print(f"Ortalama F1 skoru: {sum(cv_scores['f1']) / len(cv_scores['f1']):.2f}")
    
    # Tam eğitim setinde modeli eğit
    print("\nTam eğitim setinde model eğitiliyor...")
    analyzer.train(X_train, y_train)
    
    # Test verileri üzerinde performansı ölç
    print("\nTest performansı değerlendiriliyor...")
    y_pred = analyzer.predict(X_test)
    
    # Detaylı performans raporu
    report = classification_report(y_test, y_pred, target_names=["Negatif", "Pozitif"], output_dict=True)
    
    print("\nSınıflandırma Raporu:")
    print(f"Doğruluk: {report['accuracy']:.2f}")
    print(f"Pozitif Precision: {report['Pozitif']['precision']:.2f}")
    print(f"Pozitif Recall: {report['Pozitif']['recall']:.2f}")
    print(f"Pozitif F1 skoru: {report['Pozitif']['f1-score']:.2f}")
    print(f"Negatif Precision: {report['Negatif']['precision']:.2f}")
    print(f"Negatif Recall: {report['Negatif']['recall']:.2f}")
    print(f"Negatif F1 skoru: {report['Negatif']['f1-score']:.2f}")
    print("-" * 50)
    
    # Karışıklık matrisi (confusion matrix)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nKarışıklık Matrisi:")
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print(f"True Negative: {tn}, False Positive: {fp}")
    print(f"False Negative: {fn}, True Positive: {tp}")
    print("-" * 50)
    
    # Örnek tahminler
    print("\nÖrnek Tahminler:")
    test_examples = [
        "Borsa yeni haftaya yükselişle başladı",
        "BIST 100 endeksinde düşüş yaşandı",
        "Ekonomide belirsizlikler devam ediyor",
        "Türk Lirası dolar karşısında değer kazandı",
        "Şirketlerin karları azalmaya devam ediyor",
        "Merkez Bankası faiz kararını verdi",
        "Borsada boğalar hakimiyeti ele geçirdi",
        "Finansal kırılganlık artıyor",
        "Haftanın kazandıranları belli oldu",
        # İlave test örnekleri
        "Şirket yatırım kararı aldı",
        "Ekonomide yapısal sorunlar derinleşiyor",
        "Bankalar kredi vermeye başladı",
        "Piyasalarda olumlu hava hakim",
        "Üretim hatları kapatılıyor",
        "Marka değeri yükseldi"
    ]
    
    # Test örneklerini temizle
    cleaned_examples = [clean_text(text) for text in test_examples]
    
    predictions = analyzer.predict(cleaned_examples)
    probabilities = analyzer.predict_proba(cleaned_examples)
    
    for text, pred, prob in zip(test_examples, predictions, probabilities):
        sentiment = "Pozitif" if pred == 1 else "Negatif"
        print(f"Metin: {text}")
        print(f"Tahmin: {sentiment}")
        print(f"Güven skoru: {prob:.2f}")
        print("-" * 30)
    
    print("\nModel başarıyla eğitildi ve kaydedildi.")
    return analyzer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Duyarlılık analizi modeli eğitim scripti')
    parser.add_argument('--optimize', action='store_true', help='Hiperparametre optimizasyonu yap')
    args = parser.parse_args()
    
    train_model(optimize=args.optimize) 