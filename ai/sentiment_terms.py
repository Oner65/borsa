"""
Finansal haber analizinde kullanılacak özel terimler ve ağırlıklar.
Bu dosya, hem önceki hem de gelişmiş duyarlılık analizi sınıfları tarafından kullanılabilir.
"""

# Finans haberleri için duyarlılık analiz terimleri
# Haber metinlerinden finansal duyarlılık çıkarmak için özel terimler

# Pozitif terimler (olumlu anlam taşıyan kelimeler)
positive_terms = {
    # Genel pozitif terimler
    'yükseliş': 1.2,
    'artış': 1.2,
    'büyüme': 1.2,
    'kazanç': 1.3,
    'kâr': 1.4,
    'başarı': 1.1,
    'olumlu': 1.1,
    'güçlü': 1.2,
    'yükseldi': 1.3,
    'arttı': 1.3,
    'büyüdü': 1.3,
    'rekor': 1.5,
    'üstünde': 1.2,
    'yukarı': 1.1,
    'potansiyel': 1.1,
    'fırsat': 1.2,
    'iyileşme': 1.2,
    'toparlanma': 1.2,
    'avantaj': 1.1,
    'pozitif': 1.2,
    'istikrarlı': 1.1,
    'güven': 1.2,
    'tavsiye': 1.1,
    'artışı': 1.3,
    'iyimser': 1.3,
    'cazip': 1.2,
    'yükselişi': 1.4,
    'artışı': 1.3,
    'yükselme': 1.3,
    'iyileşti': 1.3,
    'gelişti': 1.2,
    
    # Finansal özel terimler
    'temettü': 1.6,
    'pay geri alım': 1.7,
    'kâr payı': 1.5,
    'hedef fiyat artışı': 1.6,
    'al tavsiyesi': 1.7,
    'yükselen trend': 1.5,
    'aşırı satım': 1.3,
    'düşük değerlenmiş': 1.4,
    'kârlılık artışı': 1.5,
    'gelir artışı': 1.5,
    'çeyreksel artış': 1.4,
    'güçlü bilanço': 1.5,
    'ihracat artışı': 1.4,
    'satış artışı': 1.4,
    'pazar payı artışı': 1.4,
    'ciro artışı': 1.4,
    'yatırım tavsiyesi': 1.3,
    'maliyet avantajı': 1.3,
    'verimlilik artışı': 1.3,
    'stratejik ortaklık': 1.3,
    'genişleme': 1.3,
    'başarılı sonuçlar': 1.4,
    'yeni rekor': 1.5,
    'uzun vadeli yatırım': 1.2,
    'hedef aşan': 1.5,
    'yeni sipariş': 1.4,
    'yeni müşteri': 1.3,
    'piyasa beklentisi üzerinde': 1.6,
    'tut tavsiyesi': 1.1,
    'güçlü büyüme': 1.4,
    'potansiyel değerleme': 1.2,
    'beklentilerin üzerinde': 1.5,
    'geleceğe yönelik pozitif': 1.4,
}

# Negatif terimler (olumsuz anlam taşıyan kelimeler)
negative_terms = {
    # Genel negatif terimler
    'düşüş': 1.2,
    'azalış': 1.2,
    'kayıp': 1.4,
    'zarar': 1.5,
    'başarısız': 1.3,
    'olumsuz': 1.3,
    'zayıf': 1.2,
    'düştü': 1.4,
    'azaldı': 1.3,
    'geriledi': 1.3,
    'altında': 1.2,
    'aşağı': 1.1,
    'risk': 1.2,
    'tehdit': 1.3,
    'kötüleşme': 1.4,
    'daralma': 1.3,
    'dezavantaj': 1.2,
    'negatif': 1.3,
    'belirsiz': 1.2,
    'endişe': 1.3,
    'sat tavsiyesi': 1.5,
    'azalışı': 1.3,
    'kötümser': 1.3,
    'cazip değil': 1.2,
    'düşüşü': 1.4,
    'azalışı': 1.3,
    'düşme': 1.3,
    'kötüleşti': 1.4,
    'geriledi': 1.3,
    'zayıfladı': 1.3,
    
    # Finansal özel terimler
    'iflas': 2.0,
    'borç artışı': 1.6,
    'temettü kesintisi': 1.7,
    'hedef fiyat düşüşü': 1.6,
    'sat tavsiyesi': 1.7,
    'düşen trend': 1.5,
    'aşırı alım': 1.3,
    'yüksek değerlenmiş': 1.4,
    'kârlılık düşüşü': 1.5,
    'gelir düşüşü': 1.5,
    'çeyreksel düşüş': 1.4,
    'zayıf bilanço': 1.5,
    'ihracat düşüşü': 1.4,
    'satış düşüşü': 1.4,
    'pazar payı kaybı': 1.4,
    'ciro düşüşü': 1.4,
    'yüksek borçluluk': 1.5,
    'maliyet artışı': 1.3,
    'verimlilik düşüşü': 1.3,
    'iş kaybı': 1.4,
    'daralma': 1.3,
    'başarısız sonuçlar': 1.4,
    'beklentilerin altında': 1.5,
    'iptal edilen sipariş': 1.4,
    'müşteri kaybı': 1.4,
    'piyasa beklentisi altında': 1.5,
    'finansal sıkıntı': 1.6,
    'nakit sorunu': 1.5,
    'kâr uyarısı': 1.7,
    'yavaşlayan büyüme': 1.3,
    'aşırı değerlenmiş': 1.4,
    'beklentilerin altında': 1.5,
    'geleceğe yönelik negatif': 1.4,
    'daralan marjlar': 1.4,
    'yüksek değerleme': 1.3,
    'daralan pazar': 1.3,
    'geri çekilme': 1.2,
    'soruşturma': 1.5,
    'dava': 1.4,
    'ceza': 1.5,
}

# Finans terimleri ağırlıkları - özel finansal terimler ve önemi
financial_term_weights = {
    'hisse': 1.2,
    'borsa': 1.1,
    'yatırım': 1.1,
    'piyasa': 1.0,
    'analiz': 1.0,
    'fiyat': 1.0,
    'değer': 1.0,
    'tahmin': 0.9,
    'performans': 1.0,
    'trend': 1.1,
    'beklenti': 1.0,
    'hedef fiyat': 1.3,
    'bilanço': 1.2,
    'teknik analiz': 1.1,
    'temel analiz': 1.1,
    'portföy': 0.9,
    'strateji': 0.9,
    'ekonomi': 0.8,
    'sektör': 0.9,
    'finansal tablo': 1.1,
    'kazanç': 1.2,
    'satış': 1.1,
    'gelir': 1.2,
    'maliyet': 1.1,
    'marj': 1.2,
    'raporlama': 0.9,
    'çeyrek': 1.1,
    'yıllık': 1.0,
    'açıklama': 0.9,
    'yönetim': 0.8,
}

# Özel duyurular ve ağırlıkları - çok önemli finansal olaylar
special_announcements = {
    'temettü': {'weight': 1.7, 'sentiment': 1.0},  # Genellikle olumlu
    'temettü dağıtımı': {'weight': 1.7, 'sentiment': 1.0},
    'kâr payı': {'weight': 1.7, 'sentiment': 1.0},
    'bedelsiz': {'weight': 1.6, 'sentiment': 1.0},
    'bedelli': {'weight': 1.4, 'sentiment': -0.2},  # Genelde hafif olumsuz
    'sermaye artırımı': {'weight': 1.5, 'sentiment': 0.5},  # Olumlu veya olumsuz olabilir
    'pay geri alım': {'weight': 1.7, 'sentiment': 1.0},  # Genellikle olumlu
    'hisse geri alım': {'weight': 1.7, 'sentiment': 1.0},
    'birleşme': {'weight': 1.6, 'sentiment': 0.5},  # Duruma göre
    'satın alma': {'weight': 1.6, 'sentiment': 0.7},  # Genelde olumlu
    'devralma': {'weight': 1.6, 'sentiment': 0.5},
    'ortaklık': {'weight': 1.5, 'sentiment': 0.8},
    'yeni kontrat': {'weight': 1.6, 'sentiment': 0.9},
    'anlaşma imzalandı': {'weight': 1.6, 'sentiment': 0.9},
    'teşvik': {'weight': 1.5, 'sentiment': 0.9},
    'patent': {'weight': 1.6, 'sentiment': 0.9},
    'lisans': {'weight': 1.5, 'sentiment': 0.8},
    'üst düzey atama': {'weight': 1.3, 'sentiment': 0.2},
    'yönetim değişikliği': {'weight': 1.4, 'sentiment': 0.0},  # Nötr, duruma göre değişir
    'soruşturma': {'weight': 1.5, 'sentiment': -0.9},  # Genelde olumsuz
    'dava': {'weight': 1.4, 'sentiment': -0.8},
    'ceza': {'weight': 1.5, 'sentiment': -0.9},
    'iflas': {'weight': 2.0, 'sentiment': -1.0},  # Çok olumsuz
    'konkordato': {'weight': 1.8, 'sentiment': -0.9},
    'yapılandırma': {'weight': 1.5, 'sentiment': -0.4},  # Hafif olumsuz
}

# Şirket segmentleri ve odak alanları
company_segments = {
    'bankacılık': 0.9,
    'sigorta': 0.8,
    'enerji': 0.9,
    'petrol': 0.9,
    'doğalgaz': 0.9,
    'madencilik': 0.8,
    'inşaat': 0.8,
    'otomotiv': 0.9,
    'perakende': 0.8,
    'gıda': 0.8,
    'ilaç': 0.9,
    'telekom': 0.8,
    'teknoloji': 1.0,
    'yazılım': 1.0,
    'savunma': 1.0,
    'havacılık': 0.9,
    'lojistik': 0.8,
    'turizm': 0.7,
    'tekstil': 0.7,
    'beyaz eşya': 0.8,
    'çimento': 0.7,
    'demir-çelik': 0.8,
    'cam': 0.7,
    'kağıt': 0.7,
    'ambalaj': 0.7,
    'kimya': 0.8,
    'gübre': 0.8,
    'tarım': 0.7,
    'gayrimenkul': 0.8,
    'holding': 0.7,
}

# Teknik analiz terimleri ve ağırlıkları
technical_terms = {
    'destek': {'weight': 1.3, 'sentiment': 0.3},  # Hafif olumlu
    'direnç': {'weight': 1.3, 'sentiment': -0.3},  # Hafif olumsuz (kırılmadıkça)
    'trend': {'weight': 1.2, 'sentiment': 0.0},  # Nötr (yönü belirtilmediğinde)
    'yükselen trend': {'weight': 1.4, 'sentiment': 0.8},  # Olumlu
    'düşen trend': {'weight': 1.4, 'sentiment': -0.8},  # Olumsuz
    'momentum': {'weight': 1.2, 'sentiment': 0.0},
    'hacim': {'weight': 1.1, 'sentiment': 0.0},
    'yüksek hacim': {'weight': 1.3, 'sentiment': 0.5},
    'düşük hacim': {'weight': 1.2, 'sentiment': -0.3},
    'kısa vadeli': {'weight': 1.0, 'sentiment': 0.0},
    'orta vadeli': {'weight': 1.0, 'sentiment': 0.0},
    'uzun vadeli': {'weight': 1.0, 'sentiment': 0.0},
    'aşırı alım': {'weight': 1.3, 'sentiment': -0.7},  # Olumsuz (düşüş beklentisi)
    'aşırı satım': {'weight': 1.3, 'sentiment': 0.7},  # Olumlu (yükseliş beklentisi)
    'alım fırsatı': {'weight': 1.5, 'sentiment': 0.9},
    'satım fırsatı': {'weight': 1.5, 'sentiment': -0.9},
    'fiyat hedefi': {'weight': 1.4, 'sentiment': 0.0},
    'teknik analiz': {'weight': 1.1, 'sentiment': 0.0},
    'grafik formasyonu': {'weight': 1.2, 'sentiment': 0.0},
    'baş omuz': {'weight': 1.4, 'sentiment': -0.7},
    'ters omuz baş': {'weight': 1.4, 'sentiment': 0.7},
    'üçgen': {'weight': 1.2, 'sentiment': 0.0},
    'bayrak': {'weight': 1.2, 'sentiment': 0.0},
    'kanal': {'weight': 1.2, 'sentiment': 0.0},
    'fibonacci': {'weight': 1.2, 'sentiment': 0.0},
    'MACD': {'weight': 1.2, 'sentiment': 0.0},
    'RSI': {'weight': 1.2, 'sentiment': 0.0},
    'bollinger': {'weight': 1.2, 'sentiment': 0.0},
    'hareketli ortalama': {'weight': 1.2, 'sentiment': 0.0},
    'altın kesişim': {'weight': 1.5, 'sentiment': 0.9},  # Olumlu
    'ölüm kesişimi': {'weight': 1.5, 'sentiment': -0.9},  # Olumsuz
}

def analyze_text_with_terms(text, positive_dict=None, negative_dict=None, 
                           financial_terms=None, special_anns=None,
                           company_segs=None, tech_terms=None):
    """
    Metni analiz ederek içindeki olumlu/olumsuz terimleri bulur ve 
    ağırlıklandırılmış bir duyarlılık skoru hesaplar.
    
    Args:
        text (str): Analiz edilecek metin
        positive_dict (dict): Olumlu terimler sözlüğü (terim: ağırlık)
        negative_dict (dict): Olumsuz terimler sözlüğü (terim: ağırlık)
        financial_terms (dict): Finansal terimler sözlüğü (terim: ağırlık)
        special_anns (dict): Özel duyurular (terim: {weight, sentiment})
        company_segs (dict): Şirket segmentleri (segment: ağırlık)
        tech_terms (dict): Teknik analiz terimleri (terim: {weight, sentiment})
    
    Returns:
        dict: Analiz sonuçları ve skor
    """
    if text is None or len(text.strip()) == 0:
        return {
            'score': 0.5,  # Nötr
            'positive_terms': [],
            'negative_terms': [],
            'financial_terms': [],
            'special_announcements': [],
            'company_segments': [],
            'technical_terms': []
        }
    
    # Varsayılan sözlükleri kullan
    if positive_dict is None:
        positive_dict = positive_terms
    if negative_dict is None:
        negative_dict = negative_terms
    if financial_terms is None:
        financial_terms = financial_term_weights
    if special_anns is None:
        special_anns = special_announcements
    if company_segs is None:
        company_segs = company_segments
    if tech_terms is None:
        tech_terms = technical_terms
    
    text = text.lower()
    
    # Bulunan terimler ve ağırlıkları
    found_positive = {}
    found_negative = {}
    found_financial = {}
    found_special = {}
    found_segments = {}
    found_technical = {}
    
    # Olumlu terimleri ara
    for term, weight in positive_dict.items():
        if term.lower() in text:
            found_positive[term] = weight
    
    # Olumsuz terimleri ara
    for term, weight in negative_dict.items():
        if term.lower() in text:
            found_negative[term] = weight
    
    # Finansal terimleri ara
    for term, weight in financial_terms.items():
        if term.lower() in text:
            found_financial[term] = weight
    
    # Özel duyuruları ara
    for term, info in special_anns.items():
        if term.lower() in text:
            found_special[term] = info
    
    # Şirket segmentlerini ara
    for segment, weight in company_segs.items():
        if segment.lower() in text:
            found_segments[segment] = weight
    
    # Teknik terimleri ara
    for term, info in tech_terms.items():
        if term.lower() in text:
            found_technical[term] = info
    
    # Ağırlıkları hesapla
    positive_score = sum(found_positive.values()) if found_positive else 0
    negative_score = sum(found_negative.values()) if found_negative else 0
    
    # Finansal terim ağırlık çarpanı
    fin_factor = 1.0
    if found_financial:
        fin_factor = sum(found_financial.values()) / len(found_financial)
    
    # Özel duyuru etkisi
    special_sentiment_effect = 0
    special_weight_sum = 0
    
    for term, info in found_special.items():
        special_sentiment_effect += info['sentiment'] * info['weight']
        special_weight_sum += info['weight']
    
    # Teknik terim etkisi
    tech_sentiment_effect = 0
    tech_weight_sum = 0
    
    for term, info in found_technical.items():
        tech_sentiment_effect += info['sentiment'] * info['weight']
        tech_weight_sum += info['weight']
    
    # Segment faktörü
    segment_factor = 1.0
    if found_segments:
        segment_factor = sum(found_segments.values()) / len(found_segments)
    
    # Ana skor hesaplama
    base_score = 0.5  # Başlangıç nötr skor
    
    if positive_score > 0 or negative_score > 0:
        # Temel pozitif/negatif denge
        raw_sentiment = (positive_score - negative_score) / (positive_score + negative_score)
        
        # Özel duyuru etkisi
        if special_weight_sum > 0:
            special_effect = special_sentiment_effect / special_weight_sum
            raw_sentiment = raw_sentiment * 0.7 + special_effect * 0.3
        
        # Teknik terim etkisi
        if tech_weight_sum > 0:
            tech_effect = tech_sentiment_effect / tech_weight_sum
            raw_sentiment = raw_sentiment * 0.8 + tech_effect * 0.2
        
        # Finansal ve segment faktörlerini dahil et
        raw_sentiment = raw_sentiment * (fin_factor * 0.5 + segment_factor * 0.5)
        
        # Normalize et [-1, 1] -> [0, 1]
        normalized_score = (raw_sentiment + 1) / 2
        base_score = normalized_score
    
    # Eğer hiç terim bulunamadıysa
    if not found_positive and not found_negative and not found_special and not found_technical:
        base_score = 0.5  # Nötr
    
    # Sonuçları Döndür
    return {
        'score': base_score,
        'positive_terms': list(found_positive.keys()),
        'negative_terms': list(found_negative.keys()),
        'financial_terms': list(found_financial.keys()),
        'special_announcements': list(found_special.keys()),
        'company_segments': list(found_segments.keys()),
        'technical_terms': list(found_technical.keys())
    }

# Test
if __name__ == "__main__":
    test_texts = [
        "Şirket bu çeyrekte beklentilerin üzerinde bir kar açıkladı ve temettü dağıtımı yapacağını duyurdu.",
        "Firma yüksek borçluluk nedeniyle finansal sıkıntı yaşıyor ve temettü kesintisi yaptı.",
        "Analistler şirket için al tavsiyesi verirken, hedef fiyatını yükseltti ve güçlü büyüme beklediğini açıkladı.",
        "Şirket geçen yıla göre satışlarında düşüş yaşadı ve beklentilerin altında performans gösterdi.",
        "Hisse teknik olarak aşırı satım bölgesinde ve alım fırsatı sunuyor.",
        "Bankacılık sektöründe faaliyet gösteren şirket pay geri alım programı başlattı."
    ]
    
    for i, text in enumerate(test_texts):
        result = analyze_text_with_terms(text)
        print(f"\nTest {i+1}: {text}")
        print(f"Score: {result['score']:.2f}")
        print(f"Positive terms: {result['positive_terms']}")
        print(f"Negative terms: {result['negative_terms']}")
        print(f"Financial terms: {result['financial_terms']}")
        print(f"Special announcements: {result['special_announcements']}")
        print(f"Technical terms: {result['technical_terms']}")
    
    # Sözlük büyüklüklerini yazdır
    print(f"\nPositive terms: {len(positive_terms)}")
    print(f"Negative terms: {len(negative_terms)}")
    print(f"Financial terms: {len(financial_term_weights)}")
    print(f"Special announcements: {len(special_announcements)}")
    print(f"Technical terms: {len(technical_terms)}") 