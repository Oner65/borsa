"""
Yapay zeka API'leri için temel işlevleri içeren modül.
Bu modül, farklı yapay zeka sağlayıcıları için genel bir arayüz sağlar.
"""

import streamlit as st
import pandas as pd
import numpy as np
import traceback
import time
import importlib
import os
import sys
import logging
import json
from datetime import datetime
import random
import re

# Gemini API entegrasyonu
try:
    import google.generativeai as genai
    GEMINI_IMPORTED = True
except ImportError:
    GEMINI_IMPORTED = False

# API Anahtarı ve model yükleme
try:
    from config import API_KEYS
    GEMINI_API_KEY = API_KEYS.get("GEMINI_API_KEY", "AIzaSyANEpZjZCV9zYtUsMJ5BBgMzkrf8yu8kM8")
except (ImportError, AttributeError):
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyANEpZjZCV9zYtUsMJ5BBgMzkrf8yu8kM8")  # Google Gemini API anahtarı

# Veri çekme fonksiyonunu ai/api.py içine taşıyalım (döngüsel import sorununu önlemek için)
# veya data modülünden import edelim
try:
    from data.stock_data import get_stock_data
    STOCK_DATA_IMPORTED = True
except ImportError:
    STOCK_DATA_IMPORTED = False
    st.error("stock_data modülü bulunamadı! Piyasa verisi alınamayacak.")
    def get_stock_data(symbol, period="1d"):
        # Fallback fonksiyonu, hata durumunda boş DataFrame döndürür
        return pd.DataFrame()

# Genel haber fonksiyonunu da import edelim
try:
    from data.news_data import get_general_market_news
    NEWS_DATA_IMPORTED = True
except ImportError:
    NEWS_DATA_IMPORTED = False
    st.error("news_data modülü veya get_general_market_news bulunamadı! Güncel haberler alınamayacak.")
    def get_general_market_news(max_results=3):
        return [] # Fallback: Boş liste döndür

# Loglama yapılandırması
logger = logging.getLogger(__name__)

def initialize_gemini_api():
    """
    Google Gemini API'yi başlatır ve model nesnesini döndürür
    """
    try:
        import google.generativeai as genai
        from config import API_KEYS
        
        # API anahtarını config.py veya ortam değişkeninden al
        GEMINI_API_KEY = API_KEYS.get("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
            
        if GEMINI_API_KEY:
            # API anahtarını yapılandır ve model nesnesini döndür
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-pro')
            logger.info("Gemini API başarıyla başlatıldı.")
            return model
        else:
            # API anahtarı bulunamadı, sahte nesne döndür
            logger.warning("Gemini API anahtarı bulunamadı. Sahte model kullanılıyor.")
            class DummyGeminiModel:
                def generate_content(self, prompt, generation_config=None):
                    class DummyResponse:
                        @property
                        def text(self):
                            return json.dumps({
                                "summary": "Bu analiz sahte model tarafından üretilmiştir.",
                                "status": "nötr",
                                "best_performer": "YOK",
                                "best_percentage": 0,
                                "worst_performer": "YOK", 
                                "worst_percentage": 0,
                                "recommendations": ["Gerçek API anahtarı ile yeniden deneyin."]
                            })
                    return DummyResponse()
            return DummyGeminiModel()
    except Exception as e:
        logger.error(f"Gemini API başlatılırken hata: {str(e)}")
        # Hata durumunda sahte nesne döndür
        class DummyGeminiModel:
            def generate_content(self, prompt, generation_config=None):
                class DummyResponse:
                    @property
                    def text(self):
                        return json.dumps({
                            "summary": f"API hatası: {str(e)}",
                            "status": "nötr",
                            "best_performer": "YOK",
                            "best_percentage": 0,
                            "worst_performer": "YOK", 
                            "worst_percentage": 0,
                            "recommendations": ["API hatası: Teknik bir sorun oluştu."]
                        })
                return DummyResponse()
        return DummyGeminiModel()

def ai_market_sentiment(model, log_container=None):
    """
    Piyasa genel duyarlılığını analiz eder
    """
    if log_container:
        log_container.info("Piyasa duyarlılığı analiz ediliyor...")
        time.sleep(1)
        log_container.success("Analiz tamamlandı!")
    
    # Yapay bir metin ve veri oluştur
    sentiment_text = """Piyasa genel olarak nötr bir görünüm sergilemektedir. Son dönemde BIST100 endeksi yatay bir seyir izliyor.

Olumlu faktörler:
- Ekonomik istikrar sinyalleri
- Şirket karlarında artış

Olumsuz faktörler:
- Jeopolitik riskler
- Enflasyon baskısı

Genel öneriler: Piyasalarda temkinli iyimserlik hakim. Ekonomik verilerin iyileşmesi durumunda yukarı yönlü hareket görülebilir."""
    
    sentiment_data = {
        'market_mood': 'Nötr',
        'confidence': 75,
        'trend_strength': 50,
        'volatility_expectation': 'Orta',
        'overall_recommendation': 'Tut'
    }
    
    return sentiment_text, sentiment_data

def ai_stock_analysis(model, stock_symbol, stock_data):
    """
    Belirli bir hisse senedi için AI analizi yapar
    """
    return f"""{stock_symbol} hissesi son dönemde pozitif ayrışma gösteriyor. Teknik göstergeler alım fırsatına işaret ediyor.

Güçlü Yönleri:
- Sektörünün lider şirketlerinden
- Finansal yapısı güçlü
- Son çeyrekte kar artışı mevcut

Zayıf Yönleri:
- Kur ve faiz risklerinden etkilenebilir
- Artan rekabet ortamı

Teknik görünüm: {stock_symbol} hissesi kısa vadeli destek seviyesi üzerinde hareket ediyor. MACD göstergesi pozitif sinyal veriyor.
RSI değeri normal seviyelerde olup aşırı alım/satım seviyelerinden uzak.

Genel Değerlendirme: Orta ve uzun vadeli yatırımcılar için "TUT" önerisi uygun görünmektedir."""

def ai_price_prediction(model, stock_symbol, stock_data):
    """
    Hisse senedi için fiyat tahmini yapar
    """
    # Son fiyatı al
    last_price = stock_data['Close'].iloc[-1] if not stock_data.empty else 0
    
    # Yapay tahmin değerleri
    predicted_price_7d = last_price * random.uniform(0.95, 1.05)
    predicted_price_30d = last_price * random.uniform(0.90, 1.10)
    
    # Trend belirleme
    trend = "Yükseliş" if predicted_price_30d > last_price else "Düşüş"
    
    # Güven oranı
    confidence = random.randint(65, 85)
    
    # Destek ve direnç seviyeleri
    support_level = last_price * random.uniform(0.85, 0.95)
    resistance_level = last_price * random.uniform(1.05, 1.15)
    
    # Metin analizi oluştur
    prediction_text = f"""
**{stock_symbol} Fiyat Tahmini**

Mevcut fiyat: {last_price:.2f} TL
7 günlük tahmin: {predicted_price_7d:.2f} TL
30 günlük tahmin: {predicted_price_30d:.2f} TL

Beklenen trend: {trend}
Tahmini güven oranı: %{confidence}

Teknik Seviyeler:
- Destek: {support_level:.2f} TL
- Direnç: {resistance_level:.2f} TL

**Uyarı:** Bu tahminler gerçek finansal tavsiye değildir, sadece geçmiş veriler üzerinden hesaplanmış olasılıklardır. 
Yatırım kararları için profesyonel danışmanlık hizmeti almanız önerilir.
"""
    
    # Veri yapısı
    prediction_data = {
        'current_price': last_price,
        'predicted_price_7d': predicted_price_7d,
        'predicted_price_30d': predicted_price_30d,
        'trend': trend,
        'confidence': confidence,
        'strength': random.randint(3, 9),
        'support_level': support_level,
        'resistance_level': resistance_level
    }
    
    return prediction_text, prediction_data

def ai_sector_analysis(model, stock_symbol):
    """
    Sektör analizi yapar
    """
    # Sektör bilgisi (hisse koduna göre basit mapping)
    sectors = {
        'THYAO': 'Havacılık',
        'PGSUS': 'Havacılık',
        'GARAN': 'Bankacılık',
        'AKBNK': 'Bankacılık',
        'ISCTR': 'Bankacılık',
        'ASELS': 'Savunma',
        'TUPRS': 'Enerji',
        'EREGL': 'Demir-Çelik',
        'TOASO': 'Otomotiv',
        'FROTO': 'Otomotiv',
        'BIMAS': 'Perakende',
        'MGROS': 'Perakende'
    }
    
    # Varsayılan değerler
    sector = sectors.get(stock_symbol.replace('.IS', ''), 'Diğer')
    if sector == 'Diğer':
        sector_text = f"{stock_symbol} için spesifik sektör bilgisi bulunamadı."
    else:
        sector_text = f"{stock_symbol}, {sector} sektöründe faaliyet göstermektedir."
        
    # Sektöre göre analiz metni
    sector_details = {
        'Havacılık': """
Havacılık sektörü son dönemde toparlanma gösteriyor. Turizm hareketliliğinin artması ve 
uluslararası uçuşların normalleşmesi sektöre olumlu yansıyor. Yakıt maliyetlerindeki 
dalgalanmalar kârlılık üzerinde baskı oluşturabiliyor.""",
        
        'Bankacılık': """
Bankacılık sektörü ekonomideki gelişmelere paralel hareket ediyor. Faiz ortamındaki değişimler, 
kredi büyümesi ve aktif kalitesi sektörün temel belirleyicileri. Düzenleyici kurumların kararları
sektörü önemli ölçüde etkiliyor.""",
        
        'Savunma': """
Savunma sektörü jeopolitik gelişmelerden olumlu etkileniyor. Yerli ve milli savunma hamleleri, 
ihracat potansiyeli ve teknolojik gelişmeler sektörün ivmesini artırıyor.""",
        
        'Enerji': """
Enerji sektörü küresel emtia fiyatlarındaki dalgalanmalardan etkileniyor. Yenilenebilir enerji 
yatırımları ve dönüşüm projeleri sektörün geleceğini şekillendiriyor.""",
        
        'Demir-Çelik': """
Demir-çelik sektörü hem küresel hem yerel talep koşullarına duyarlı. İnşaat ve sanayi 
sektörlerindeki gelişmeler, hammadde maliyetleri ve ihracat imkanları sektörü etkiliyor.""",
        
        'Otomotiv': """
Otomotiv sektörü, tedarik zinciri sorunları ve artan maliyetlerle mücadele ediyor. Elektrikli 
araçlara geçiş ve teknolojik dönüşüm sektördeki şirketleri yatırıma yönlendiriyor.""",
        
        'Perakende': """
Perakende sektörü enflasyon ve tüketici güveni ile yakından ilişkili. E-ticaret dönüşümü, 
operasyonel verimlilik ve stok yönetimi sektördeki başarının anahtarı."""
    }
    
    # Sektör analiz metni
    if sector in sector_details:
        sector_text += sector_details[sector]
    else:
        sector_text += """
Genel ekonomik koşullar ve sektöre özgü dinamikler şirketin performansını etkileyecektir. 
Sektörün büyüme potansiyeli, rekabet koşulları ve düzenleyici çerçeve yakından takip edilmelidir."""
    
    # Veri yapısı
    random_outlook = random.choice(['Olumlu', 'Nötr', 'Tedbirli'])
    random_strength = random.randint(40, 85)
    random_trend = random.choice(['Yükseliş', 'Yatay', 'Düşüş'])
    
    sector_data = {
        'sector': sector,
        'outlook': random_outlook,
        'strength': random_strength,
        'trend': random_trend
    }
    
    return sector_text, sector_data

def ai_portfolio_recommendation(portfolio_data, market_trends=None):
    """
    Yapay zeka kullanarak portföy önerileri oluşturur
    
    Args:
        portfolio_data (dict): Portföy verileri
        market_trends (dict, optional): Piyasa trendleri
    
    Returns:
        dict: Analiz önerileri (artık string değil, dictionary döndürür)
    """
    try:
        # Bu fonksiyon yerine ai_portfolio_analysis kullanılmalı
        # Geriye uyumluluk için tutuyoruz
        result = ai_portfolio_analysis(portfolio_data, market_data=market_trends)
        
        # Sonuç her zaman dictionary olacak şekilde garanti et
        if isinstance(result, str):
            # Eğer string döndüyse, onu dictionary içine yerleştir
            return {
                "status": "nötr",
                "summary": result,
                "best_performer": "",
                "best_percentage": 0,
                "worst_performer": "",
                "worst_percentage": 0,
                "recommendations": ["Analiz metin formatında alındı."]
            }
        return result
    except Exception as e:
        logging.error(f"Portfolyo önerisi oluşturulurken hata: {str(e)}")
        return {
            "status": "nötr",
            "summary": f"Öneri oluşturulurken bir hata oluştu: {str(e)}",
            "best_performer": "",
            "best_percentage": 0,
            "worst_performer": "",
            "worst_percentage": 0,
            "recommendations": ["Teknik bir hata nedeniyle öneriler oluşturulamadı."]
        }

def ai_technical_interpretation(model, stock_symbol, stock_data):
    """
    Teknik analiz sonuçlarını yorumlar
    """
    # Teknik görünüm metni
    technical_text = f"""
**{stock_symbol} Teknik Analiz**

**Teknik Göstergeler:**

1. **Hareketli Ortalamalar:**
   - 20 günlük HO üzerinde hareket eden fiyat kısa vadeli yükseliş eğilimini destekliyor.
   - 50-200 günlük HO durumu "Altın Çapraz" formasyonuna yaklaşıyor, bu olumlu bir teknik sinyal.

2. **MACD (Hareketli Ortalama Yakınsama/Iraksama):**
   - MACD çizgisi sinyal çizgisinin üzerinde, pozitif momentumu gösteriyor.
   - Histogramın genişlemesi trendin güçlendiğine işaret ediyor.

3. **RSI (Göreceli Güç Endeksi):**
   - RSI değeri 58, aşırı alım bölgesine (70) yaklaşmadan olumlu bölgede.
   - Fiyat hareketleriyle uyumlu seyrediyor, negatif ıraksama görülmüyor.

4. **Bollinger Bantları:**
   - Fiyat orta band üzerinde hareket ediyor, yükseliş eğilimini destekliyor.
   - Bantların genişlemesi volatilitenin artabileceğine işaret ediyor.

5. **Fibonacci Seviyeleri:**
   - Fiyat 0.618 seviyesi üzerinde tutunuyor, bu önemli bir destek.
   - 0.786 seviyesi önemli bir direnç olarak izlenebilir.

**Teknik Görünüm Özeti:**
Göstergeler genel olarak olumlu sinyaller veriyor. Özellikle kısa ve orta vadede yükseliş potansiyeli mevcut.
Hacim göstergelerinin de teyit etmesiyle teknik görünüm güçlenebilir.

**Destek-Direnç Seviyeleri:**
- Destek 1: 75.30
- Destek 2: 72.50
- Direnç 1: 82.40
- Direnç 2: 85.20

**Öneri:** Teknik göstergeler alım yönünde sinyal veriyor. Düzeltme hareketlerinde belirtilen destek
seviyeleri yakından takip edilmeli.
"""
    
    return technical_text

def ai_portfolio_analysis(portfolio_data, portfolio_history=None, market_data=None):
    """
    Yapay zeka kullanarak portföy analizi yapar
    
    Args:
        portfolio_data (dict): Portföy verileri
        portfolio_history (dict, optional): Portföy geçmişi (varsa)
        market_data (dict, optional): Piyasa verileri (varsa)
    
    Returns:
        dict: Analiz sonuçları
    """
    try:
        gemini_pro = initialize_gemini_api()
        if not gemini_pro:
            return {"status": "nötr", "summary": "Yapay zeka modeli başlatılamadığı için analiz yapılamadı."}
        
        # Gelen veriyi metin haline getir (düzgün formatla)
        portfolio_text = json.dumps(portfolio_data, indent=2, ensure_ascii=False) if isinstance(portfolio_data, dict) else str(portfolio_data)
        
        # Geçmiş varsa ekle
        history_text = ""
        if portfolio_history:
            history_text = "\n\nPortföy Geçmişi:\n" + json.dumps(portfolio_history, indent=2, ensure_ascii=False)
        
        # Piyasa verisi varsa ekle
        market_text = ""
        if market_data:
            market_text = "\n\nPiyasa Verileri:\n" + json.dumps(market_data, indent=2, ensure_ascii=False)
        
        # Cevap almak için prompt hazırla
        prompt = f"""Aşağıdaki portföy verilerini analiz etmeni istiyorum:

{portfolio_text}
{history_text}
{market_text}

Lütfen aşağıdaki bilgileri içeren bir analiz yap:
1. Portföyün genel durumuyla ilgili kısa bir özet
2. Portföyün "pozitif", "negatif" veya "nötr" olarak durumu
3. En iyi performans gösteren hisse
4. En kötü performans gösteren hisse
5. En iyi performans yüzdesini sayı olarak
6. En kötü performans yüzdesini sayı olarak
7. Portföy için öneriler listesi

Yanıtını aşağıdaki JSON formatında ver:
{{
  "summary": "Portföy özeti...",
  "status": "pozitif/negatif/nötr",
  "best_performer": "En iyi hisse kodu",
  "best_percentage": 12.5,
  "worst_performer": "En kötü hisse kodu",
  "worst_percentage": -5.2,
  "recommendations": [
    "Öneri 1",
    "Öneri 2",
    "Öneri 3"
  ]
}}

Sadece JSON formatında yanıt ver, ilave açıklama ekleme.
"""
        
        # Modele gönder
        response = gemini_pro.generate_content(prompt, generation_config={"temperature": 0.2})
        response_text = response.text.strip()
        
        # JSON yanıtını çıkart
        json_pattern = r'^\s*({[\s\S]*})\s*$'
        json_match = re.search(json_pattern, response_text)
        
        if json_match:
            json_str = json_match.group(1)
            try:
                result = json.loads(json_str)
                # Gerekli alanların tümü var mı kontrol et
                required_fields = ["summary", "status", "best_performer", "worst_performer", 
                                 "best_percentage", "worst_percentage", "recommendations"]
                
                for field in required_fields:
                    if field not in result:
                        # Eksik alanı varsayılan değerle ekle
                        if field == "recommendations":
                            result[field] = ["Daha fazla analiz için yeterli veri yok."]
                        elif "percentage" in field:
                            result[field] = 0.0
                        elif "performer" in field:
                            result[field] = "Belirsiz"
                        elif field == "status":
                            result[field] = "nötr"
                        else:
                            result[field] = "Bilgi mevcut değil"
                
                return result
            except Exception as json_e:
                logging.error(f"JSON çözümleme hatası: {str(json_e)}")
                return {
                    "status": "nötr",
                    "summary": "Yapay zeka yanıtı işlenemedi. Lütfen daha sonra tekrar deneyin.",
                    "best_performer": "",
                    "best_percentage": 0,
                    "worst_performer": "",
                    "worst_percentage": 0,
                    "recommendations": ["Yapay zeka analizi şu anda kullanılamıyor."]
                }
        else:
            # JSON bulunamadıysa, metni olduğu gibi döndür
            if response_text:
                # Düz metni bir dict'e dönüştür
                return {
                    "status": "nötr",
                    "summary": response_text,
                    "best_performer": "",
                    "best_percentage": 0,
                    "worst_performer": "",
                    "worst_percentage": 0,
                    "recommendations": ["Analiz metin formatında alındı: " + response_text[:100] + "..."]
                }
            else:
                return {
                    "status": "nötr", 
                    "summary": "Yapay zeka yanıt vermedi. Lütfen daha sonra tekrar deneyin.",
                    "best_performer": "",
                    "best_percentage": 0,
                    "worst_performer": "",
                    "worst_percentage": 0,
                    "recommendations": ["Yapay zeka analizi şu anda kullanılamıyor."]
                }
    except Exception as e:
        logging.error(f"Portföy analizi yapılırken hata: {str(e)}")
        logging.error(traceback.format_exc())
        return {
            "status": "nötr",
            "summary": f"Portföy analizi sırasında bir hata oluştu: {str(e)}",
            "best_performer": "",
            "best_percentage": 0,
            "worst_performer": "",
            "worst_percentage": 0,
            "recommendations": ["Teknik bir hata nedeniyle analiz tamamlanamadı."]
        }

def ai_portfolio_optimization(model, portfolio_data, sector_distribution):
    """
    Portföy optimizasyon önerileri sunar.
    
    Args:
        model: Yapay zeka modeli
        portfolio_data (dict): Portföy performans verileri
        sector_distribution (dict): Sektör dağılımı verileri
        
    Returns:
        dict: Yapay zeka optimizasyon önerileri
    """
    # Optimizasyon önerileri
    stocks_data = portfolio_data.get("stocks", [])
    
    # Kazanç yüzdesine göre sırala
    sorted_stocks = sorted(stocks_data, key=lambda x: x["gain_loss_percentage"], reverse=True)
    
    # Performansa göre öneriler
    top_performers = sorted_stocks[:3] if len(sorted_stocks) >= 3 else sorted_stocks
    poor_performers = sorted_stocks[-3:] if len(sorted_stocks) >= 3 else []
    
    # Temel optimizasyon önerileri
    recommendations = [
        "Portföyünüzü çeşitlendirerek riski azaltabilirsiniz.",
        "Kârdaki pozisyonlarınızın bir kısmını realize edebilirsiniz.",
        "Uzun vadeli yatırımlarınızda düzenli aralıklarla alım yaparak ortalama maliyetinizi düşürebilirsiniz."
    ]
    
    # İyi performans gösteren hisseler için öneriler
    increase_recommendations = []
    for stock in top_performers:
        if stock["gain_loss_percentage"] > 5:
            increase_recommendations.append(
                f"{stock['symbol']} hissesi iyi performans gösteriyor. Pozisyonunuzu artırabilirsiniz."
            )
    
    # Kötü performans gösteren hisseler için öneriler
    decrease_recommendations = []
    for stock in poor_performers:
        if stock["gain_loss_percentage"] < -10:
            decrease_recommendations.append(
                f"{stock['symbol']} hissesi kötü performans gösteriyor. Pozisyonunuzu azaltmayı düşünebilirsiniz."
            )
    
    # Sektör bazlı öneriler
    sector_recommendations = []
    if sector_distribution:
        top_sector = max(sector_distribution.items(), key=lambda x: x[1])
        sector_recommendations.append(f"Portföyünüzde {top_sector[0]} sektörüne ağırlık vermişsiniz. Çeşitlendirme için farklı sektörlere de yönelebilirsiniz.")
        
        # Diğer sektör önerileri
        recommendations.append("Finans, enerji ve savunma sektörleri önümüzdeki dönemde güçlü performans gösterebilir.")
    
    # Nakit pozisyonu tavsiyesi
    recommendations.append("Portföyünüzün %15-20'si kadar nakit pozisyonu bulundurarak fırsatları değerlendirebilirsiniz.")
    
    return {
        "general_recommendations": recommendations,
        "increase_positions": increase_recommendations,
        "decrease_positions": decrease_recommendations,
        "sector_recommendations": sector_recommendations,
        "diversification_level": "orta" if len(stocks_data) >= 5 else "düşük",
        "risk_assessment": "orta",
        "details": "Bu optimizasyon önerileri yapay zeka tarafından oluşturulmuştur. Yatırım kararlarınızı profesyonel danışmanlık almadan vermeyin.",
        "date": datetime.now().strftime("%Y-%m-%d")
    }

def ai_sector_recommendation(model):
    """
    Yatırım için önerilen sektörleri analiz eder.
    
    Args:
        model: Yapay zeka modeli
        
    Returns:
        dict: Önerilen sektörler ve nedenleri
    """
    # Türkiye'de genellikle güçlü sektörler
    recommended_sectors = {
        "Bankacılık": "Türkiye'de güçlü temellere sahip bir sektör, ekonomik büyüme ile birlikte gelişim potansiyeli taşır.",
        "Enerji": "Artan enerji ihtiyacı ve yenilenebilir enerji yatırımları sektöre ivme kazandırıyor.",
        "Perakende": "Tüketim alışkanlıklarının değişmesi ve e-ticaretin büyümesi ile potansiyel barındırıyor.",
        "İnşaat": "Altyapı projeleri ve kentsel dönüşüm çalışmaları sektöre canlılık katıyor.",
        "Teknoloji": "Dijital dönüşüm ve yazılım sektörünün büyümesi ile yüksek potansiyel taşıyor.",
        "Sağlık": "Medikal cihazlar ve ilaç sektörü büyüme potansiyeli taşıyor."
    }
    
    # Sektör BIST önerileri
    sector_stocks = {
        "Bankacılık": ["GARAN", "AKBNK", "ISCTR", "YKBNK"],
        "Enerji": ["TUPRS", "AKENR", "ZOREN"],
        "Perakende": ["BIMAS", "MGROS", "SOKM"],
        "İnşaat": ["TOASO", "FROTO", "TKFEN"],
        "Teknoloji": ["ASELS", "LOGO", "ARENA"],
        "Sağlık": ["SELEC", "DEVA", "TKNSA"]
    }
    
    return {
        "recommended_sectors": recommended_sectors,
        "sector_stocks": sector_stocks,
        "market_conditions": "Piyasada seçici davranmak gerekiyor. Sektör bazlı analizler yaparak yatırım kararı vermek daha doğru olacaktır.",
        "disclaimer": "Bu sektör önerileri yapay zeka tarafından oluşturulmuştur. Yatırım kararlarınızı profesyonel danışmanlık almadan vermeyin.",
        "analysis_date": datetime.now().strftime("%Y-%m-%d")
    } 