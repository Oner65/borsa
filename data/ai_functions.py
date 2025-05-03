"""
ai/api.py modülündeki AI işlevlerine yönlendiren köprü modülü.
Bu modül, kodun geri kalanının mevcut import ifadeleriyle çalışmasını sağlamak için oluşturulmuştur.
"""
from datetime import datetime
import os
import logging
import requests
import pandas as pd
import yfinance as yf
import random

# AI API fonksiyonlarını buraya yönlendiriyoruz
from ai.api import (
    initialize_gemini_api as _initialize_gemini_api,
    ai_market_sentiment as _ai_market_sentiment,
    ai_stock_analysis as _ai_stock_analysis,
    ai_price_prediction as _ai_price_prediction,
    ai_sector_analysis as _ai_sector_analysis,
    ai_portfolio_recommendation as _ai_portfolio_recommendation,
    ai_technical_interpretation as _ai_technical_interpretation,
    ai_portfolio_analysis as _ai_portfolio_analysis,
    ai_portfolio_optimization as _ai_portfolio_optimization,
    ai_sector_recommendation as _ai_sector_recommendation,
)

# API'yi başlatmak için kullanılan fonksiyon
def load_gemini_pro():
    """
    Gemini API'yi başlatır ve hazır bir model döndürür
    """
    return _initialize_gemini_api()

# API fonksiyonlarını yönlendirme
def ai_market_sentiment(gemini_pro, log_container=None):
    """
    Piyasa genel duyarlılığını analiz eder
    Returns:
        tuple: (sentiment_text, sentiment_data) tüple'ı döndürür
    """
    try:
        if log_container:
            log_container.info("BIST100 verilerini alıyorum...")
        
        # BIST100 verilerini al
        from data.stock_data import get_stock_data
        bist100_data = get_stock_data("XU100.IS", period="1mo")
        
        if bist100_data is None or bist100_data.empty:
            raise Exception("BIST100 verileri alınamadı")
        
        if log_container:
            log_container.info("Piyasa haberleri toplanıyor...")
        
        # Piyasa haberleri
        try:
            from data.news_data import get_general_market_news
            market_news = get_general_market_news(max_results=5)
        except Exception as e:
            if log_container:
                log_container.warning(f"Haberler alınamadı: {str(e)}")
            market_news = []
        
        # Temel piyasa analizini yap
        current_price = bist100_data['Close'].iloc[-1]
        previous_price = bist100_data['Close'].iloc[-2]
        price_change = (current_price - previous_price) / previous_price * 100
        
        # Son bir aydaki değişim
        month_start_price = bist100_data['Close'].iloc[0]
        month_change = (current_price - month_start_price) / month_start_price * 100
        
        # Volatilite hesapla
        volatility = bist100_data['Close'].pct_change().std() * 100
        
        # Trend tespiti
        sma20 = bist100_data['Close'].rolling(window=20).mean().iloc[-1] if len(bist100_data) >= 20 else 0
        
        # Piyasa durumunu belirle
        if price_change > 1.5:
            market_mood = "Olumlu"
        elif price_change < -1.5:
            market_mood = "Olumsuz"
        else:
            market_mood = "Nötr"
        
        # Trend gücünü belirle
        if month_change > 5:
            trend_strength = 80
        elif month_change > 3:
            trend_strength = 65
        elif month_change > 0:
            trend_strength = 55
        elif month_change > -3:
            trend_strength = 45
        elif month_change > -5:
            trend_strength = 35
        else:
            trend_strength = 20
        
        # Volatilite seviyesini belirle
        if volatility > 3:
            volatility_expectation = "Yüksek"
        elif volatility > 1.5:
            volatility_expectation = "Orta"
        else:
            volatility_expectation = "Düşük"
        
        # Tavsiye belirle
        if market_mood == "Olumlu" and current_price > sma20 and volatility_expectation != "Yüksek":
            recommendation = "Al"
        elif market_mood == "Olumsuz" and current_price < sma20:
            recommendation = "Sat"
        else:
            recommendation = "Tut"
        
        # Güven oranını belirle
        confidence = 70 + (abs(month_change) if month_change > 0 else -abs(month_change)) / 2
        confidence = max(60, min(95, confidence))
        
        # Veri sözlüğünü oluştur
        sentiment_data = {
            'market_mood': market_mood,
            'confidence': int(confidence),
            'trend_strength': int(trend_strength),
            'volatility_expectation': volatility_expectation,
            'overall_recommendation': recommendation
        }
        
        # Yapay analiz metnini oluştur
        olumlu_faktorler = [
            "Ekonomik istikrar sinyalleri",
            "Şirket karlarında artış"
        ]
        
        olumsuz_faktorler = [
            "Jeopolitik riskler",
            "Enflasyon baskısı"
        ]
        
        # Haberlere göre olumlu/olumsuz faktörleri değiştir
        if market_news:
            for news in market_news[:2]:
                if "büyüme" in news.get("title", "").lower() or "artış" in news.get("title", "").lower():
                    olumlu_faktorler.append(news.get("title"))
                elif "düşüş" in news.get("title", "").lower() or "sorun" in news.get("title", "").lower():
                    olumsuz_faktorler.append(news.get("title"))
        
        sentiment_text = f"""Piyasa genel olarak {market_mood.lower()} bir görünüm sergilemektedir. Son dönemde BIST100 endeksi {month_change:.2f}% değişimle {'yükseliş' if month_change > 0 else 'düşüş'} eğiliminde.

Olumlu faktörler:
- {olumlu_faktorler[0]}
- {olumlu_faktorler[1]}

Olumsuz faktörler:
- {olumsuz_faktorler[0]}
- {olumsuz_faktorler[1]}

Genel öneriler: Piyasalarda {'iyimserlik' if month_change > 0 else 'temkinli duruş'} hakim. {'Yükseliş trendinin sürmesi bekleniyor.' if month_change > 3 else 'Yüksek volatilite nedeniyle temkinli olunmalı.' if volatility > 2.5 else 'Ekonomik verilerin iyileşmesi durumunda yukarı yönlü hareket görülebilir.'}"""
        
        if log_container:
            log_container.success("Piyasa analizi tamamlandı!")
        
        return sentiment_text, sentiment_data
        
    except Exception as e:
        if log_container:
            log_container.error(f"Piyasa analizi hatası: {str(e)}")
        
        # Hata durumunda varsayılan değerler
        default_text = """Piyasa genel olarak nötr bir görünüm sergilemektedir. Son dönemde BIST100 endeksi yatay bir seyir izliyor.

Olumlu faktörler:
- Ekonomik istikrar sinyalleri
- Şirket karlarında artış

Olumsuz faktörler:
- Jeopolitik riskler
- Enflasyon baskısı

Genel öneriler: Piyasalarda temkinli iyimserlik hakim. Ekonomik verilerin iyileşmesi durumunda yukarı yönlü hareket görülebilir."""
        
        default_data = {
            'market_mood': 'Nötr',
            'confidence': 75,
            'trend_strength': 50,
            'volatility_expectation': 'Orta',
            'overall_recommendation': 'Tut'
        }
        
        return default_text, default_data

def ai_stock_analysis(gemini_pro, stock_symbol, stock_data):
    """
    Belirli bir hisse senedi için AI analizi yapar
    """
    return _ai_stock_analysis(gemini_pro, stock_symbol, stock_data)

def ai_price_prediction(gemini_pro, stock_symbol, stock_data):
    """
    Hisse senedi için fiyat tahmini yapar
    """
    return _ai_price_prediction(gemini_pro, stock_symbol, stock_data)

def ai_sector_analysis(gemini_pro, stock_symbol):
    """
    Sektör analizi yapar
    """
    return _ai_sector_analysis(gemini_pro, stock_symbol)

def ai_portfolio_recommendation(gemini_pro, budget=10000):
    """
    Portföy önerileri sunar
    """
    return _ai_portfolio_recommendation(gemini_pro, budget)

def ai_technical_interpretation(gemini_pro, stock_data):
    """
    Teknik analiz sonuçlarını yorumlar
    """
    return _ai_technical_interpretation(gemini_pro, stock_data)

def ai_portfolio_analysis(gemini_pro, portfolio_data):
    """
    Portföy analizini yapay zeka ile gerçekleştirir.
    
    Args:
        gemini_pro: Yapay zeka modeli
        portfolio_data (dict): Portföy performans verileri
        
    Returns:
        dict: Yapay zeka analiz sonuçları
    """
    try:
        return _ai_portfolio_analysis(gemini_pro, portfolio_data)
    except Exception as e:
        # API hatası veya bağlantı sorunu durumunda basit bir analiz dön
        print(f"AI portföy analizi hatası: {str(e)}")
        
        # Basit analiz oluştur
        stocks_data = portfolio_data.get("stocks", [])
        total_gain_loss = portfolio_data.get("total_gain_loss", 0)
        
        # Kazançlı ve zarardaki hisseleri ayır
        profitable_stocks = [s for s in stocks_data if s["gain_loss"] > 0]
        loss_stocks = [s for s in stocks_data if s["gain_loss"] < 0]
        
        # En iyi ve en kötü performansa sahip hisseleri bul
        best_stock = max(stocks_data, key=lambda x: x["gain_loss_percentage"]) if stocks_data else None
        worst_stock = min(stocks_data, key=lambda x: x["gain_loss_percentage"]) if stocks_data else None
        
        # Genel durum belirle
        if total_gain_loss > 0:
            status = "pozitif"
            advice = "Portföyünüz kazançta. Kâr realizasyonu yapabilir veya güçlü hisselerde pozisyonunuzu artırabilirsiniz."
        elif total_gain_loss < 0:
            status = "negatif"
            advice = "Portföyünüz zararda. Zarardaki hisseleri gözden geçirin ve performansı iyi olan sektörlere yönelebilirsiniz."
        else:
            status = "nötr"
            advice = "Portföyünüz başabaş durumda. Fırsatları değerlendirerek daha güçlü hisselere yönelebilirsiniz."
        
        # Analiz sonuçlarını oluştur
        return {
            "status": status,
            "summary": f"Portföyünüzde {len(stocks_data)} adet hisse bulunuyor. {len(profitable_stocks)} tanesi kârda, {len(loss_stocks)} tanesi zararda.",
            "best_performer": best_stock["symbol"] if best_stock else "Yok",
            "worst_performer": worst_stock["symbol"] if worst_stock else "Yok",
            "best_percentage": best_stock["gain_loss_percentage"] if best_stock else 0,
            "worst_percentage": worst_stock["gain_loss_percentage"] if worst_stock else 0,
            "recommendations": advice,
            "details": "Yapay zeka analizi şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin."
        }

def ai_portfolio_optimization(gemini_pro, portfolio_data, sector_distribution):
    """
    Portföy optimizasyon önerileri sunar.
    
    Args:
        gemini_pro: Yapay zeka modeli
        portfolio_data (dict): Portföy performans verileri
        sector_distribution (dict): Sektör dağılımı verileri
        
    Returns:
        dict: Yapay zeka optimizasyon önerileri
    """
    try:
        return _ai_portfolio_optimization(gemini_pro, portfolio_data, sector_distribution)
    except Exception as e:
        # API hatası veya bağlantı sorunu durumunda basit öneriler dön
        print(f"AI portföy optimizasyonu hatası: {str(e)}")
        
        # Basit optimizasyon önerileri
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
        
        return {
            "general_recommendations": recommendations,
            "increase_positions": increase_recommendations,
            "decrease_positions": decrease_recommendations,
            "sector_recommendations": sector_recommendations,
            "details": "Yapay zeka optimizasyon önerileri şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin."
        }

def ai_sector_recommendation(gemini_pro):
    """
    Yatırım için önerilen sektörleri analiz eder.
    
    Args:
        gemini_pro: Yapay zeka modeli
        
    Returns:
        dict: Önerilen sektörler ve nedenleri
    """
    try:
        return _ai_sector_recommendation(gemini_pro)
    except Exception as e:
        # API hatası veya bağlantı sorunu durumunda varsayılan öneriler
        print(f"AI sektör önerisi hatası: {str(e)}")
        
        # Türkiye'de genellikle güçlü sektörler
        default_sectors = {
            "Bankacılık": "Türkiye'de güçlü temellere sahip bir sektör, ekonomik büyüme ile birlikte gelişim potansiyeli taşır.",
            "Enerji": "Artan enerji ihtiyacı ve yenilenebilir enerji yatırımları sektöre ivme kazandırıyor.",
            "Perakende": "Tüketim alışkanlıklarının değişmesi ve e-ticaretin büyümesi ile potansiyel barındırıyor.",
            "İnşaat": "Altyapı projeleri ve kentsel dönüşüm çalışmaları sektöre canlılık katıyor.",
            "Teknoloji": "Dijital dönüşüm ve yazılım sektörünün büyümesi ile yüksek potansiyel taşıyor.",
            "Sağlık": "Medikal cihazlar ve ilaç sektörü büyüme potansiyeli taşıyor."
        }
        
        return {
            "recommended_sectors": default_sectors,
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "details": "Yapay zeka sektör önerileri şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin."
        } 