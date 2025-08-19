import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import numpy as np

def create_stock_chart(df, symbol, indicators=True):
    """
    Hisse senedi için interaktif grafik oluşturur
    """
    fig = make_subplots(
        rows=3, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{symbol} Fiyat Grafiği", "Hacim", "Göstergeler")
    )
    
    # Mum grafiği ekle
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'], 
            high=df['High'],
            low=df['Low'], 
            close=df['Close'],
            name="Mum"
        ),
        row=1, col=1
    )
    
    # Göstergeleri ekle
    if indicators:
        # SMA çizgileri
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], mode='lines', name='SMA20', line=dict(color='blue', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA50', line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], mode='lines', name='SMA200', line=dict(color='red', width=1)), row=1, col=1)
        
        # Bollinger Bantları
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Band'], mode='lines', name='Üst Bant', line=dict(color='rgba(250,0,0,0.3)', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Band'], mode='lines', name='Alt Bant', line=dict(color='rgba(250,0,0,0.3)', width=1), fill='tonexty', fillcolor='rgba(250,0,0,0.05)'), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple', width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=[70] * len(df), mode='lines', name='Aşırı Alım', line=dict(color='red', width=1, dash='dash')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=[30] * len(df), mode='lines', name='Aşırı Satım', line=dict(color='green', width=1, dash='dash')), row=3, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='blue', width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='Sinyal', line=dict(color='red', width=1)), row=3, col=1)
    
    # Hacim grafiği
    colors = ['green' if row['Close'] >= row['Open'] else 'red' for i, row in df.iterrows()]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Hacim', marker_color=colors),
        row=2, col=1
    )
    
    # Grafik düzenini ayarla
    fig.update_layout(
        title=f"{symbol} Hisse Senedi Analizi",
        xaxis_title="Tarih",
        yaxis_title="Fiyat (TL)",
        height=800,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Y-eksenlerini güncelle
    fig.update_yaxes(title_text="Fiyat (TL)", row=1, col=1)
    fig.update_yaxes(title_text="Hacim", row=2, col=1)
    fig.update_yaxes(title_text="Osilatörler", row=3, col=1)
    
    return fig

def detect_chart_patterns(df):
    """
    Grafik desenlerini tespit eder
    """
    patterns = {}
    
    # Son 10 günde desenleri kontrol et
    window = min(len(df), 30)
    
    # Destek ve Direnç seviyeleri
    recent_df = df.tail(window)
    
    # Fibonacci Retracement Seviyeleri için hesaplama
    max_price = recent_df['High'].max()
    min_price = recent_df['Low'].min()
    diff = max_price - min_price
    
    patterns['fibonacci_levels'] = {
        '0.0': min_price,
        '0.236': min_price + 0.236 * diff,
        '0.382': min_price + 0.382 * diff,
        '0.5': min_price + 0.5 * diff,
        '0.618': min_price + 0.618 * diff,
        '0.786': min_price + 0.786 * diff,
        '1.0': max_price
    }
    
    # Çift Tepe
    if len(df) >= 15:
        highs = df['High'].rolling(5).max()
        if any(highs.diff().tail(10) < 0):
            patterns['double_top'] = True
    
    # Çift Dip
    if len(df) >= 15:
        lows = df['Low'].rolling(5).min()
        if any(lows.diff().tail(10) > 0):
            patterns['double_bottom'] = True
    
    # Baş ve Omuzlar
    # Bu basit bir yaklaşım, gerçek bir sistemde daha karmaşık algoritmalar kullanılmalı
    if len(df) >= 20:
        head_shoulder = False
        for i in range(5, len(df)-5):
            left = df.iloc[i-5:i]['High'].max()
            head = df.iloc[i:i+5]['High'].max()
            right = df.iloc[i+5:i+10]['High'].max()
            
            if head > left and head > right and abs(left - right) / left < 0.1:
                head_shoulder = True
                break
                
        if head_shoulder:
            patterns['head_and_shoulders'] = True
    
    # Yükseliş/Düşüş Kanalı
    # Basit bir doğrusal regresyon kanalı bulmaya çalışıyoruz
    if len(df) >= 20:
        x = range(len(df.tail(20)))
        y_close = df['Close'].tail(20).values
        
        # İstatistiksel analizler için numpy kullan
        slope, intercept = np.polyfit(x, y_close, 1)
        
        if slope > 0.01:
            patterns['uptrend_channel'] = True
        elif slope < -0.01:
            patterns['downtrend_channel'] = True
    
    # Son kapanış fiyatına göre destek/direnç seviyeleri
    last_close = df['Close'].iloc[-1]
    
    patterns['last_close'] = last_close
    patterns['support_levels'] = [round(level, 2) for level in recent_df['Low'].nsmallest(3).values if level < last_close]
    patterns['resistance_levels'] = [round(level, 2) for level in recent_df['High'].nlargest(3).values if level > last_close]
    
    return patterns 