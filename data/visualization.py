import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

def create_stock_chart(df, stock_symbol):
    """
    Hisse senedi verilerini ve teknik göstergeleri içeren interaktif bir Plotly grafiği oluşturur.
    
    Parametreler:
        df (pandas.DataFrame): Hisse senedi verisi ve teknik göstergeleri içeren veri çerçevesi
        stock_symbol (str): Hisse senedi sembolü
    
    Döndürür:
        plotly.graph_objects.Figure: Oluşturulan grafik 
    """
    # İlk saygı diye verileri bir kopya yapalım
    df = df.copy()
    
    # Verilerin index'inin DatetimeIndex olduğundan emin olalım
    if not isinstance(df.index, pd.DatetimeIndex):
        df['Date'] = pd.to_datetime(df.index)
        df.set_index('Date', inplace=True)
    
    # Alt grafikleri oluşturalım: Fiyat, Hacim ve RSI için
    fig = make_subplots(
        rows=3, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(
            f"{stock_symbol} Hisse Fiyatı ve Teknik Göstergeler", 
            "İşlem Hacmi", 
            "RSI (Göreceli Güç Endeksi)"
        )
    )
    
    # Mum grafiğini ekle
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Fiyat",
            increasing_line_color='green', 
            decreasing_line_color='red'
        ),
        row=1, col=1
    )
    
    # Hareketli ortalamaları ekle
    if 'SMA20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SMA20'],
                name="SMA 20",
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
    
    if 'SMA50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SMA50'],
                name="SMA 50",
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
    
    if 'SMA200' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SMA200'],
                name="SMA 200",
                line=dict(color='purple', width=1)
            ),
            row=1, col=1
        )
    
    # Bollinger Bantlarını ekle
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns and 'BB_Middle' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Upper'],
                name="Bollinger Üst",
                line=dict(color='rgba(0,128,0,0.3)', width=1),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Lower'],
                name="Bollinger Alt",
                line=dict(color='rgba(0,128,0,0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(0,128,0,0.05)',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # İşlem hacmi grafiğini ekle
    colors = ['green' if row['Close'] >= row['Open'] else 'red' for _, row in df.iterrows()]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name="Hacim",
            marker_color=colors,
            opacity=0.8
        ),
        row=2, col=1
    )
    
    # Hacim ortalamasını ekle
    if 'Volume_SMA20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Volume_SMA20'],
                name="Hacim Ort. (20)",
                line=dict(color='blue', width=1)
            ),
            row=2, col=1
        )
    
    # RSI grafiğini ekle
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI'],
                name="RSI",
                line=dict(color='blue', width=1)
            ),
            row=3, col=1
        )
        
        # RSI aşırı alım/satım çizgileri
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # Grafik düzeni ayarları
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # X-ekseni format ayarları
    fig.update_xaxes(
        rangeslider_visible=False,
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # Hafta sonlarını gizle
        ]
    )
    
    # Y-ekseni başlığı
    fig.update_yaxes(title_text="Fiyat (TL)", row=1, col=1)
    fig.update_yaxes(title_text="Hacim", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    
    return fig

def create_comparison_chart(symbols, period="6mo"):
    """
    Farklı hisseleri karşılaştırmak için grafik oluşturur
    
    Parametreler:
        symbols (list): Karşılaştırılacak hisse senetlerinin listesi
        period (str): Veri dönemi (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
    Döndürür:
        plotly.graph_objects.Figure: Oluşturulan grafik
    """
    from data.stock_data import get_stock_data
    
    fig = go.Figure()
    
    # Her bir hisse için veriyi al ve grafiğe ekle
    for symbol in symbols:
        # Veriyi al
        df = get_stock_data(symbol, period=period)
        
        if df is not None and not df.empty:
            # İlk günün değerine göre normalize et
            normalized_data = df['Close'] / df['Close'].iloc[0] * 100
            
            # Grafiğe ekle
            fig.add_trace(go.Scatter(
                x=df.index,
                y=normalized_data,
                mode='lines',
                name=symbol.replace('.IS', '')
            ))
    
    # Grafik başlığı ve etiketler
    fig.update_layout(
        title="Hisse Senedi Performans Karşılaştırması (Normalize: 100)",
        xaxis_title="Tarih",
        yaxis_title="Normalize Değer (Başlangıç: 100)",
        legend_title="Hisseler",
        template="plotly_white"
    )
    
    return fig

def create_portfolio_pie_chart(portfolio_data):
    """
    Portföy dağılımını gösteren pasta grafik oluşturur
    
    Args:
        portfolio_data (list): Portföy hisselerinin listesi, her biri bir sözlük:
                              [{"symbol": "THYAO", "current_value": 5000, ...}, ...]
    
    Returns:
        plotly.graph_objects.Figure: Oluşturulan pasta grafik
    """
    if not portfolio_data:
        # Boş veri için boş grafik döndür
        fig = go.Figure()
        fig.update_layout(
            title="Portföy Dağılımı (Veri Yok)",
            template="plotly_white"
        )
        return fig
    
    # Sembolleri ve değerleri ayır
    symbols = [item["symbol"] for item in portfolio_data]
    values = [item["current_value"] for item in portfolio_data]
    
    # Pasta grafiği oluştur
    fig = px.pie(
        names=symbols,
        values=values,
        title="Portföy Dağılımı",
        hover_data=[
            [f"{item['current_value']:.2f} TL ({(item['current_value']/sum(values)*100):.1f}%)" for item in portfolio_data]
        ],
        labels={'names': 'Hisse', 'values': 'Değer (TL)'}
    )
    
    # Grafik ayarları
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        marker=dict(line=dict(color='#FFFFFF', width=2))
    )
    
    fig.update_layout(
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=0, xanchor="center", x=0.5)
    )
    
    return fig

def create_portfolio_sector_chart(sector_values):
    """
    Portföyün sektör dağılımını gösteren pasta grafik oluşturur
    
    Args:
        sector_values (dict): Sektör dağılımı (sektör adı -> değer)
    
    Returns:
        plotly.graph_objects.Figure: Oluşturulan pasta grafik
    """
    if not sector_values:
        # Boş veri için boş grafik döndür
        fig = go.Figure()
        fig.update_layout(
            title="Sektör Dağılımı (Veri Yok)",
            template="plotly_white"
        )
        return fig
    
    # Sektörleri ve değerleri ayır
    sectors = list(sector_values.keys())
    values = list(sector_values.values())
    
    # Pasta grafiği oluştur
    fig = px.pie(
        names=sectors,
        values=values,
        title="Portföy Sektör Dağılımı",
        hover_data=[
            [f"{value:.2f} TL ({(value/sum(values)*100):.1f}%)" for value in values]
        ],
        labels={'names': 'Sektör', 'values': 'Değer (TL)'}
    )
    
    # Grafik ayarları
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        marker=dict(line=dict(color='#FFFFFF', width=2))
    )
    
    fig.update_layout(
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=0, xanchor="center", x=0.5)
    )
    
    return fig

def create_portfolio_performance_chart(portfolio_data):
    """
    Portföy performansını gösteren çubuk grafik oluşturur
    
    Args:
        portfolio_data (list): Portföy hisselerinin listesi, her biri bir sözlük:
                              [{"symbol": "THYAO", "gain_loss_percentage": 5.2, ...}, ...]
    
    Returns:
        plotly.graph_objects.Figure: Oluşturulan çubuk grafik
    """
    if not portfolio_data:
        # Boş veri için boş grafik döndür
        fig = go.Figure()
        fig.update_layout(
            title="Portföy Performansı (Veri Yok)",
            template="plotly_white"
        )
        return fig
    
    # Verileri hazırla
    symbols = [item["symbol"] for item in portfolio_data]
    gain_loss_percentages = [item["gain_loss_percentage"] for item in portfolio_data]
    
    # Renkleri ayarla: Kazanç yeşil, kayıp kırmızı
    colors = ['green' if p >= 0 else 'red' for p in gain_loss_percentages]
    
    # Çubuk grafik oluştur
    fig = go.Figure()
    
    # Yüzdelik değerleri formatla (+ veya - işaretini ekleyerek)
    formatted_percentages = []
    for p in gain_loss_percentages:
        if p > 0:
            formatted_percentages.append(f"+{p:.2f}%")
        else:
            formatted_percentages.append(f"{p:.2f}%")
    
    fig.add_trace(
        go.Bar(
            x=symbols,
            y=gain_loss_percentages,
            marker_color=colors,
            text=formatted_percentages,
            textposition='outside',
            textfont=dict(size=12, color='black')
        )
    )
    
    # Grafik ayarları
    fig.update_layout(
        title="Portföydeki Hisselerin Performansı (%)",
        xaxis_title="Hisseler",
        yaxis_title="Kazanç/Kayıp (%)",
        template="plotly_white",
        height=500,
        uniformtext_minsize=10,
        uniformtext_mode='hide',
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    # Eksenleri ayarla
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.1)',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='black'
    )
    
    return fig

def create_portfolio_history_chart(transactions, symbol=None):
    """
    Portföy işlem geçmişini gösteren zaman serisi grafik oluşturur
    
    Args:
        transactions (list): İşlem kayıtlarının listesi
        symbol (str, optional): Filtrelenecek hisse sembolü
    
    Returns:
        plotly.graph_objects.Figure: Oluşturulan grafik
    """
    if not transactions:
        # Boş veri için boş grafik döndür
        fig = go.Figure()
        fig.update_layout(
            title="Portföy İşlem Geçmişi (Veri Yok)",
            template="plotly_white"
        )
        return fig
    
    # Verileri filtrele
    if symbol:
        transactions = [t for t in transactions if t["symbol"] == symbol]
    
    # Verileri tarihe göre sırala
    transactions = sorted(transactions, key=lambda x: x["transaction_date"])
    
    # Zaman serisi grafiği için veriler
    dates = [datetime.strptime(t["transaction_date"], "%Y-%m-%d") for t in transactions]
    prices = [t["price"] for t in transactions]
    symbols = [t["symbol"] for t in transactions]
    types = [t["transaction_type"] for t in transactions]
    quantities = [t["quantity"] for t in transactions]
    
    # İşlem tipine göre renk belirle
    colors = ['green' if t == "ALIŞ" else 'red' for t in types]
    
    # Zaman serisi grafiği oluştur
    fig = go.Figure()
    
    # Her sembol için ayrı çizgi ekle
    unique_symbols = set(symbols)
    for sym in unique_symbols:
        symbol_indices = [i for i, s in enumerate(symbols) if s == sym]
        symbol_dates = [dates[i] for i in symbol_indices]
        symbol_prices = [prices[i] for i in symbol_indices]
        
        fig.add_trace(
            go.Scatter(
                x=symbol_dates,
                y=symbol_prices,
                name=sym,
                mode='lines+markers',
                line=dict(dash='dot'),
                connectgaps=False
            )
        )
    
    # İşlem noktalarını ekle
    for i in range(len(dates)):
        fig.add_trace(
            go.Scatter(
                x=[dates[i]],
                y=[prices[i]],
                mode='markers',
                marker=dict(color=colors[i], size=12, symbol='circle'),
                name=f"{symbols[i]} {types[i]}",
                text=f"{symbols[i]}: {types[i]} {quantities[i]} adet @ {prices[i]} TL",
                hoverinfo='text',
                showlegend=False
            )
        )
    
    # Grafik ayarları
    if symbol:
        title = f"{symbol} İşlem Geçmişi"
    else:
        title = "Portföy İşlem Geçmişi"
        
    fig.update_layout(
        title=title,
        xaxis_title="Tarih",
        yaxis_title="İşlem Fiyatı (TL)",
        template="plotly_white",
        height=600,
        hovermode="closest"
    )
    
    return fig

def create_portfolio_investment_chart(transactions):
    """
    Portföydeki toplam yatırım değişimini gösteren kümülatif grafik oluşturur
    
    Args:
        transactions (list): İşlem kayıtlarının listesi
    
    Returns:
        plotly.graph_objects.Figure: Oluşturulan grafik
    """
    if not transactions:
        # Boş veri için boş grafik döndür
        fig = go.Figure()
        fig.update_layout(
            title="Portföy Yatırım Değişimi (Veri Yok)",
            template="plotly_white"
        )
        return fig
    
    # Verileri tarihe göre sırala
    transactions = sorted(transactions, key=lambda x: x["transaction_date"])
    
    # Tarih ve tutar verileri
    dates = [datetime.strptime(t["transaction_date"], "%Y-%m-%d") for t in transactions]
    amounts = []
    cumulative = 0
    
    for t in transactions:
        # ALIŞ işlemi pozitif, SATIŞ işlemi negatif etki yapar
        if t["transaction_type"] == "ALIŞ":
            change = t["total_amount"]
        else:  # SATIŞ
            change = -t["total_amount"]
        
        cumulative += change
        amounts.append(cumulative)
    
    # Kümülatif grafik oluştur
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=amounts,
            mode='lines+markers',
            line=dict(width=2),
            fill='tozeroy'
        )
    )
    
    # Grafik ayarları
    fig.update_layout(
        title="Portföy Yatırım Değişimi",
        xaxis_title="Tarih",
        yaxis_title="Toplam Yatırım (TL)",
        template="plotly_white",
        height=500
    )
    
    return fig

def create_portfolio_prediction_chart(portfolio_data, predictions):
    """
    Portföydeki hisselerin fiyat tahminlerini gösteren grafik oluşturur
    
    Args:
        portfolio_data (list): Portföy hisselerinin listesi
        predictions (dict): Hisselerin fiyat tahminleri:
                            {"THYAO": {"prediction": 50.2, "confidence": 0.8, ...}, ...}
    
    Returns:
        plotly.graph_objects.Figure: Oluşturulan grafik
    """
    if not portfolio_data or not predictions:
        # Boş veri için boş grafik döndür
        fig = go.Figure()
        fig.update_layout(
            title="Hisse Tahminleri (Veri Yok)",
            template="plotly_white"
        )
        return fig
    
    # Sadece tahminleri olan hisseleri filtrele
    symbols = []
    current_prices = []
    predicted_prices = []
    changes = []
    confidences = []
    
    for item in portfolio_data:
        symbol = item["symbol"]
        if symbol in predictions:
            symbols.append(symbol)
            current_prices.append(item["current_price"])
            predicted_prices.append(predictions[symbol]["prediction"])
            changes.append((predictions[symbol]["prediction"] - item["current_price"]) / item["current_price"] * 100)
            confidences.append(predictions[symbol]["confidence"])
    
    if not symbols:
        # Tahmin bulunmayan hisseler için boş grafik döndür
        fig = go.Figure()
        fig.update_layout(
            title="Hisse Tahminleri (Tahmin Yok)",
            template="plotly_white"
        )
        return fig
    
    # Çubuk grafik oluştur
    fig = go.Figure()
    
    # Mevcut fiyatlar
    fig.add_trace(
        go.Bar(
            x=symbols,
            y=current_prices,
            name="Mevcut Fiyat",
            marker_color='blue'
        )
    )
    
    # Tahmin edilen fiyatlar
    fig.add_trace(
        go.Bar(
            x=symbols,
            y=predicted_prices,
            name="Tahmin Edilen Fiyat",
            marker_color='orange',
            text=[f"{p:.2f} TL<br>(%{c:.1f})" for p, c in zip(predicted_prices, changes)],
            textposition='auto'
        )
    )
    
    # Grafik ayarları
    fig.update_layout(
        title="Portföy Hisselerinin Fiyat Tahminleri",
        xaxis_title="Hisseler",
        yaxis_title="Fiyat (TL)",
        template="plotly_white",
        height=500,
        barmode='group'
    )
    
    return fig 