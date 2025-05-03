import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data.stock_data import get_stock_data, get_company_info, get_stock_news
from analysis.indicators import calculate_indicators
from ai.predictions import ml_price_prediction, backtest_models
from data.utils import save_analysis_result, load_analysis_results

def render_ml_prediction_tab():
    """
    Makine öğrenimi tahmin sekmesini oluşturur
    """
    st.header("Makine Öğrenimi Fiyat Tahmini")
    
    col1, col2, col3 = st.columns([3, 1, 2])
    
    with col1:
        stock_symbol = st.text_input("Hisse Senedi Kodu (örn: THYAO)", "THYAO", key="ml_stock_symbol")
        
    with col2:
        days_to_predict = st.selectbox(
            "Tahmin Dönemi",
            [7, 14, 30, 60, 90],
            index=2  # Default 30 gün
        )
    
    with col3:
        st.write("Analiz Yap")
        analyze_btn = st.button("Tüm Modellerle Tahmin Et")
    
    # Geçmiş tahminleri yükle - ancak bunları gösterme
    previous_results = load_analysis_results(analysis_type="ml")
    
    if analyze_btn or ('ml_last_symbol' in st.session_state and st.session_state.ml_last_symbol == stock_symbol):
        # Symbol validation and formatting
        stock_symbol = stock_symbol.upper().strip()
        st.session_state.ml_last_symbol = stock_symbol
        
        with st.spinner(f"{stock_symbol} için makine öğrenimi tahmini yapılıyor..."):
            try:
                # Get stock data
                df = get_stock_data(stock_symbol, "1y" if days_to_predict <= 30 else "2y")
                
                if len(df) == 0:
                    st.error(f"{stock_symbol} için veri bulunamadı. Lütfen geçerli bir hisse senedi kodu girdiğinizden emin olun.")
                    return
                
                # Şirket bilgilerini al
                company_info = get_company_info(stock_symbol)
                if company_info:
                    st.subheader(f"{company_info.get('name', stock_symbol)} ({stock_symbol})")
                
                # Calculate indicators
                df_with_indicators = calculate_indicators(df)
                
                # Kullanılacak modeller
                model_types = ["RandomForest", "XGBoost", "LightGBM", "Ensemble", "Hibrit Model"]
                
                # Her model için tahmin yap
                all_predictions = {}
                
                # Progress bar
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                for i, model_type in enumerate(model_types):
                    progress_text.text(f"{model_type} modeli çalıştırılıyor...")
                    progress_bar.progress((i) / len(model_types))
                    
                    # ML price prediction with current model
                    prediction = ml_price_prediction(stock_symbol, df_with_indicators, days_to_predict=days_to_predict, model_type=model_type)
                    
                    if prediction is not None:
                        all_predictions[model_type] = prediction
                    
                    progress_bar.progress((i + 1) / len(model_types))
                
                progress_text.text("Tüm modeller tamamlandı!")
                progress_bar.progress(1.0)
                
                if not all_predictions:
                    st.error("Tahmin yapılırken bir hata oluştu. Lütfen tekrar deneyin.")
                    return
                
                # Tahmin sonuçlarını göster
                st.subheader(f"{stock_symbol} ML Fiyat Tahminleri Karşılaştırması")
                
                # Modelleri sonuç tablosu oluştur
                comparison_data = []
                for model_type, prediction in all_predictions.items():
                    current_price = prediction['last_price']
                    predicted_price = prediction['prediction_30d'] if days_to_predict >= 30 else prediction['prediction_7d']
                    
                    if predicted_price is not None:
                        change_pct = ((predicted_price - current_price) / current_price) * 100
                        
                        # Tahmin metriklerini ekle
                        comparison_data.append({
                            'Model': model_type,
                            'Mevcut Fiyat': f"{current_price:.2f} TL",
                            f'{days_to_predict} Gün Tahmini': f"{predicted_price:.2f} TL",
                            'Değişim (%)': f"{change_pct:.2f}%",
                            'R² Skoru': f"{prediction['r2_score']:.4f}",
                            'Güven': f"%{prediction['confidence']:.1f}",
                            'Trend': prediction['trend'],
                            'change_value': change_pct,  # Sıralama için
                            'r2_value': prediction['r2_score']  # Sıralama için
                        })
                
                # Karşılaştırma tablosunu göster
                comparison_df = pd.DataFrame(comparison_data)
                
                # Değişim oranına göre sırala
                comparison_df = comparison_df.sort_values('r2_value', ascending=False)
                
                # Gösterim için gereksiz kolonları kaldır
                display_df = comparison_df.drop(['change_value', 'r2_value'], axis=1)
                
                # Renklendirme için stil tanımla
                def color_negative_red(val):
                    try:
                        if '%' in val:
                            # Değişim değeri
                            val_num = float(val.replace('%', ''))
                            return 'color: red' if val_num < 0 else 'color: green'
                    except:
                        pass
                    return ''
                
                st.dataframe(display_df.style.applymap(color_negative_red, subset=['Değişim (%)']), use_container_width=True)
                
                # Farklı modellerin sonuçlarını görsel olarak karşılaştır
                st.subheader("Model Tahminleri Karşılaştırması")
                
                # Plotly ile tahmin grafiği oluştur
                fig = go.Figure()
                
                # Gerçek fiyatları ekle
                fig.add_trace(
                    go.Candlestick(
                        x=df.index[-30:],
                        open=df['Open'][-30:],
                        high=df['High'][-30:],
                        low=df['Low'][-30:],
                        close=df['Close'][-30:],
                        name="Geçmiş Fiyatlar"
                    )
                )
                
                # Her modelin tahminini farklı renk ve çizgi stiliyle ekle
                colors = ['rgba(0, 128, 255, 0.8)', 'rgba(255, 0, 0, 0.8)', 'rgba(0, 255, 0, 0.8)', 
                          'rgba(255, 165, 0, 0.8)', 'rgba(128, 0, 128, 0.8)']
                
                for i, (model_type, prediction) in enumerate(all_predictions.items()):
                    predictions_df = prediction['predictions_df']
                    
                    # Tahmin çizgisi
                    fig.add_trace(
                        go.Scatter(
                            x=predictions_df.index,
                            y=predictions_df['Predicted Price'],
                            mode='lines',
                            line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                            name=f'{model_type} Tahmini'
                        )
                    )
                
                # Grafik düzeni
                fig.update_layout(
                    title=f"{stock_symbol} - Farklı ML Modelleri Tahmin Karşılaştırması",
                    xaxis_rangeslider_visible=False,
                    height=600,
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                
                # Grafik gösterimi
                st.plotly_chart(fig, use_container_width=True)
                
                # Backtest sonuçlarını göster
                st.subheader("Model Backtest Performansı")
                
                with st.spinner("Backtest yapılıyor..."):
                    # Backtest için önceki verileri al ve modelleri test et
                    backtest_length = min(180, len(df) - 30)  # En fazla 180 gün veya veri uzunluğunun izin verdiği kadar
                    backtest_periods = [7, 14, 30]  # Test edilecek tahmin dönemleri
                    
                    # Her model için backtest yap
                    backtest_results = {}
                    
                    for model_type in model_types:
                        # Her tahmin dönemi için backtest
                        model_results = {}
                        for period in backtest_periods:
                            if period <= backtest_length:
                                # Backtest sonuçlarını al
                                period_results = backtest_models(df_with_indicators, period=period, model_type=model_type, test_size=backtest_length)
                                model_results[period] = period_results
                        
                        backtest_results[model_type] = model_results
                    
                    # Backtest sonuç tablosunu oluştur
                    backtest_table_data = []
                    
                    for model_type, periods in backtest_results.items():
                        # Her dönem için metrikleri hesapla
                        model_row = {'Model': model_type}
                        
                        for period, results in periods.items():
                            mae = results.get('mae', np.nan)
                            rmse = results.get('rmse', np.nan)
                            accuracy = results.get('accuracy', np.nan) * 100  # Yüzde olarak
                            
                            model_row[f'{period}g MAE'] = f"{mae:.2f}" if not np.isnan(mae) else "N/A"
                            model_row[f'{period}g RMSE'] = f"{rmse:.2f}" if not np.isnan(rmse) else "N/A"
                            model_row[f'{period}g Doğruluk'] = f"%{accuracy:.1f}" if not np.isnan(accuracy) else "N/A"
                            
                            # Görsel gösterim için kullanılacak değerleri de ekle
                            model_row[f'{period}g_acc_val'] = accuracy
                        
                        backtest_table_data.append(model_row)
                    
                    # Tabloyu oluştur
                    backtest_df = pd.DataFrame(backtest_table_data)
                    # En yüksek doğruluk oranına göre sırala
                    if f'30g_acc_val' in backtest_df.columns:
                        backtest_df = backtest_df.sort_values(f'30g_acc_val', ascending=False)
                    
                    # Gösterim için gereksiz sütunları kaldır
                    display_cols = [col for col in backtest_df.columns if not col.endswith('_val')]
                    display_backtest_df = backtest_df[display_cols]
                    
                    st.dataframe(display_backtest_df, use_container_width=True)
                    
                    # Backtest görselleştirme
                    selected_period = st.selectbox(
                        "Backtest Dönemi",
                        backtest_periods,
                        index=len(backtest_periods)-1  # Varsayılan olarak en uzun dönem
                    )
                    
                    # Backtest grafiğini oluştur
                    if all(selected_period in periods for model_type, periods in backtest_results.items()):
                        backtest_fig = make_subplots(specs=[[{"secondary_y": False}]])
                        
                        # Gerçek fiyatları ekle
                        backtest_fig.add_trace(
                            go.Scatter(
                                x=df.index[-backtest_length:],
                                y=df['Close'][-backtest_length:],
                                mode='lines',
                                line=dict(color='black', width=2),
                                name="Gerçek Fiyatlar"
                            )
                        )
                        
                        # Her modelin tahminlerini ekle
                        colors = ['rgba(0, 128, 255, 0.8)', 'rgba(255, 0, 0, 0.8)', 'rgba(0, 255, 0, 0.8)', 
                                'rgba(255, 165, 0, 0.8)', 'rgba(128, 0, 128, 0.8)']
                        
                        for i, (model_type, periods) in enumerate(backtest_results.items()):
                            if selected_period in periods:
                                predictions = periods[selected_period].get('predictions', None)
                                if predictions is not None and isinstance(predictions, pd.DataFrame):
                                    backtest_fig.add_trace(
                                        go.Scatter(
                                            x=predictions.index,
                                            y=predictions['Predicted'],
                                            mode='lines',
                                            line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                                            name=f'{model_type} Tahmini'
                                        )
                                    )
                        
                        # Grafik düzeni
                        backtest_fig.update_layout(
                            title=f"{stock_symbol} - {selected_period} Günlük Backtest Sonuçları",
                            xaxis_title="Tarih",
                            yaxis_title="Fiyat (TL)",
                            height=600,
                            hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                        )
                        
                        # Grafik gösterimi
                        st.plotly_chart(backtest_fig, use_container_width=True)
                        
                        # Backtest sonuçlarının yorumlanması
                        st.subheader("Backtest Analizi")
                        
                        # En iyi modeli bul
                        best_model = None
                        best_accuracy = -1
                        
                        for model_type, periods in backtest_results.items():
                            if selected_period in periods:
                                accuracy = periods[selected_period].get('accuracy', 0) * 100
                                if accuracy > best_accuracy:
                                    best_accuracy = accuracy
                                    best_model = model_type
                        
                        if best_model:
                            st.markdown(f"**En İyi Performans Gösteren Model:** {best_model} (Doğruluk: %{best_accuracy:.1f})")
                            
                            # Genel analiz
                            avg_accuracy = np.mean([periods[selected_period].get('accuracy', 0) * 100 
                                                for model_type, periods in backtest_results.items() 
                                                if selected_period in periods])
                            
                            if avg_accuracy > 70:
                                performance = "çok iyi"
                            elif avg_accuracy > 60:
                                performance = "iyi"
                            elif avg_accuracy > 50:
                                performance = "orta"
                            else:
                                performance = "zayıf"
                            
                            st.markdown(f"**Ortalama Model Performansı:** %{avg_accuracy:.1f} ({performance})")
                            
                            # Modeller arası karşılaştırma
                            st.markdown("**Modeller Arası Karşılaştırma:**")
                            for model_type, periods in backtest_results.items():
                                if selected_period in periods:
                                    accuracy = periods[selected_period].get('accuracy', 0) * 100
                                    mae = periods[selected_period].get('mae', 0)
                                    rmse = periods[selected_period].get('rmse', 0)
                                    
                                    st.markdown(f"- {model_type}: Doğruluk %{accuracy:.1f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
                            
                            # Yorum
                            st.info(f"""
                            Bu backtest sonuçları, modellerin geçmiş {backtest_length} gün içinde {selected_period} günlük tahminlerde ne kadar başarılı olduğunu gösteriyor. 
                            Backtest doğruluğu, modelin gelecekteki performansı için bir gösterge olabilir, ancak piyasa koşulları değiştikçe modelin performansı da değişebilir.
                            """)
                
                # Her model için detaylı inceleme
                st.subheader("Her Model İçin Detaylı Sonuçlar")
                
                for model_type, prediction in all_predictions.items():
                    with st.expander(f"{model_type} Model Detayları"):
                        col1, col2, col3 = st.columns(3)
                        
                        # Current and predicted prices
                        with col1:
                            current_price = prediction['last_price']
                            predicted_price = prediction['prediction_30d'] if days_to_predict >= 30 else prediction['prediction_7d']
                            
                            if predicted_price is not None:
                                change_pct = ((predicted_price - current_price) / current_price) * 100
                                change_color = "green" if change_pct > 0 else "red"
                                
                                st.markdown(f"<h4 style='text-align: center;'>Fiyat Tahmini ({days_to_predict} Gün)</h4>", unsafe_allow_html=True)
                                st.markdown(f"<p style='text-align: center;'>Mevcut: {current_price:.2f} TL</p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='text-align: center;'>Tahmin: <span style='color: {change_color};'>{predicted_price:.2f} TL</span></p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='text-align: center;'>Değişim: <span style='color: {change_color};'>{change_pct:.2f}%</span></p>", unsafe_allow_html=True)
                        
                        # Model performance
                        with col2:
                            st.markdown(f"<h4 style='text-align: center;'>Model Başarısı</h4>", unsafe_allow_html=True)
                            
                            r2 = prediction['r2_score']
                            r2_pct = max(min(r2 * 100, 100), 0)  # Convert to percentage, capped at 0-100
                            
                            r2_color = "green" if r2_pct > 70 else ("orange" if r2_pct > 40 else "red")
                            
                            st.markdown(f"<p style='text-align: center;'>R² Skoru: <span style='color: {r2_color};'>{r2:.4f}</span></p>", unsafe_allow_html=True)
                            st.markdown(f"<p style='text-align: center;'>Doğruluk: <span style='color: {r2_color};'>%{r2_pct:.1f}</span></p>", unsafe_allow_html=True)
                            
                            confidence = prediction['confidence']
                            # Tahmin güven oranına göre ek bilgi
                            if confidence > 75:
                                confidence_info = "Yüksek güven"
                                confidence_color = "green"
                            elif confidence > 50:
                                confidence_info = "Orta güven"
                                confidence_color = "orange"
                            else:
                                confidence_info = "Düşük güven"
                                confidence_color = "red"
                                
                            st.markdown(f"<p style='text-align: center;'>Güven: <span style='color: {confidence_color};'>%{confidence:.1f} ({confidence_info})</span></p>", unsafe_allow_html=True)
                        
                        # Trend prediction
                        with col3:
                            trend = prediction['trend']
                            trend_color = "green" if trend == "Yükseliş" else "red"
                            
                            st.markdown(f"<h4 style='text-align: center;'>Trend Tahmini</h4>", unsafe_allow_html=True)
                            st.markdown(f"<p style='text-align: center;'>Beklenen Trend: <span style='color: {trend_color};'>{trend}</span></p>", unsafe_allow_html=True)
                            
                            if prediction['prediction_7d'] is not None and days_to_predict > 7:
                                change_7d = ((prediction['prediction_7d'] - current_price) / current_price) * 100
                                color_7d = "green" if change_7d > 0 else "red"
                                st.markdown(f"<p style='text-align: center;'>7 Günlük: <span style='color: {color_7d};'>{prediction['prediction_7d']:.2f} TL ({change_7d:.2f}%)</span></p>", unsafe_allow_html=True)
                        
                        # Show prediction dataframe
                        st.subheader("Günlük Tahmin Verileri")
                        
                        # Format the predictions dataframe
                        predictions_df = prediction['predictions_df'].copy()
                        predictions_df.index = predictions_df.index.strftime('%Y-%m-%d')
                        predictions_df = predictions_df.rename(columns={"Predicted Price": "Tahmin Edilen Fiyat"})
                        
                        # Add daily change column
                        predictions_df['Günlük Değişim (%)'] = predictions_df['Tahmin Edilen Fiyat'].pct_change() * 100
                        predictions_df.loc[predictions_df.index[0], 'Günlük Değişim (%)'] = ((predictions_df.loc[predictions_df.index[0], 'Tahmin Edilen Fiyat'] - current_price) / current_price) * 100
                        
                        # Add cumulative change column
                        predictions_df['Kümülatif Değişim (%)'] = ((predictions_df['Tahmin Edilen Fiyat'] - current_price) / current_price) * 100
                        
                        # Format the dataframe for display
                        display_df = predictions_df.copy()
                        display_df['Tahmin Edilen Fiyat'] = display_df['Tahmin Edilen Fiyat'].map('{:.2f} TL'.format)
                        display_df['Günlük Değişim (%)'] = display_df['Günlük Değişim (%)'].map('{:.2f}%'.format)
                        display_df['Kümülatif Değişim (%)'] = display_df['Kümülatif Değişim (%)'].map('{:.2f}%'.format)
                        
                        # Tekrarlayan indeksleri önlemek için reset_index uygula
                        display_df = display_df.reset_index()
                        display_df = display_df.rename(columns={"index": "Tarih"})
                        
                        # Function to color the values based on positive/negative
                        def color_values(val):
                            if isinstance(val, str) and '%' in val:
                                try:
                                    num_val = float(val.replace('%', ''))
                                    return 'color: green' if num_val > 0 else 'color: red'
                                except:
                                    return ''
                            return ''
                        
                        # Display the styled dataframe
                        st.dataframe(display_df.style.map(color_values, subset=['Günlük Değişim (%)', 'Kümülatif Değişim (%)']), use_container_width=True)
                
                # Modellerin önerilerine dayalı genel bir öneri oluştur
                st.subheader("Nihai Yatırım Önerisi")
                
                # Tahminlerdeki değişim yüzdesine göre genel bir öneri oluştur
                weighted_changes = []
                for model_type, prediction in all_predictions.items():
                    current_price = prediction['last_price']
                    predicted_price = prediction['prediction_30d'] if days_to_predict >= 30 else prediction['prediction_7d']
                    
                    if predicted_price is not None:
                        change_pct = ((predicted_price - current_price) / current_price) * 100
                        r2_weight = prediction['r2_score'] * prediction['confidence'] / 100
                        weighted_changes.append(change_pct * r2_weight)
                
                avg_weighted_change = sum(weighted_changes) / len(weighted_changes) if weighted_changes else 0
                
                # Final recommendation based on weighted average
                if avg_weighted_change > 10:
                    rec_text = "GÜÇLÜ AL"
                    rec_color = "darkgreen"
                elif avg_weighted_change > 3:
                    rec_text = "AL"
                    rec_color = "green"
                elif avg_weighted_change < -10:
                    rec_text = "GÜÇLÜ SAT"
                    rec_color = "darkred"
                elif avg_weighted_change < -3:
                    rec_text = "SAT"
                    rec_color = "red"
                else:
                    rec_text = "TUT"
                    rec_color = "gray"
                
                st.markdown(f"<h3 style='text-align: center; color: {rec_color};'>Modeller Arası Konsensus: {rec_text}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>Ağırlıklı Ortalama Değişim Beklentisi: <span style='color: {rec_color};'>{avg_weighted_change:.2f}%</span></p>", unsafe_allow_html=True)
                
                # Detaylı yatırım önerisi
                st.markdown(f"<p><strong>Tahmin Dönemi:</strong> {days_to_predict} gün</p>", unsafe_allow_html=True)
                
                # Model önerilerinin dağılımı
                model_recs = []
                for model_type, prediction in all_predictions.items():
                    current_price = prediction['last_price']
                    predicted_price = prediction['prediction_30d'] if days_to_predict >= 30 else prediction['prediction_7d']
                    
                    if predicted_price is not None:
                        change_pct = ((predicted_price - current_price) / current_price) * 100
                        if change_pct > 10:
                            model_recs.append("GÜÇLÜ AL")
                        elif change_pct > 3:
                            model_recs.append("AL")
                        elif change_pct < -10:
                            model_recs.append("GÜÇLÜ SAT")
                        elif change_pct < -3:
                            model_recs.append("SAT")
                        else:
                            model_recs.append("TUT")
                
                buy_count = model_recs.count("GÜÇLÜ AL") + model_recs.count("AL")
                sell_count = model_recs.count("GÜÇLÜ SAT") + model_recs.count("SAT")
                hold_count = model_recs.count("TUT")
                
                st.markdown(f"<p><strong>Model Önerileri Dağılımı:</strong> AL: {buy_count}, TUT: {hold_count}, SAT: {sell_count}</p>", unsafe_allow_html=True)
                
                # Risk seviyesi
                volatility = df['Close'].pct_change().std() * 100
                volatility_level = "Yüksek" if volatility > 3 else ("Orta" if volatility > 1.5 else "Düşük")
                risk_level = "Yüksek" if volatility_level == "Yüksek" or avg_weighted_change < -5 or avg_weighted_change > 10 else ("Orta" if volatility_level == "Orta" else "Düşük")
                risk_color = "red" if risk_level == "Yüksek" else ("orange" if risk_level == "Orta" else "green")
                
                st.markdown(f"<p><strong>Volatilite:</strong> {volatility:.2f}% ({volatility_level})</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Risk Seviyesi:</strong> <span style='color: {risk_color};'>{risk_level}</span></p>", unsafe_allow_html=True)
                
                # Haberleri göster (eğer varsa)
                try:
                    news = get_stock_news(stock_symbol, limit=3)
                    if news and len(news) > 0:
                        st.subheader("İlgili Haberler")
                        for item in news:
                            st.markdown(f"**{item.get('title', '')}** - {item.get('date', '')}")
                            st.markdown(f"{item.get('summary', '')[:150]}...")
                except:
                    # Haber çekme hatası olursa sessizce geç
                    pass
                
                # Analiz sonuçlarını kaydet
                ml_analysis_result = {
                    "symbol": stock_symbol,
                    "company_name": company_info.get("name", ""),
                    "last_price": all_predictions["RandomForest"]['last_price'],
                    "price_change": avg_weighted_change,
                    "recommendation": rec_text,
                    "trend": all_predictions["RandomForest"]['trend'],
                    "risk_level": risk_level,
                    "model_accuracy": np.mean([pred['r2_score'] * 100 for pred in all_predictions.values()]),
                    "predicted_price": all_predictions["RandomForest"]['prediction_30d'] if days_to_predict >= 30 else all_predictions["RandomForest"]['prediction_7d'],
                    "days_to_predict": days_to_predict,
                    "confidence": np.mean([pred['confidence'] for pred in all_predictions.values()]),
                    "predictions": {
                        "7d": all_predictions["RandomForest"]['prediction_7d'],
                        "30d": all_predictions["RandomForest"]['prediction_30d']
                    },
                    "analysis_type": "ml",
                    "model_type": "Multi-Model",
                    "volatility": volatility
                }
                
                # Mevcut fiyatı değişkene alalım
                current_price = all_predictions["RandomForest"]['last_price']
                
                # Analiz sonuçlarını kaydet - fiyat parametresini doğru şekilde geçir
                save_analysis_result(
                    symbol=stock_symbol, 
                    analysis_type="ml", 
                    price=current_price, 
                    result_data=ml_analysis_result, 
                    indicators=None, 
                    notes=f"{days_to_predict} günlük ML tahmin: {rec_text}"
                )
                
            except Exception as e:
                st.error(f"Tahmin oluşturulurken bir hata oluştu: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                return
            
            # Add a note about ML predictions
            with st.expander("ML Tahminleri Hakkında Bilgi"):
                st.info("Not: Makine öğrenimi tahminleri sadece tarihsel veriler kullanılarak yapılmaktadır. "
                        "Ekonomik, politik veya şirkete özel gelişmeler tahmin dışında tutulmuştur. "
                        "Bu tahminleri yatırım tavsiyesi olarak değil, teknik analiz sonucu olarak değerlendirin.")
                
                st.markdown("""
                **Model Bilgileri:**
                * RandomForest: Birden fazla karar ağacının birleştirilmesiyle oluşan güçlü bir regresyon modelidir.
                * XGBoost: Gradyan artırma tekniğiyle çalışan, genellikle daha yüksek performans sunan gelişmiş bir modeldir.
                * LightGBM: Hafif ve hızlı gradyan artırma modelidir, büyük veri setleri için optimize edilmiştir.
                * Ensemble: Tüm modellerin tahminlerini birleştirerek daha güvenilir sonuçlar elde etmeyi amaçlar.
                * Hibrit Model: Makine öğrenimi ile teknik göstergeleri (RSI, MACD vb.) birleştiren bir yaklaşımdır.
                
                **Tahmin Güvenilirliği:**
                * R² Skoru: Modelin veriyi ne kadar iyi açıkladığını gösterir. 1'e yakın değerler daha iyidir.
                * Güven: Modelin tahmin üzerindeki güven seviyesini gösterir.
                
                **Kullanılan Göstergeler:**
                * Hareketli Ortalamalar (SMA, EMA)
                * RSI (Göreceli Güç İndeksi)
                * MACD (Hareketli Ortalama Yakınsama/Iraksama)
                * Bollinger Bantları
                * Stokastik Osilatör
                * ADX (Ortalama Yön Endeksi)
                """)
    
    else:
        st.info("Hisse senedi kodunu girin ve 'Tüm Modellerle Tahmin Et' butonuna tıklayın.") 
        
        # ML tahmin özelliklerini tanıt
        st.subheader("Çoklu Model Tahmin Sistemi Nasıl Çalışır?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Çoklu Model Tahmin Özellikleri:**
            * 5 farklı ML modeli (RandomForest, XGBoost, LightGBM, Ensemble, Hibrit)
            * 7, 14, 30, 60 veya 90 günlük tahminler
            * Modeller arası karşılaştırma ve konsensus
            * Ağırlıklı ortalama yatırım tavsiyeleri
            * Her model için detaylı inceleme imkanı
            """)
            
        with col2:
            st.markdown("""
            **Model Karakteristikleri:**
            * RandomForest: Dengeli ve güvenilir
            * XGBoost: Yüksek performanslı, karmaşık ilişkileri yakalama
            * LightGBM: Hızlı ve hafif, büyük veriler için ideal
            * Ensemble: Tüm modellerin güçlü yanlarını birleştirir
            * Hibrit: Teknik göstergelerle ML'i harmanlayan akıllı sistem
            """) 