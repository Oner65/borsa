@echo off
echo ========================================
echo  Kompakt Borsa Analiz Uygulaması
echo  BIST 100 - Hisse Analizi - ML Tahmin
echo ========================================
echo.

REM Sanal ortamı etkinleştir
if exist "venv\Scripts\activate.bat" (
    echo Sanal ortam etkinleştiriliyor...
    call venv\Scripts\activate.bat
) else (
    echo UYARI: Sanal ortam bulunamadi. Ana Python kullanilacak.
)

echo.
echo Kompakt borsa uygulaması başlatılıyor...
echo.
echo Tarayıcınızda http://localhost:8502 adresinde açılacak.
echo Uygulamayı kapatmak için bu pencerede Ctrl+C tuşlarına basın.
echo.

REM Streamlit uygulamasını başlat (farklı port kullan)
streamlit run compact_borsa_app.py --server.port=8502 --server.headless=false

pause 