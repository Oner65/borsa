@echo off
echo Borsa Istanbul Hisse Analiz Uygulamasi baslatiliyor...
echo.

REM Python venv kontrolü
if exist venv (
    echo Virtual environment bulundu, aktive ediliyor...
    call venv\Scripts\activate
) else (
    echo Virtual environment olusturuluyor...
    python -m venv venv
    call venv\Scripts\activate
    
    echo Gerekli paketler yukleniyor...
    pip install -r requirements.txt
)

echo.
echo Uygulama baslatiliyor...
echo.

REM Uygulamayı başlat
streamlit run borsa.py

REM Hata durumunda bekle
if %ERRORLEVEL% neq 0 (
    echo.
    echo Hata olustu! Cikis yapmak icin bir tusa basin...
    pause >nul
) 