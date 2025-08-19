@echo off
echo ========================================
echo     STREAMLIT CLOUD DEPLOY HAZIRLIGI
echo ========================================
echo.

REM Deploy için gerekli dosyaları kontrol et
echo 📋 Gerekli dosyalar kontrol ediliyor...

if exist "compact_borsa_app.py" (
    echo ✅ compact_borsa_app.py - TAMAM
) else (
    echo ❌ compact_borsa_app.py - EKSIK!
    goto :error
)

if exist "requirements.txt" (
    echo ✅ requirements.txt - TAMAM
) else (
    echo ❌ requirements.txt - EKSIK!
    goto :error
)

if exist ".streamlit\config.toml" (
    echo ✅ .streamlit/config.toml - TAMAM
) else (
    echo ❌ .streamlit/config.toml - EKSIK!
    goto :error
)

if exist "packages.txt" (
    echo ✅ packages.txt - TAMAM
) else (
    echo ❌ packages.txt - EKSIK!
    goto :error
)

if exist ".gitignore" (
    echo ✅ .gitignore - TAMAM
) else (
    echo ❌ .gitignore - EKSIK!
    goto :error
)

echo.
echo 🎉 Tüm dosyalar hazır!
echo.

REM Git durumunu kontrol et
echo 📋 Git durumu kontrol ediliyor...
git status > nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Git repository başlatılmamış.
    echo.
    set /p "choice=Git repository başlatmak istiyor musunuz? (y/n): "
    if /i "%choice%"=="y" (
        git init
        echo ✅ Git repository başlatıldı!
    )
)

echo.
echo ========================================
echo          SONRAKİ ADIMLAR
echo ========================================
echo.
echo 1️⃣ GitHub'da yeni repository oluşturun:
echo    https://github.com/new
echo    📝 Adı: kompakt-borsa-app
echo    📝 Public olarak ayarlayın
echo.
echo 2️⃣ Terminal'de bu komutları çalıştırın:
echo    git add .
echo    git commit -m "İlk commit"
echo    git remote add origin https://github.com/USERNAME/kompakt-borsa-app.git
echo    git branch -M main
echo    git push -u origin main
echo.
echo 3️⃣ Streamlit Cloud'a deploy edin:
echo    https://share.streamlit.io
echo    📝 Repository: USERNAME/kompakt-borsa-app
echo    📝 Branch: main
echo    📝 Main file: compact_borsa_app.py
echo.
echo 🔗 Detaylı rehber için: HIZLI_DEPLOY.md dosyasını okuyun
echo.
goto :end

:error
echo.
echo ❌ HATA: Bazı dosyalar eksik!
echo 📋 Lütfen önce tüm dosyaların oluşturulduğundan emin olun.
echo.

:end
echo ========================================
pause 