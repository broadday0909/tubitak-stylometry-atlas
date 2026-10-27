@echo off
chcp 65001 > nul
title TÜBİTAK - Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası
cls

echo =========================================================================================
echo  TÜBİTAK: EVRENSEL İNGİLİZ DİLİ VE EDEBİYATI STİLOMETRİ ATLASI (CHAUCER'DAN MODERNİZM'E)
echo  Developed by Bayram Akpamukçu | TÜBİTAK Computational Literary Studies & Stylometry
echo =========================================================================================
echo.
echo [*] Sistem yapılandırması kontrol ediliyor...

:: 1. Öncelik: Derlenmiş Dağıtım .EXE Klasörü
if exist "dist\StylometryAtlas_BayramAkpamukcu\StylometryAtlas_BayramAkpamukcu.exe" (
    echo [+] Derlenmiş Executable bulundu. Başlatılıyor...
    start "" "dist\StylometryAtlas_BayramAkpamukcu\StylometryAtlas_BayramAkpamukcu.exe"
    exit /b 0
)

:: 2. Öncelik: Derlenmiş Tek Dosya .EXE
if exist "dist\StylometryAtlas_BayramAkpamukcu.exe" (
    echo [+] Derlenmiş Tek Dosya (.exe) bulundu. Başlatılıyor...
    start "" "dist\StylometryAtlas_BayramAkpamukcu.exe"
    exit /b 0
)

:: 3. Öncelik: Yerel Python Sanal Ortamı (.venv)
if exist ".venv\Scripts\python.exe" (
    echo [+] Python sanal ortamı (.venv) üzerinden başlatıcı çalıştırılıyor...
    echo [+] Web tarayıcınız (http://localhost:8501) birazdan açılacaktır...
    echo.
    ".venv\Scripts\python.exe" launcher.py
    goto end
)

:: 4. Öncelik: Genel Sistem Python'ı
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [+] Sistem Python'ı üzerinden başlatılıyor...
    python launcher.py
    goto end
)

echo.
echo [HATA] Sistemde ne derlenmiş bir .exe ne de Python ortamı bulunamadı!
echo Lütfen Python 3.10+ kurunuz ve 'pip install -r requirements.txt' komutunu çalıştırınız.
echo.
pause

:end
