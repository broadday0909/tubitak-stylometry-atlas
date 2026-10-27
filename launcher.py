"""
TÜBİTAK Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası
===========================================================
Programatik Başlatıcı ve Otomatik Web Tarayıcısı Açıcı (launcher.py)
Developed by Bayram Akpamukçu | TÜBİTAK Computational Literary Studies & Stylometry Project
"""

from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Windows konsolunda UTF-8 çıktı desteği
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def resolve_base_directory() -> Path:
    """
    Çalışma dizinini tespit eder.
    PyInstaller tek dosya (.exe) modunda sys._MEIPASS dizinini,
    normal Python ortamında ise dosyanın bulunduğu dizini döner.
    """
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            return Path(sys._MEIPASS)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def open_browser_delayed(url: str = "http://localhost:8501", delay_seconds: float = 2.0) -> None:
    """Streamlit sunucusu açılana kadar kısa bir gecikmeyle tarayıcıyı başlatır."""
    def _open():
        time.sleep(delay_seconds)
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"[!] Tarayıcı otomatik açılamadı: {e}. Lütfen tarayıcınızdan {url} adresine gidiniz.")

    thread = threading.Thread(target=_open, daemon=True)
    thread.start()


def print_banner() -> None:
    """Açılış kurumsal kimlik başlığını yazdırır."""
    print("=" * 95)
    print(" TÜBİTAK: EVRENSEL İNGİLİZ DİLİ VE EDEBİYATI STİLOMETRİ ATLASI (CHAUCER'DAN MODERNİZM'E)")
    print(" Developed by Bayram Akpamukçu | TÜBİTAK Computational Literary Studies & Stylometry Project")
    print(" 7 Edebi Dönem, 37 Kanonik Yazar, Hiyerarşik Koşullandırma & Burrows' Delta Motoru")
    print("=" * 95)


def main() -> None:
    print_banner()
    base_dir = resolve_base_directory()

    # sys.path içine ana dizini ekle
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    # Konsol modu istenmişse doğrudan main.py boru hattını çalıştır
    if "--console" in sys.argv or "--test" in sys.argv:
        print("\n[+] Konsol Doğrulama Modu Seçildi. main.py çalıştırılıyor...\n")
        try:
            import main as console_main
            console_main.main()
            return
        except Exception as err:
            print(f"[HATA] Konsol modu çalışırken hata oluştu: {err}")
            sys.exit(1)

    app_path = base_dir / "app.py"
    if not app_path.exists():
        # Fallback to current working directory
        cwd_app = Path.cwd() / "app.py"
        if cwd_app.exists():
            app_path = cwd_app
        else:
            print(f"[HATA] 'app.py' dosyası bulunamadı: {app_path}")
            sys.exit(1)

    port = 8501
    url = f"http://localhost:{port}"

    print(f"\n[+] Streamlit Web Arayüzü Başlatılıyor: {url}")
    print("[+] Varsayılan web tarayıcınız 2 saniye içinde otomatik olarak açılacaktır...")
    print("[+] Uygulamayı durdurmak için bu konsol penceresini kapatabilir veya Ctrl+C tuşlarına basabilirsiniz.\n")

    open_browser_delayed(url, delay_seconds=2.0)

    try:
        from streamlit.web import cli as stcli
    except ImportError:
        print("[HATA] 'streamlit' kütüphanesi bulunamadı. Lütfen 'pip install -r requirements.txt' komutunu çalıştırınız.")
        sys.exit(1)

    # Streamlit CLI argümanları
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        f"--server.port={port}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--theme.base=light",
    ]

    try:
        sys.exit(stcli.main())
    except SystemExit:
        pass
    except KeyboardInterrupt:
        print("\n[+] Atlas uygulaması kullanıcı tarafından sonlandırıldı.")


if __name__ == "__main__":
    main()
