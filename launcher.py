"""
TÜBİTAK Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası
===========================================================
Programatik Yerel Masaüstü Başlatıcı (launcher.py)
Developed by Bayram Akpamukçu | TÜBİTAK Computational Literary Studies & Stylometry Project
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

# PyInstaller paketleme ortamında Streamlit'in developmentMode varsaymasını önle
os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

# Windows konsolunda UTF-8 çıktı desteği
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# PyInstaller'ın Streamlit çalışma zamanı ve CLI modüllerini kaçırmasını önleyen açık importlar
import streamlit.runtime.scriptrunner.magic_funcs
import streamlit.web.bootstrap as bootstrap
import streamlit.web.cli as stcli
import streamlit.config as _st_config

# Streamlit'in arka plan iş parçacığında çalışabilmesi için ana iş parçacığı sinyal yöneticisini devre dışı bırak
bootstrap._set_up_signal_handler = lambda server: None


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


def find_available_port(start_port: int = 8501, max_search: int = 30) -> int:
    """Belirtilen porttan başlayarak boş bir TCP portu bulur."""
    import socket
    for port in range(start_port, start_port + max_search):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start_port


def wait_for_server(port: int, max_wait: float = 30.0) -> bool:
    """Belirtilen portta Streamlit sunucusunun dinlemeye başlamasını bekler."""
    import socket
    start = time.time()
    while time.time() - start < max_wait:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(0.3)
    return False


def start_streamlit_server(app_path: Path, port: int) -> None:
    """Streamlit web sunucusunu arka plan iş parçacığında çalıştırır."""
    try:
        _st_config.set_option("global.developmentMode", False)
    except Exception:
        pass

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        f"--server.port={port}",
        "--server.headless=true",
        "--global.developmentMode=false",
        "--browser.gatherUsageStats=false",
        "--theme.base=light",
    ]

    try:
        stcli.main()
    except SystemExit:
        pass
    except Exception as e:
        print(f"[!] Arka plan sunucu hatası: {e}")


def launch_browser_app_mode(url: str) -> None:
    """
    pywebview / WebView2 bulunamadığında Edge veya Chrome'u
    adres çubuğu olmayan yerel masaüstü uygulama penceresi modunda açar.
    """
    candidates = [
        os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ]
    for exe_path in candidates:
        if os.path.exists(exe_path):
            try:
                print(f"[+] Bağımsız masaüstü penceresi başlatılıyor ({Path(exe_path).name} App Modu)...")
                p = subprocess.Popen([exe_path, f"--app={url}", "--window-size=1440,900"])
                p.wait()
                return
            except Exception:
                continue

    # Hiçbir tarayıcı bulunamazsa varsayılan tarayıcıyı aç
    webbrowser.open(url)


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

    # Konsol doğrulama modu kontrolü
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
        cwd_app = Path.cwd() / "app.py"
        if cwd_app.exists():
            app_path = cwd_app
        else:
            print(f"[HATA] 'app.py' dosyası bulunamadı: {app_path}")
            sys.exit(1)

    port = find_available_port(8501)
    url = f"http://127.0.0.1:{port}"

    print(f"\n[+] Streamlit Arka Plan Çekirdeği Başlatılıyor: {url}")
    print("[+] Yerel Masaüstü Uygulama Penceresi (Native Desktop Window) hazırlanıyor...")

    # Streamlit sunucusunu arka planda sessiz iş parçacığında başlat
    server_thread = threading.Thread(
        target=start_streamlit_server,
        args=(app_path, port),
        daemon=True,
    )
    server_thread.start()

    # Sunucu portunun dinlemeye başlamasını bekle
    if not wait_for_server(port, max_wait=30.0):
        print(f"\n[HATA] Atlas sunucusu {port} portunda başlatılamadı.")
        input("\nKapatmak icin Enter tusuna basin...")
        sys.exit(1)

    print("[✓] Çekirdek sunucu hazır! Masaüstü penceresi açılıyor...\n")

    app_title = "TÜBİTAK Stilometri Atlası - Developed by Bayram Akpamukçu"

    # pywebview ile masaüstü penceresi oluştur
    opened_with_webview = False
    try:
        import webview
        webview.create_window(
            app_title,
            url,
            width=1440,
            height=900,
            min_size=(1024, 700),
        )
        opened_with_webview = True
        webview.start()
    except Exception as w_err:
        print(f"[!] WebView başlatılamadı ({w_err}). Yedek masaüstü penceresi modu devreye giriyor...")
        opened_with_webview = False

    if not opened_with_webview:
        launch_browser_app_mode(url)

    print("\n[+] Atlas masaüstü penceresi kapatıldı. Uygulama güvenle sonlandırıldı.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as se:
        if se.code not in (0, None):
            print(f"\n[!] Uygulama {se.code} koduyla sonlandı.")
            input("\nKapatmak icin Enter tusuna basin...")
    except KeyboardInterrupt:
        print("\n[+] Atlas uygulaması sonlandırıldı.")
    except Exception as ex:
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 95)
        print(f"[KRİTİK HATA] Atlas başlatılırken beklenmedik hata oluştu: {ex}")
        print("=" * 95)
        input("\nKapatmak icin Enter tusuna basin...")
