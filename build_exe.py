"""
TÜBİTAK Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası
===========================================================
Otomatik PyInstaller .EXE Derleyici (build_exe.py)
Developed by Bayram Akpamukçu | TÜBİTAK Computational Literary Studies & Stylometry Project
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Windows konsolunda UTF-8 çıktı desteği
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def clean_previous_builds(base_dir: Path) -> None:
    """Önceki derlemeden kalan geçici klasörleri temizler."""
    dirs_to_remove = [base_dir / "build", base_dir / "dist"]
    files_to_remove = list(base_dir.glob("*.spec"))

    for d in dirs_to_remove:
        if d.exists() and d.is_dir():
            print(f"[*] Eski derleme dizini temizleniyor: {d.name}...")
            shutil.rmtree(d, ignore_errors=True)

    for f in files_to_remove:
        try:
            f.unlink()
        except OSError:
            pass


def build_executable(onefile: bool = False) -> Path:
    """
    PyInstaller kullanarak çalıştırılabilir .exe dosyasını derler.
    
    Args:
        onefile: True ise tek bir .exe dosyası üretir. False ise hızlı açılan klasör paketi üretir.
        
    Returns:
        Path: Üretilen .exe dosyasının yolu.
    """
    base_dir = Path(__file__).resolve().parent
    launcher_path = base_dir / "launcher.py"
    app_path = base_dir / "app.py"
    src_dir = base_dir / "src"
    data_dir = base_dir / "data"

    print("=" * 95)
    print(" TÜBİTAK: EVRENSEL İNGİLİZ DİLİ VE EDEBİYATI STİLOMETRİ ATLASI - .EXE DERLEME ARACI")
    print(" Developed by Bayram Akpamukçu | TÜBİTAK Computational Literary Studies & Stylometry Project")
    print("=" * 95)

    clean_previous_builds(base_dir)

    target_name = "StylometryAtlas_BayramAkpamukcu"
    separator = ";" if sys.platform == "win32" else ":"

    # PyInstaller argümanları listesi
    pyinstaller_args = [
        str(launcher_path),
        f"--name={target_name}",
        "--clean",
        "--noconfirm",
        # Veri klasörleri ve betikleri pakete dahil et
        f"--add-data={app_path}{separator}.",
        f"--add-data={src_dir}{separator}src",
        f"--add-data={data_dir}{separator}data",
        # Streamlit ve gerekli kütüphanelerin metadata/statik dosyaları
        "--copy-metadata=streamlit",
        "--copy-metadata=tqdm",
        "--copy-metadata=regex",
        "--copy-metadata=requests",
        "--copy-metadata=packaging",
        "--collect-data=streamlit",
        # Gizli modül (hidden import) tanımları
        "--hidden-import=streamlit",
        "--hidden-import=streamlit.web.cli",
        "--hidden-import=sklearn",
        "--hidden-import=sklearn.utils._typedefs",
        "--hidden-import=sklearn.neighbors._typedefs",
        "--hidden-import=sklearn.tree",
        "--hidden-import=sklearn.ensemble",
        "--hidden-import=sklearn.svm",
        "--hidden-import=matplotlib",
        "--hidden-import=seaborn",
        "--hidden-import=pandas",
        "--hidden-import=numpy",
        "--hidden-import=tabulate",
        "--hidden-import=nltk",
        "--hidden-import=src",
        "--hidden-import=src.preprocessor",
        "--hidden-import=src.feature_extractor",
        "--hidden-import=src.corpus_builder",
        "--hidden-import=src.classifier",
        "--hidden-import=src.visualizer",
        "--hidden-import=src.canon_catalog",
        "--hidden-import=src.gutenberg_fetcher",
        "--hidden-import=src.advanced_forensics",
        "--hidden-import=src.literary_network",
        "--hidden-import=src.prosody_atmosphere",
        "--hidden-import=src.style_coach",
        "--hidden-import=src.academic_exporter",
        "--hidden-import=scipy",
        "--hidden-import=scipy.cluster",
        "--hidden-import=scipy.cluster.hierarchy",
        "--hidden-import=scipy.spatial",
        "--hidden-import=scipy.spatial.distance",
    ]

    if onefile:
        print("[+] Derleme Modu: Tek Dosya (--onefile) Executable")
        pyinstaller_args.append("--onefile")
        expected_exe = base_dir / "dist" / f"{target_name}.exe"
    else:
        print("[+] Derleme Modu: Yüksek Performanslı Klasör Paketi (--onedir)")
        pyinstaller_args.append("--onedir")
        expected_exe = base_dir / "dist" / target_name / f"{target_name}.exe"

    print(f"[+] Hedef Dosya: {expected_exe.name}")
    print("[+] Derleme başlatılıyor, lütfen bekleyiniz (bu işlem 1-2 dakika sürebilir)...\n")

    try:
        import PyInstaller.__main__
        PyInstaller.__main__.run(pyinstaller_args)
    except Exception as err:
        print(f"\n[HATA] PyInstaller derleme sırasında hata oluştu: {err}")
        sys.exit(1)

    if expected_exe.exists():
        size_mb = expected_exe.stat().st_size / (1024 * 1024)
        print("\n" + "=" * 95)
        print(" [✓] DERLEME BAŞARIYLA TAMAMLANDI!")
        print(f" [✓] Üretilen Executable: {expected_exe}")
        print(f" [✓] Boyut: {size_mb:.2f} MB")
        print("=" * 95)
        print("\n[İPUCU] Artık bu dosyayı çift tıklayarak veya 'Baslat_Stylometry_Atlas.bat' ile başlatabilirsiniz.")
        return expected_exe
    else:
        print(f"\n[HATA] Beklenen dosya oluşturulamadı: {expected_exe}")
        sys.exit(1)


if __name__ == "__main__":
    is_onefile = "--onefile" in sys.argv
    build_executable(onefile=is_onefile)
