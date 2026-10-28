# TÜBİTAK Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası
## Antigravity IDE Proje Belleği ve Geliştirici Kılavuzu

**Proje Sahibi & Geliştirici:** Bayram Akpamukçu  
**Proje Türü:** TÜBİTAK 2209 Computational Literary Studies & Stylometry Project  
**GitHub Deposu:** `broadday0909/tubitak-stylometry-atlas`  
**Canlı Alan Adı:** `https://bayramakpamukcu.site` & `https://www.bayramakpamukcu.site`  
**Bulut Sunucu:** Streamlit Community Cloud (`tubitak-stylometry-atlas.streamlit.app`)  

---

### 1. Sistem Mimarisi ve Temel Dosyalar

- **`app.py`**: 7 sekmeden oluşan etkileşimli Streamlit web arayüzü (Metin Analizi, Yazar Karşılaştırma, Stilistik Galaksi 3D PCA, Gelişmiş Adli Analiz, Edebi Ağ & Filogeni, Prosodi & Atmosfer, Stil Koçu & Akademik Rapor).
- **`launcher.py`**: Çift tıklamayla çalışan yerel masaüstü başlatıcısı. Streamlit'i arka plan sessiz iş parçacığında (`daemon thread`) başlatır, `pywebview` (1440x900) bağımsız masaüstü penceresi açar. WebView2 yoksa Edge/Chrome `--app=` moduna geri düşer.
- **`build_exe.py`**: PyInstaller ile tek tıkla `.exe` ve `StylometryAtlas_Desktop_BayramAkpamukcu.zip` oluşturan derleyici.
- **`src/` Modülleri (Aşama 1-12)**:
  - `preprocessor.py`: Metin temizleme ve tokenizasyon.
  - `feature_extractor.py`: 60+ stilometrik öznitelik çıkarımı (sözcüksel, sözdizimsel, okunabilirlik, noktalama).
  - `corpus_builder.py`: Adaptif metin bloklama (`min_chunks=4` garantisi ile dengeli veri seti).
  - `classifier.py`: Çift katmanlı (Dönem + Yazar) hiyerarşik sınıflandırıcı ve Burrows' Delta z-skor motoru.
  - `visualizer.py`: Matplotlib/Seaborn radar, dendrogram, konfüzyon matrisi ve stil galaksisi grafikleri.
  - `canon_catalog.py`: 7 dönem ve 37+ kanonik yazar kataloğu.
  - `gutenberg_fetcher.py`: Gutenberg edebi metin senkronizasyonu.
  - `advanced_forensics.py`: Adli metin analizi, yapay zeka tespit heuristiği, yuvarlanan delta.
  - `literary_network.py`: 37x37 edebi mesafe matrisi ve filogenetik ağ motoru.
  - `prosody_atmosphere.py`: Ses estetiği, aliterasyon, asonans ve edebi atmosfer analizi.
  - `style_coach.py`: Yazar taklit ve stil koçluğu motoru.
  - `academic_exporter.py`: TÜBİTAK standartlarında APA7 akademik rapor ve BibTeX dışa aktarıcı.

---

### 2. Kritik Kurallar ve Çözülen Sorunlar (Asla Unutulmamalı)

1. **Streamlit Development Mode Çakışması:**
   - PyInstaller paketlerinde Streamlit `site-packages` dışında çalıştığı için `global.developmentMode = True` varsayabilir.
   - Bu nedenle `launcher.py` en başında `os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"` ve `os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"` mutlaka tanımlı kalmalıdır.
2. **Streamlit Arka Plan İş Parçacığı & Signal Hatası:**
   - Streamlit ana iş parçacığı dışında başlatıldığında `signal only works in main thread` hatası verir.
   - Bu nedenle `launcher.py` içinde `streamlit.web.bootstrap._set_up_signal_handler = lambda server: None` monkeypatch'i korunmalıdır.
3. **`magic_funcs` ve PyInstaller Paketleme:**
   - Streamlit AST analizinde dinamik import yaptığı için `build_exe.py` içinde `--collect-all=streamlit`, `--collect-all=webview` ve `--hidden-import=streamlit.runtime.scriptrunner.magic_funcs` bayrakları şarttır.
4. **Sınıflandırma ve Veri Seti Bölme Kararlılığı:**
   - `corpus_builder.chunk_text_by_words`: Her yazar için en az `min_chunks=4` blok üretir.
   - `app.py`: `train_test_split(..., stratify=s_y_author)` kullanarak 41 yazarın tamamının test setinde yer almasını garanti eder.
   - `src/classifier.py` içindeki `evaluate`: `classification_report` ve `confusion_matrix` çağrılarında `labels=labels_list, target_names=target_names_list, zero_division=0` parametreleri açıkça verilmiştir; sınıf boyutu uyuşmazlığı hatası (`ValueError`) fırlatılması engellenmiştir.
5. **Bulut ve Masaüstü Bağımlılık Ayrımı:**
   - `requirements.txt`: Streamlit Community Cloud (Linux) için temiz bağımlılıklar (`streamlit`, `scikit-learn`, `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `nltk`, `tabulate`).
   - `requirements-dev.txt`: Windows masaüstü araçları (`pyinstaller`, `pywebview`).
   - `.gitignore`: 152 MB'lık `*.zip` dosyalarını GitHub'a pushlamaktan korur.

---

### 3. Canlı Yayın ve Alan Adı (bayramakpamukcu.site)

- **GitHub Pages:** `main` dalı `/docs` klasörü üzerinden yayınlanır.
- **`docs/CNAME`:** `bayramakpamukcu.site`
- **`docs/index.html`:** Tam ekran (100vw, 100vh) Streamlit iframe kapsayıcısıdır.
- **Otomatik CI/CD:** Projede yapılan herhangi bir commit & push işlemi, Streamlit Cloud ve GitHub Pages tarafından saniyeler içinde algılanıp canlı sitede otomatik güncellenir.
