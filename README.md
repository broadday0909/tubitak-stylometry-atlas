# 🏛️ TÜBİTAK: Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası
### Universal English Language & Literature Stylometry Atlas (Chaucer'dan Modernizm'e)

<div align="center">

![Developer](https://img.shields.io/badge/Developed%20by-Bayram%20Akpamuk%C3%A7u-0A66C2?style=for-the-badge&logo=github&logoColor=white)
![TÜBİTAK](https://img.shields.io/badge/T%C3%9CB%C4%B0TAK-2209%20Research-E11D48?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-25%2F25%20Passing-success?style=for-the-badge&logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

<p align="center">
  <b>Hesaplamalı Edebiyat (Computational Literary Studies) ve Doğal Dil İşleme (NLP) Altyapısı</b><br>
  7 Edebi Dönem • 37 Kanonik Yazar • Çift Katmanlı Sınıflandırma • Hiyerarşik Koşullandırma • Burrows' Delta Z-Skor Mesafesi
</p>

</div>

---

## 📌 Proje Genel Bakışı

Bu proje, edebi metinlerin biçimsel, sözcüksel ve sözdizimsel üslup özelliklerini (**70+ stilometrik boyut / stylometric fingerprint**) matematiksel olarak analiz ederek **edebi dönem tespiti (era detection)**, **yazar atfı (authorship attribution)** ve **üslup evrimi karşılaştırması** yapan ileri düzey bir yapay zekâ ve Doğal Dil İşleme (NLP) sistemidir.

TÜBİTAK standartlarına uygun olarak tasarlanan mimari, Geoffrey Chaucer'ın Orta Çağ İngilizcesinden Virginia Woolf ve James Joyce'un Modernist bilinç akışına uzanan **600 yıllık İngiliz ve Amerikan edebiyatı tarihini** 7 ana edebi dönem ve 37 kanonik yazar üzerinden modeller.

---

## 🌟 Aşama 6 & 7 Öne Çıkan Yenilikler

1. **Hiyerarşik Koşullu Yazar Tahmini (Hierarchical Conditioning):**
   * Multi-class SVM sınıflandırıcısında yaşanan anakronik sapmaları (örn: Viktorya dönemi tespit edilen Dickens pasajında 20. yüzyıl yazarı Hemingway tahmini yapılması) ortadan kaldırmak için Katman 1 Dönem olasılıkları $P(\text{Era}|X)$, ham yazar olasılıkları $P(\text{Author}|X)$ ve Burrows' Delta yakınlığı ($\exp(-\Delta)$) tek bir olasılık dağılımında birleştirilmiştir.
   * **Kör Test Başarısı:** 7 edebi dönemin 7 bağımsız test pasajında **%100 Dönem (7/7)** ve **%100 Top-1 Yazar (7/7)** doğruluk sağlanmıştır.
2. **Çift Tıklama ile Çalışan .EXE & Başlatıcı Altyapısı:**
   * Python veya bağımlılık kurulumu gerektirmeyen `StylometryAtlas_BayramAkpamukcu.exe` ve tek tıkla çalışan `Baslat_Stylometry_Atlas.bat` araçları geliştirilmiştir.
3. **CI/CD & GitHub Standartları:**
   * GitHub Actions iş akışı (`.github/workflows/ci.yml`), 25 birim testinden oluşan test süitini her commit ve pull request işleminde otomatik olarak doğrular.
   * MIT Lisansı ve GitHub dosya limitlerini gözeten profesyonel `.gitignore` yapılandırması tamamlanmıştır.

---

## 🧠 Mimari ve Matematiksel Temeller

```mermaid
flowchart TD
    A[Bilinmeyen / Anonim Edebi Metin] --> B[NLP Ön İşleme & Tokenizasyon]
    B --> C[70+ Stilometrik Öznitelik Çıkarıcı]
    
    C --> D[Katman 1: Edebi Dönem Sınıflandırıcısı]
    C --> E[Katman 2: Kanonik Yazar Sınıflandırıcısı]
    C --> F[Burrows' Delta Manhattan Z-Skor Motoru]
    
    D -->|Dönem Olasılıkları P(Era|X)| G[Hiyerarşik Koşullandırma Füzyonu]
    E -->|Ham Yazar Olasılıkları P(Author|X)| G
    F -->|Z-Skor Yakınlığı exp(-Delta)| G
    
    G --> H[Final Kalibre Edilmiş Yazar & Dönem Tahmini]
    H --> I[Streamlit Etkileşimli Atlas & Görselleştirme]
```

### 1. Hiyerarşik Koşullandırma (Hierarchical Conditioning) Formülü
Her bir yazar adayı $A_i$ için bileşik olasılık skoru:
$$S(A_i) = P_{\text{raw}}(A_i|X) \cdot [P(\text{Era}(A_i)|X) + \epsilon]^{1.2} \cdot \exp(-\Delta_i) \cdot W_{\text{era\_boost}}$$
Burada:
* $P_{\text{raw}}(A_i|X)$: Linear SVM / Random Forest çok sınıflı olasılık çıktısı.
* $P(\text{Era}(A_i)|X)$: Yazarın ait olduğu edebi dönemin Katman 1 modelindeki olasılığı.
* $\Delta_i$: Metnin külliyattaki yazara olan Burrows' Delta Manhattan Z-skor mesafesi.
* $W_{\text{era\_boost}}$: Tahmin edilen dönemin yazarlarına uygulanan pozitif güçlendirme katsayısı ($3.0$), yüzyıl dışı adaylara uygulanan sönümlendirme ($1 / 3.0^{1.5}$).

### 2. John Burrows' Delta Metriği (Hesaplamalı Edebiyat Altın Standardı)
$$\Delta(D, A) = \frac{1}{n} \sum_{k=1}^n \left| \frac{f_k(D) - \mu_k}{\sigma_k + \epsilon} - \frac{f_k(A) - \mu_k}{\sigma_k + \epsilon} \right|$$

---

## 📚 Edebi Dönemler ve 37 Kanonik Yazar Kataloğu

| Edebi Dönem | Kronolojik Aralık | Kapsanan Kanonik Yazarlar |
| :--- | :--- | :--- |
| **1. Middle English** | c. 1100 – 1500 | Geoffrey Chaucer |
| **2. Renaissance & Early Modern** | c. 1500 – 1660 | William Shakespeare, Christopher Marlowe, John Milton, John Donne, Francis Bacon, Edmund Spenser |
| **3. Enlightenment & 18th Century** | c. 1660 – 1789 | Jonathan Swift, Daniel Defoe, Alexander Pope, Samuel Johnson, Joseph Addison |
| **4. Romanticism** | c. 1789 – 1837 | Jane Austen, Mary Shelley, William Blake, William Wordsworth, John Keats, Lord Byron, Samuel Taylor Coleridge |
| **5. Victorian Era** | c. 1837 – 1901 | Charles Dickens, Oscar Wilde, Charlotte Brontë, Emily Brontë, George Eliot, Thomas Hardy, Arthur Conan Doyle, Bram Stoker |
| **6. American Classics** | 19. Yüzyıl | Edgar Allan Poe, Herman Melville, Mark Twain, Walt Whitman, Emily Dickinson, Nathaniel Hawthorne |
| **7. Modernism** | c. 1900 – 1945 | Virginia Woolf, James Joyce, Ernest Hemingway, T.S. Eliot, F. Scott Fitzgerald, Joseph Conrad, George Orwell, W.B. Yeats |

---

## 🚀 Hızlı Başlangıç ve Çalıştırma Rehberi

Projeyi üç farklı yöntemle anında çalıştırabilirsiniz:

### Yöntem A: Tek Tıkla Başlatma (Windows `.bat` / `.exe` - Önerilen)
1. Proje ana dizinindeki **`Baslat_Stylometry_Atlas.bat`** (veya `Baslat_Atlas.bat`) dosyasına çift tıklayın.
2. Başlatıcı otomatik olarak sistem yapılandırmasını kontrol eder, Streamlit sunucusunu başlatır ve varsayılan web tarayıcınızda `http://localhost:8501` adresini açar.

### Yöntem B: Python Ortamı ile Çalıştırma
Gereksinim: Python 3.10 veya üzeri.

```bash
# 1. Depoyu klonlayın veya indirin
git clone https://github.com/broadday0909/tubitak-stylometry-atlas.git
cd tubitak-stylometry-atlas

# 2. Sanal ortam oluşturun ve aktif edin
python -m venv .venv
# Windows için:
.venv\Scripts\activate
# Linux/macOS için:
source .venv/bin/activate

# 3. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 4. Web uygulamasını başlatın
streamlit run app.py
# veya otomatik tarayıcı açıcı ile:
python launcher.py
```

### Yöntem C: Konsol Modunda 7 Edebi Dönem Kör Testini Çalıştırma
```bash
python main.py
# veya
python launcher.py --console
```

---

## 🧪 Birim Testleri Çalıştırma (Unit Testing)

Tüm matematiksel formüller, Türkçe/İngilizce tokenizasyon kuralları ve hiyerarşik tutarlılık testleri tek komutla doğrulanır:

```bash
python -m unittest discover -s tests -v
```

**Sonuç Özeti:**
```text
test_hierarchical_conditioning_consistency ... ok
test_predict_era ... ok
test_predict_top_authors ... ok
test_compute_burrows_delta ... ok
test_archaic_words_extraction ... ok
test_morphological_suffixes ... ok
test_stream_of_consciousness_index ... ok
...
Ran 25 tests in 0.276s - OK (0 fail, 0 error)
```

---

## 📦 Kendi `.EXE` Dosyanızı Derleme (PyInstaller)

İstediğiniz zaman projeyi taşınabilir Windows Executable formatına derlemek için:

```bash
# Hızlı açılan klasör paketi derlemesi:
python build_exe.py

# Tek dosya (.exe) derlemesi:
python build_exe.py --onefile
```
Derlenen dosyalar `dist/StylometryAtlas_BayramAkpamukcu/` klasörüne yerleştirilir.

---

## 🌐 GitHub Releases'a `.EXE` Yükleme Rehberi

GitHub'da 100 MB üzeri ikili dosyalar Git geçmişinde tutulmaz; bunun yerine **GitHub Releases** kullanılır:

1. Proje dizininde `dist/StylometryAtlas_BayramAkpamukcu` klasörünü bir `.zip` arşivi haline getirin (Örn: `StylometryAtlas_v1.0_Windows.zip`).
2. GitHub deponuzun ana sayfasına gidin, sağ taraftaki **Releases -> Create a new release** butonuna tıklayın.
3. Tag versiyonunu `v1.0` olarak belirleyin.
4. Başlık olarak `TÜBİTAK Stylometry Atlas v1.0 - Standalone Windows Release` yazın.
5. Hazırladığınız `StylometryAtlas_v1.0_Windows.zip` dosyasını sürüm dosyaları alanına sürükleyip bırakın ve **Publish release** butonuna basın.

---

## 📁 Proje Dizin Hiyerarşisi

```text
tubitak/
│
├── .github/
│   └── workflows/
│       └── ci.yml                          # Otomatik test ve doğrulama CI iş akışı
│
├── data/
│   └── raw/                                # 37 Kanonik yazar metin külliyatı (.txt)
│
├── src/
│   ├── __init__.py                         # Paket modülü dışa aktarımları
│   ├── canon_catalog.py                    # 7 Edebi dönem, 37 yazar künye ve pasaj kataloğu
│   ├── gutenberg_fetcher.py                # Külliyat senkronizasyonu ve Gutenberg yöneticisi
│   ├── preprocessor.py                     # NLP tokenizasyon, arkaizm ve noktalama filtreleri
│   ├── feature_extractor.py                # 70+ stilometrik boyut çıkaran motor
│   ├── corpus_builder.py                   # Dinamik bloklama ve (X, y_author, y_era) üretici
│   ├── classifier.py                       # Çift Katmanlı Model + Hiyerarşik Koşullandırma + Delta
│   └── visualizer.py                       # Confusion Matrix, Feature Importance ve Radar grafikleri
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessor.py                # Metin işleme ve tokenizasyon testleri
│   ├── test_feature_extractor.py           # Matematiksel stilometri metrik testleri
│   └── test_classifier.py                  # Çift katman, Delta ve hiyerarşik tutarlılık testleri
│
├── app.py                                  # Streamlit 4 Sekmeli İnteraktif Web Atlası
├── main.py                                 # Konsol kör test ve doğrulama boru hattı
├── launcher.py                             # Programatik başlatıcı ve otomatik tarayıcı açıcı
├── build_exe.py                            # PyInstaller .exe derleme aracı
├── Baslat_Stylometry_Atlas.bat             # Çift tıklama ile başlatan Windows scripti
├── Baslat_Atlas.bat                        # Kısa başlatıcı alternatifi
├── requirements.txt                        # Python kütüphane bağımlılıkları
├── .gitignore                              # Git dosya filtreleme kuralları
├── LICENSE                                 # MIT Lisansı (Bayram Akpamukçu)
└── README.md                               # Vitrin proje dokümantasyonu
```

---

## 👨‍💻 Geliştirici ve Kurumsal Bilgiler

* **Proje Adı:** Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası
* **Geliştirici:** **Bayram Akpamukçu**
* **Kurum:** TÜBİTAK Computational Literary Studies & Stylometry Project
* **Lisans:** [MIT License](LICENSE) (Copyright © 2026 Bayram Akpamukçu)
