"""
TÜBİTAK Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası
===========================================================
Konsol Çalıştırma ve Çift Katmanlı Model Doğrulama Hattı (main.py)

Chaucer'dan Modernizm'e: 7 Edebi Dönem, 37 Kanonik Yazar,
Hiyerarşik Sınıflandırma ve Burrows' Delta Z-Skor Mesafesi Analizi.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import pandas as pd
from tabulate import tabulate

from src.advanced_forensics import AdvancedForensicsEngine
from src.canon_catalog import CANON_CATALOG, LITERARY_ERAS
from src.classifier import StylometryClassifier
from src.corpus_builder import build_hierarchical_stylometry_dataset
from src.feature_extractor import StylometricFeatureExtractor
from src.gutenberg_fetcher import load_universal_canon_corpus
from src.literary_network import LiteraryPhylogenyEngine
from src.prosody_atmosphere import ProsodyAndAtmosphereAnalyzer
from src.style_coach import StylometricCoach
from src.academic_exporter import AcademicExporter


def get_blind_era_and_author_passages() -> List[Tuple[str, str, str, str]]:
    """
    Modelin eğitim setinde olmayan, 7 farklı edebi dönemi temsil eden kör test pasajları.
    (Pasaj_ID, Gerçek_Dönem, Gerçek_Yazar, Metin)
    """
    return [
        (
            "Test 1 (Middle English)",
            "Middle English",
            "Geoffrey Chaucer",
            "Whan that Aprille with his shoures soote the droghte of March hath perced to the roote, "
            "and bathed every veyne in swich licour of which vertu engendred is the flour; "
            "whan Zephirus eek with his sweete breeth inspired hath in every holt and heeth."
        ),
        (
            "Test 2 (Renaissance)",
            "Renaissance & Early Modern",
            "William Shakespeare",
            "To be, or not to be, that is the question: whether 'tis nobler in the mind to suffer "
            "the slings and arrows of outrageous fortune, or to take arms against a sea of troubles "
            "and by opposing end them. To die—to sleep, no more."
        ),
        (
            "Test 3 (Enlightenment)",
            "Enlightenment & 18th Century",
            "Jonathan Swift",
            "My father had a small estate in Nottinghamshire; I was the third of five sons. He sent me to Emanuel College "
            "in Cambridge, at fourteen years old, where I resided three years, and applied myself close to my studies."
        ),
        (
            "Test 4 (Romanticism)",
            "Romanticism",
            "Jane Austen",
            "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. "
            "However little known the feelings or views of such a man may be on his first entering a neighbourhood."
        ),
        (
            "Test 5 (Victorian)",
            "Victorian Era",
            "Charles Dickens",
            "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, "
            "it was the epoch of belief, it was the epoch of incredulity, it was the season of light, it was the season of darkness."
        ),
        (
            "Test 6 (American Classics)",
            "American Classics",
            "Edgar Allan Poe",
            "True!—nervous—very, very dreadfully nervous I had been and am; but why will you say that I am mad? "
            "The disease had sharpened my senses—not destroyed—not dulled them. Above all was the sense of hearing acute."
        ),
        (
            "Test 7 (Modernism)",
            "Modernism",
            "Virginia Woolf",
            "Mrs. Dalloway said she would buy the flowers herself. For Lucy had her work cut out for her. "
            "And then, thought Clarissa Dalloway, what a morning—fresh as if issued to children on a beach. What a lark! What a plunge!"
        ),
    ]


def main() -> None:
    print("=" * 95)
    print(" TÜBİTAK: EVRENSEL İNGİLİZ DİLİ VE EDEBİYATI STİLOMETRİ ATLASI (CHAUCER'DAN MODERNİZM'E)")
    print(" Developed by Bayram Akpamukçu | TÜBİTAK Computational Literary Studies & Stylometry Project")
    print(" 7 Edebi Dönem, 37 Kanonik Yazar, Hiyerarşik Koşullandırma (Hierarchical Conditioning) & Delta")
    print("=" * 95)

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data" / "raw"

    print("\n[1] Kanon Kataloğu ve Külliyat Hazırlanıyor...")
    corpus_dict = load_universal_canon_corpus(data_dir, try_download=False)
    print(f"    -> Yüklenen Kanonik Yazar Sayısı: {len(corpus_dict)}")
    print(f"    -> Kapsanan Edebi Dönem Sayısı : {len(LITERARY_ERAS)}")

    print("\n[2] Hiyerarşik Veri Seti Üretiliyor (Örneklem Bloklama)...")
    extractor = StylometricFeatureExtractor()
    df_X, s_y_author, s_y_era = build_hierarchical_stylometry_dataset(
        corpus_dict,
        extractor=extractor,
        chunk_size=90,
        overlap=35,
    )
    print(f"    -> Toplam Örnek Blok Sayısı: {len(df_X)}")
    print(f"    -> Çıkarılan Stilometrik Boyut: {df_X.shape[1]} öznitelik")

    print("\n[3] Çift Katmanlı Sınıflandırıcı (Linear SVM Pipeline) Eğitiliyor...")
    clf = StylometryClassifier(model_type="svm", cv_folds=3, random_state=42, feature_extractor=extractor)
    train_res = clf.train_hierarchical(df_X, s_y_author, s_y_era)

    print(f"    -> Katman 1 (Dönem) 3-Fold CV Doğruluğu: %{train_res['era_layer']['mean_accuracy']*100:.1f}")
    print(f"    -> Katman 2 (Yazar) 3-Fold CV Doğruluğu: %{train_res['author_layer']['mean_accuracy']*100:.1f}")

    print("\n" + "=" * 95)
    print(" [4] KÖR TEST: 7 EDEBİ DÖNEMDEN SEÇİLMİŞ METİNLERİN DÖNEM, YAZAR VE DELTA ANALİZİ")
    print("=" * 95)
    blind_tests = get_blind_era_and_author_passages()

    results_table = []
    for test_id, true_era, true_author, passage in blind_tests:
        pred_era, era_probs = clf.predict_era(passage)
        top_authors = clf.predict_top_authors(passage, top_k=3, use_hierarchical=True)
        delta_list = clf.compute_burrows_delta(passage, top_k=2)

        best_author, best_prob = top_authors[0]
        era_match = "✓" if pred_era == true_era else "✗"
        auth_match = "✓" if best_author == true_author else "✗"

        # Burrows' Delta 1. aday
        delta_top_author, delta_dist, delta_sim = delta_list[0]

        results_table.append([
            test_id,
            f"{true_era[:12]}..",
            f"{pred_era[:12]}.. ({era_match})",
            f"{true_author.split()[-1]}",
            f"{best_author.split()[-1]} ({auth_match}) %{best_prob*100:.0f}",
            f"{delta_top_author.split()[-1]} (Δ:{delta_dist:.2f})",
        ])

    headers = ["Test Pasajı", "Gerçek Dönem", "Tahmin Dönem", "Gerçek Yazar", "Tahmin Yazar", "Burrows' Delta En Yakın"]
    print(tabulate(results_table, headers=headers, tablefmt="fancy_grid"))

    print("\n[5] En Ayırt Edici Kanonik Stilometri Öznitelikleri (Top 10):")
    df_imp = clf.get_feature_importances(top_n=10)
    print(tabulate(df_imp, headers="keys", tablefmt="fancy_grid", showindex=True))

    print("\n" + "=" * 95)
    print(" [6] İLERİ DÜZEY ADLİ STİLOMETRİ, AI DEDEKTÖRÜ & 2B PCA GALAKSİSİ (AŞAMA 8)")
    print("=" * 95)
    forensics = AdvancedForensicsEngine(clf, extractor)

    # A. AI vs. Human Tespiti
    sample_human = blind_tests[1][3]
    ai_res = forensics.detect_ai_imitation(sample_human)
    print(f"\n-> AI vs. İnsan Dedektörü Doğrulaması (Shakespeare Pasajı):")
    print(f"   Karar: {ai_res['verdict']}")
    print(f"   Organik Üslup: %{ai_res['human_authenticity_pct']} | LLM Taklit Riski: %{ai_res['ai_probability_pct']}")
    print(f"   Burstiness CV: {ai_res['sentence_variation_cv']} | Hapax Oranı: {ai_res['hapax_ratio']}")

    # B. Rolling Stylometry
    collab_sample = (
        "Was this the face that launch'd a thousand ships, and burnt the topless towers of Ilium? "
        "Sweet Helen, make me immortal with a kiss! Her lips suck forth my soul; see where it flies! "
        "To be, or not to be, that is the question: whether 'tis nobler in the mind to suffer "
        "the slings and arrows of outrageous fortune, or to take arms against a sea of troubles."
    )
    segs = forensics.rolling_stylometry(collab_sample, "William Shakespeare", "Christopher Marlowe", window_words=20, step_words=10)
    print(f"\n-> Rolling Stylometry (Shakespeare vs. Marlowe - {len(segs)} Kayan Segment):")
    seg_table = [[s["segment_id"], s["word_range"], f"%{s['William Shakespeare']}", f"%{s['Christopher Marlowe']}", s["dominant_author"]] for s in segs]
    print(tabulate(seg_table, headers=["Segment #", "Aralık", "Shakespeare (%)", "Marlowe (%)", "Baskın Yazar"], tablefmt="fancy_grid"))

    # C. 2D PCA Galaksisi
    pca_data = forensics.compute_stylometric_galaxy_pca(df_X, list(s_y_author), list(s_y_era), query_text=sample_human)
    var1, var2 = pca_data["explained_variance"]
    q_pt = pca_data["query_point"]
    print(f"\n-> 2B PCA Üslup Galaksisi:")
    print(f"   37 Yazar Merkez Noktaları Hesaplandı. PC1: %{var1} | PC2: %{var2} varyans açıklandı.")
    print(f"   Test Metni Koordinatı: PC1={q_pt['PC1']}, PC2={q_pt['PC2']}")

    print("\n" + "=" * 95)
    print(" [7] EDEBİ SOY AĞACI, FILOGENETİK KÜMELEME & ÜSLUP İKİZLERİ (AŞAMA 9)")
    print("=" * 95)
    phylogeny = LiteraryPhylogenyEngine(df_X, list(s_y_author), list(s_y_era), metric="cosine")
    linkage_mat = phylogeny.compute_linkage_matrix(method="average")
    print(f"\n-> Hiyerarşik Edebi Soy Ağacı (Linkage Matrix):")
    print(f"   37 Kanonik Yazar için {linkage_mat.shape[0]} adet kümeleme birleşimi hesaplandı.")

    sample_target = "Jonathan Swift"
    soulmates = phylogeny.get_stylistic_soulmates(sample_target, top_k=3)
    print(f"\n-> Tarihsel Üslup İkizleri ({sample_target} İçin En Yakın 3 İkiz):")
    soul_table = [
        [idx, s["author"], s["era"], f"%{s['similarity_pct']}", s["distance"], s["relationship_type"][:38] + ".."]
        for idx, s in enumerate(soulmates, 1)
    ]
    print(tabulate(soul_table, headers=["#", "Üslup İkizi", "Dönemi", "Benzerlik", "Mesafe", "İlişki Türü"], tablefmt="fancy_grid"))

    print("\n" + "=" * 95)
    print(" [8] PROSODİ/VEZİN ANALİZİ & 6 BOYUTLU EDEBİ ATMOSFER RADARI (AŞAMA 10)")
    print("=" * 95)
    prosody = ProsodyAndAtmosphereAnalyzer()

    # Shakespeare Hamlet Prosodisi
    shak_sample = (
        "To be, or not to be, that is the question:\n"
        "Whether 'tis nobler in the mind to suffer\n"
        "The slings and arrows of outrageous fortune,\n"
        "Or to take arms against a sea of troubles,\n"
        "And by opposing end them."
    )
    shak_meter = prosody.analyze_poetic_meter(shak_sample)
    shak_atmos = prosody.analyze_literary_atmosphere(shak_sample)

    print(f"\n-> Shakespeare Prosodi & Hece Analizi (Hamlet Tiradı):")
    print(f"   Vezin Eğilimi : {shak_meter['poetic_form_tendency']}")
    print(f"   Ortalama Hece : {shak_meter['avg_syllables_per_line']} hece/dize | 10 Hece (Blank Verse): %{shak_meter['iambic_pentameter_pct']}")
    print(f"   Ritim Düzeni  : {shak_meter['meter_regularity_index']} / 100")

    # Poe Gotik Atmosferi
    poe_sample = blind_tests[5][3]
    poe_atmos = prosody.analyze_literary_atmosphere(poe_sample)

    print(f"\n-> 6 Boyutlu Edebi Atmosfer Profili Karşılaştırması (Shakespeare vs. Poe):")
    atmos_table = [
        [cat, f"%{shak_atmos.get(cat, 0.0):.1f}", f"%{poe_atmos.get(cat, 0.0):.1f}"]
        for cat in prosody.ATMOSPHERE_LEXICONS.keys()
    ]
    print(tabulate(atmos_table, headers=["Atmosfer Boyutu", "Shakespeare", "Edgar Allan Poe"], tablefmt="fancy_grid"))

    print("\n" + "=" * 95)
    print(" [9] ÜSLUP DÖNÜŞÜM REÇETESİ & YAZAR KOÇLUĞU (AŞAMA 11)")
    print("=" * 95)
    coach = StylometricCoach(classifier=clf, extractor=extractor)

    # Hedef: George Orwell için Zarf Budama ve Yalınlık Testi
    orwell_draft = (
        "He walked slowly and cautiously down the utterly dark corridor, breathing deeply and extremely nervously. "
        "The wind blew violently against the shattered window pane, fiercely shaking the fragile wooden frames. "
        "He quietly whispered into the shadows, hoping desperately that someone would quickly answer."
    )
    coach_orwell = coach.compare_with_target_author(orwell_draft, "George Orwell")
    rx_orwell = coach.generate_prescription(orwell_draft, "George Orwell")
    markers_orwell = coach.highlight_stylistic_markers(orwell_draft)

    print(f"\n-> Hedef Yazar: George Orwell (Uyum Skoru: %{coach_orwell['similarity_pct']} | Burrows' Delta: {coach_orwell['mean_delta_distance']})")
    print(f"   Metin Röntgeni: {markers_orwell['total_words']} Kelime | {markers_orwell['adverb_count']} Adet '-ly' Zarfı: {markers_orwell['adverbs_ly']}")
    print(f"\n-> En Büyük 5 Stilometrik Sapma (Z-Skor Farkları):")
    dev_table = [
        [d["friendly_name"], d["user_value"], d["target_value"], f"{'+' if d['z_diff']>0 else ''}{d['z_diff']}", d["direction"]]
        for d in coach_orwell["top_deviations"]
    ]
    print(tabulate(dev_table, headers=["Öznitelik", "Kullanıcı", "Orwell", "ΔZ", "Gerekli Eylem"], tablefmt="fancy_grid"))

    print(f"\n-> George Orwell Üslup Dönüşüm Reçetesi (Adım Adım Eylem Planı):")
    for idx, rx in enumerate(rx_orwell, 1):
        print(f"   [{idx}] [{rx['action']}] {rx['category']}: {rx['recommendation']}")
        print(f"       -> Etki: {rx['impact']}")

    print("\n" + "=" * 95)
    print(" [10] AKADEMİK VERİ İHRACI (LATEX / CSV) & ÇOKLU DOSYA ANALİZİ (AŞAMA 12)")
    print("=" * 95)
    exporter = AcademicExporter(classifier=clf, extractor=extractor)

    # 1. LaTeX Tablosu İhracı
    latex_code = exporter.export_latex_results_table(
        cv_results={"era_layer": train_res["era_layer"], "author_layer": train_res["author_layer"]},
    )
    print("\n-> Akademik LaTeX Tablosu (Önizleme):")
    for l_line in latex_code.splitlines()[:13]:
        print(f"   {l_line}")
    print("   ... [Tam LaTeX Tablosu Kodlandı]")

    # 2. CSV İhracı Doğrulaması
    csv_text = exporter.export_corpus_summary_csv(df_X, s_y_author, s_y_era)
    print(f"\n-> Külliyat Stilometrik Veri Seti (CSV):")
    print(f"   {len(df_X)} Örnek Blok, {df_X.shape[1]} Stilometrik Boyut, Toplam {len(csv_text.encode('utf-8')) / 1024:.1f} KB CSV üretildi.")

    # 3. Çoklu Dosya Toplu Taraması (Batch File Analysis)
    mock_files = {
        "orwell_1984_draft.txt": blind_tests[6][3],
        "shakespeare_soliloquy.txt": blind_tests[1][3],
        "dickens_tale_cities.txt": blind_tests[4][3],
    }
    df_batch = exporter.batch_analyze_files(mock_files)
    print(f"\n-> Çoklu Dosya Toplu Tarama Tablosu (Batch Analysis - {len(df_batch)} Dosya):")
    batch_table = [
        [row["Dosya Adı"], row["Kelime Sayısı"], row["Tahmin Edilen Dönem"], f"%{row['Dönem Güveni (%)']}", row["Tahmin Edilen Yazar"], f"%{row['Yazar Güveni (%)']}"]
        for _, row in df_batch.iterrows()
    ]
    print(tabulate(batch_table, headers=["Dosya Adı", "Kelime", "Tahmin Dönem", "Dönem Güveni", "Tahmin Yazar", "Yazar Güveni"], tablefmt="fancy_grid"))

    print("\n[BAŞARILI] Evrensel İngiliz Dili ve Edebiyatı Atlası (Aşama 12) boru hattı kusursuz çalıştı.")


if __name__ == "__main__":
    main()


