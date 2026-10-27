"""
TÜBİTAK Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası
===========================================================
Etkileşimli Streamlit Web Uygulaması (app.py)
Chaucer'dan Modernizm'e: 7 Edebi Dönem, 37 Kanonik Yazar, Hiyerarşik Sınıflandırma ve Burrows' Delta
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Windows konsolunda Türkçe ve Unicode karakterlerin düzgün görüntülenmesi için
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from src.advanced_forensics import AdvancedForensicsEngine
from src.canon_catalog import CANON_CATALOG, LITERARY_ERAS, AuthorEntry, get_all_eras, get_authors_by_era
from src.classifier import StylometryClassifier
from src.corpus_builder import build_hierarchical_stylometry_dataset
from src.feature_extractor import StylometricFeatureExtractor
from src.gutenberg_fetcher import load_universal_canon_corpus, sync_author_corpus
from src.literary_network import LiteraryPhylogenyEngine
from src.prosody_atmosphere import ProsodyAndAtmosphereAnalyzer
from src.visualizer import (
    plot_atmosphere_radar,
    plot_author_distance_heatmap,
    plot_author_stylistic_comparison,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_literary_dendrogram,
    plot_probability_bars,
    plot_rolling_stylometry,
    plot_stylometric_galaxy,
)

# ==============================================================================
# Sayfa Konfigürasyonu ve CSS Teması
# ==============================================================================
st.set_page_config(
    page_title="TÜBİTAK | İngiliz Edebiyatı Stilometri Atlası",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .era-badge {
        display: inline-block;
        background-color: #3b82f6;
        color: white;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .prediction-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        color: white;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin: 15px 0 20px 0;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15);
    }
    .prediction-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        margin: 8px 0;
    }
    .era-title {
        font-size: 1.3rem;
        color: #93c5fd;
        font-weight: 600;
    }
    .kpi-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .kpi-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1e3a8a;
    }
    .kpi-lbl {
        font-size: 0.82rem;
        color: #64748b;
        font-weight: 600;
    }
    .dev-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #38bdf8;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.88rem;
        font-weight: 700;
        border: 1px solid rgba(56, 189, 248, 0.4);
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
        margin-bottom: 16px;
    }
    .ai-detector-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        border: 1px solid #6366f1;
        border-radius: 12px;
        padding: 18px 22px;
        color: #ffffff;
        margin: 16px 0;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.2);
    }
    .dev-footer {
        text-align: center;
        padding: 24px 0 16px 0;
        margin-top: 40px;
        border-top: 1px solid #e2e8f0;
        color: #64748b;
        font-size: 0.92rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Model ve Külliyat Başlatma (Streamlit Cache)
# ==============================================================================
@st.cache_resource(show_spinner="Evrensel Edebiyat Külliyatı taranıyor ve çift katmanlı model eğitiliyor...")
def initialize_universal_atlas(
    model_type: str = "svm",
    chunk_size: int = 90,
    overlap: int = 35,
) -> Tuple[StylometryClassifier, Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    37 kanonik yazarın edebi metinlerini hazırlar, bloklara böler ve
    çift katmanlı (Dönem + Yazar) sınıflandırıcıyı eğitir.
    """
    data_dir = Path(__file__).resolve().parent / "data" / "raw"
    corpus_dict = load_universal_canon_corpus(data_dir, try_download=False)

    extractor = StylometricFeatureExtractor()
    df_X, s_y_author, s_y_era = build_hierarchical_stylometry_dataset(
        corpus_dict,
        extractor=extractor,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    X_train, X_test, y_auth_train, y_auth_test, y_era_train, y_era_test = train_test_split(
        df_X,
        s_y_author,
        s_y_era,
        test_size=0.20,
        random_state=42,
        stratify=s_y_era,
    )

    clf = StylometryClassifier(
        model_type=model_type,
        cv_folds=3,
        random_state=42,
        feature_extractor=extractor,
    )

    training_results = clf.train_hierarchical(X_train, y_auth_train, y_era_train)
    eval_results = clf.evaluate(X_test, y_auth_test)

    atlas_info = {
        "total_samples": len(df_X),
        "feature_count": df_X.shape[1],
        "total_authors": len(set(s_y_author)),
        "total_eras": len(set(s_y_era)),
        "authors_list": sorted(list(set(s_y_author))),
        "eras_list": LITERARY_ERAS,
        "author_cv": training_results["author_layer"],
        "era_cv": training_results["era_layer"],
        "eval_results": eval_results,
        "df_X": df_X,
        "y_authors": s_y_author,
        "y_eras": s_y_era,
    }

    return clf, atlas_info, X_train, X_test, y_auth_train, y_auth_test, y_era_train, y_era_test


# ==============================================================================
# Hızlı Test İçin Kanonik Edebi Pasaj Şablonları
# ==============================================================================
PRELOADED_PASSAGES = {
    "Seçiniz (Hazır Kanonik Test Pasajı)": "",
    "Middle English: Geoffrey Chaucer (Canterbury Tales)": (
        "Whan that Aprille with his shoures soote the droghte of March hath perced to the roote, "
        "and bathed every veyne in swich licour of which vertu engendred is the flour; "
        "whan Zephirus eek with his sweete breeth inspired hath in every holt and heeth the tendre croppes."
    ),
    "Renaissance: William Shakespeare (Hamlet)": (
        "To be, or not to be, that is the question: whether 'tis nobler in the mind to suffer "
        "the slings and arrows of outrageous fortune, or to take arms against a sea of troubles "
        "and by opposing end them. To die—to sleep, no more; and by a sleep to say we end "
        "the heart-ache and the thousand natural shocks that flesh is heir to."
    ),
    "Renaissance: Christopher Marlowe (Doctor Faustus)": (
        "Was this the face that launch'd a thousand ships, and burnt the topless towers of Ilium? "
        "Sweet Helen, make me immortal with a kiss! Her lips suck forth my soul; see where it flies! "
        "Come, Helen, come, give me my soul again. Here will I dwell, for heaven is in these lips."
    ),
    "Renaissance: John Milton (Paradise Lost)": (
        "Of Man's first disobedience, and the fruit of that forbidden tree, whose mortal taste "
        "brought death into the World, and all our woe, with loss of Eden, till one greater Man "
        "restore us, and regain the blissful seat, sing Heavenly Muse."
    ),
    "Enlightenment: Jonathan Swift (Gulliver's Travels)": (
        "My father had a small estate in Nottinghamshire; I was the third of five sons. He sent me to Emanuel College "
        "in Cambridge, at fourteen years old, where I resided three years, and applied myself close to my studies; "
        "but the charge of maintaining me, being too great for a narrow fortune, I was bound apprentice to Mr. Bates."
    ),
    "Romanticism: Jane Austen (Pride and Prejudice)": (
        "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. "
        "However little known the feelings or views of such a man may be on his first entering a neighbourhood, "
        "this truth is so well fixed in the minds of the surrounding families."
    ),
    "Romanticism: Mary Shelley (Frankenstein)": (
        "It was on a dreary night of November that I beheld the accomplishment of my toils. With an anxiety that "
        "almost amounted to agony, I collected the instruments of life around me, that I might infuse a spark of being "
        "into the lifeless thing that lay at my feet. The rain pattered dismally against the panes."
    ),
    "Victorian: Charles Dickens (A Tale of Two Cities)": (
        "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, "
        "it was the epoch of belief, it was the epoch of incredulity, it was the season of light, it was the season of darkness, "
        "it was the spring of hope, it was the winter of despair, we had everything before us."
    ),
    "Victorian: Oscar Wilde (The Picture of Dorian Gray)": (
        "The studio was filled with the rich odour of roses, and when the light summer wind stirred amidst the trees of the garden, "
        "there came through the open door the heavy scent of the lilac, or the more delicate perfume of the pink-flowering thorn. "
        "The only way to get rid of a temptation is to yield to it."
    ),
    "American Classics: Edgar Allan Poe (The Tell-Tale Heart)": (
        "True!—nervous—very, very dreadfully nervous I had been and am; but why will you say that I am mad? "
        "The disease had sharpened my senses—not destroyed—not dulled them. Above all was the sense of hearing acute. "
        "I heard all things in the heaven and in the earth. Hearken! and observe how healthily I tell you the story."
    ),
    "American Classics: Herman Melville (Moby Dick)": (
        "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse, "
        "and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. "
        "Towards thee I roll, thou all-destroying but unconquering whale; to the last I grapple with thee!"
    ),
    "Modernism: Virginia Woolf (Mrs. Dalloway)": (
        "Mrs. Dalloway said she would buy the flowers herself. For Lucy had her work cut out for her. "
        "And then, thought Clarissa Dalloway, what a morning—fresh as if issued to children on a beach. "
        "What a lark! What a plunge! For so it had always seemed to her, with a little squeak of the hinges."
    ),
    "Modernism: James Joyce (Ulysses)": (
        "Stately, plump Buck Mulligan came from the stairhead, bearing a bowl of lather on which a mirror and a razor lay crossed. "
        "A yellow dressinggown, ungirdled, was sustained gently behind him on the mild morning air. He held the bowl aloft and intoned: "
        "—Introibo ad altare Dei. Halted, he peered down the dark winding stairs."
    ),
    "Modernism: Ernest Hemingway (A Farewell to Arms)": (
        "In the late summer of that year we lived in a house in a village that looked across the river and the plain to the mountains. "
        "In the bed of the river there were pebbles and boulders, dry and white in the sun, and the water was clear and swiftly moving."
    ),
}


# ==============================================================================
# Başlık ve Kenar Çubuğu (Sidebar)
# ==============================================================================
st.markdown('<div class="main-header">🏛️ TÜBİTAK: Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Chaucer\'dan Modernizm\'e 7 Edebi Dönem, 37 Kanonik Yazar, '
    'Çift Katmanlı Sınıflandırma ve Burrows\' Delta Z-Skor Mesafesi Analizi</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="dev-badge">👨‍💻 Developed by Bayram Akpamukçu | TÜBİTAK Computational Literary Studies & Stylometry Project</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/tr/b/b8/T%C3%BCbitak_logo.png", width=140)
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #38bdf8; padding: 10px 14px; border-radius: 8px; font-weight: 700; font-size: 0.85rem; text-align: center; border: 1px solid rgba(56, 189, 248, 0.35); margin: 8px 0 16px 0;">
            ✨ Developed by Bayram Akpamukçu
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### ⚙️ Atlas Yapılandırması")

    selected_model_type = st.selectbox(
        "Sınıflandırma Motoru",
        options=["svm", "rf"],
        format_func=lambda x: "Linear SVM (Destek Vektör Makinesi)" if x == "svm" else "Random Forest (Rastgele Orman)",
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📚 Külliyat ve Dönem Kapsamı")

# Modeli Eğit / Yükle
clf, atlas_info, X_train, X_test, y_auth_train, y_auth_test, y_era_train, y_era_test = initialize_universal_atlas(
    model_type=selected_model_type,
    chunk_size=90,
    overlap=35,
)

with st.sidebar:
    st.write(f"• **Kapsanan Dönem Sayısı:** `{atlas_info['total_eras']}` Edebi Dönem")
    st.write(f"• **Kanonik Yazar Sayısı:** `{atlas_info['total_authors']}` Yazar")
    st.write(f"• **Toplam Örnek Blok:** `{atlas_info['total_samples']}` Metin Bloğu")
    st.write(f"• **Stilometrik Boyut:** `{atlas_info['feature_count']}` Öznitelik")
    st.markdown("---")
    st.markdown(
        """
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px; text-align: center; color: #475569; font-size: 0.8rem; font-weight: 600;">
            🚀 <b>Developed by Bayram Akpamukçu</b><br>
            TÜBİTAK 2209 / Computational Literary Studies
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# ALTI ANA SEKME (TABS)
# ==============================================================================
tab_prediction, tab_catalog, tab_comparison, tab_performance, tab_forensics, tab_network = st.tabs([
    "🔍 Sekme 1: Canlı Yazar & Dönem Tespiti",
    "📖 Sekme 2: Edebiyat Tarihi & Külliyat Gezgini",
    "⚖️ Sekme 3: Üslup Evrimi & Yazar Karşılaştırma",
    "📊 Sekme 4: Model Performansı & Birim Testler",
    "🕵️‍♂️ Sekme 5: Adli Stilometri & 2B Üslup Galaksisi",
    "🌳 Sekme 6: Edebi Soy Ağacı & Etki Ağı",
])


# ------------------------------------------------------------------------------
# SEKME 1: CANLI YAZAR & DÖNEM TESPİTİ
# ------------------------------------------------------------------------------
with tab_prediction:
    st.markdown("### ✍️ Bilinmeyen / Anonim Bir Edebi Metin Giriniz")
    st.markdown(
        "Aşağıdaki alana İngiliz veya Amerikan edebiyatından bir pasaj yapıştırın "
        "veya hazır test pasajlarından birini seçerek analizi başlatın."
    )

    col_pick, col_era_filter = st.columns([3, 2])
    with col_pick:
        chosen_sample = st.selectbox(
            "Hazır Kanonik Test Pasajı Seçin:",
            options=list(PRELOADED_PASSAGES.keys()),
            index=0,
        )
    with col_era_filter:
        era_constraint = st.selectbox(
            "Dönem Kısıtı (Opsiyonel):",
            options=["Tüm Dönemler (Otomatik Katman 1)"] + LITERARY_ERAS,
            index=0,
            help="İsterseniz tahmin Katman 2'de yalnızca seçtiğiniz dönemin yazarları arasında sınırlandırılır.",
        )

    use_hierarchical = st.checkbox(
        "🧠 Hiyerarşik Dönem Filtresi (Hierarchical Conditioning)",
        value=True,
        help="Açık olduğunda Katman 1 edebi dönem olasılıkları ve Burrows' Delta yakınlığı Katman 2 yazar olasılıklarıyla birleştirilerek anakronik (yüzyıl dışı) sapmalar engellenir.",
    )

    initial_input = PRELOADED_PASSAGES[chosen_sample] if chosen_sample != "Seçiniz (Hazır Kanonik Test Pasajı)" else ""

    text_input = st.text_area(
        "Analiz Edilecek Edebi Metin:",
        value=initial_input,
        height=160,
        placeholder="Örn: To be, or not to be, that is the question: whether 'tis nobler in the mind to suffer...",
    )

    run_btn = st.button("🚀 Çift Katmanlı Analizi ve Burrows' Delta'yı Çalıştır", type="primary", use_container_width=True)

    if run_btn:
        if not text_input.strip():
            st.warning("⚠️ Lütfen analiz edilmesi için en az bir cümlelik bir edebi metin giriniz.")
        else:
            with st.spinner("Katman 1 (Dönem) ve Katman 2 (Yazar) sınıflandırıcıları çalışıyor, Burrows' Delta hesaplanıyor..."):
                # 1. Katman 1: Edebi Dönem Tahmini
                predicted_era, era_probabilities = clf.predict_era(text_input)
                era_conf = era_probabilities.get(predicted_era, 1.0)

                # 2. Katman 2: Yazar Tahmini (Top-5 Adaylar)
                active_filter = None if era_constraint.startswith("Tüm") else era_constraint
                top_authors = clf.predict_top_authors(
                    text_input,
                    era_filter=active_filter,
                    top_k=5,
                    use_hierarchical=use_hierarchical,
                )
                best_author, best_author_prob = top_authors[0]

                # 3. Burrows' Delta Z-Skor Mesafesi
                delta_results = clf.compute_burrows_delta(text_input, top_k=5)

                # 4. Anlık Metrik Röntgeni
                extractor = StylometricFeatureExtractor()
                text_metrics = extractor.extract_features(text_input)

            hierarchical_badge = (
                '<span style="background: rgba(56, 189, 248, 0.25); color: #38bdf8; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; margin-left: 8px;">🧠 Hiyerarşik Koşullandırma: AKTİF</span>'
                if use_hierarchical else
                '<span style="background: rgba(148, 163, 184, 0.2); color: #94a3b8; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; margin-left: 8px;">Ham Tekil Model (Filtresiz)</span>'
            )

            # Büyük İkili Tahmin Kartı
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="era-title">🏛️ KATMAN 1 (EDEBİ DÖNEM): {predicted_era.upper()} (%{era_conf*100:.1f} Güven)</div>
                    <div style="font-size: 1.1rem; opacity: 0.9; margin-top: 6px;">KATMAN 2 (EN GÜÇLÜ YAZAR ADAYI) {hierarchical_badge}</div>
                    <div class="prediction-title">✍️ {best_author}</div>
                    <div style="font-size: 1.05rem; opacity: 0.95;">Model Olasılığı: %{best_author_prob*100:.1f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # İki Kolonlu Detay Analiz
            col_left, col_right = st.columns([1.1, 1], gap="medium")

            with col_left:
                st.markdown("#### 🎯 En Olası 5 Yazar Adayı (Top-5 Authors)")
                top_authors_dict = {auth: prob for auth, prob in top_authors}
                fig_probs = plot_probability_bars(top_authors_dict)
                st.pyplot(fig_probs)
                plt.close(fig_probs)

                st.markdown("#### 📐 Burrows' Delta Z-Skor Mesafesi (Altın Standart)")
                st.caption("Daha düşük delta mesafesi = Metne üslupça en yakın yazar.")
                delta_df = pd.DataFrame(
                    delta_results,
                    columns=["Yazar", "Burrows' Delta Mesafesi", "Benzerlik İndeksi (%)"]
                )
                st.dataframe(delta_df, use_container_width=True, hide_index=True)

            with col_right:
                st.markdown("#### 🔬 Metnin Edebi & Kronolojik Röntgeni")
                k1, k2 = st.columns(2)
                with k1:
                    st.metric("Arkaik Sözcük Oranı", f"%{text_metrics['archaic_total_ratio']*100:.2f}",
                              help="thou, thee, hath, doth, 'tis gibi arkaik kelimelerin toplam kelimelere oranı.")
                    st.metric("-eth / -est Arkaik Fiil", f"%{(text_metrics['suffix_eth_ratio']+text_metrics['suffix_est_ratio'])*100:.2f}",
                              help="Rönesans ve Orta Çağ fiil çekimleri.")
                    st.metric("Bilinç Akışı İndeksi", f"{text_metrics['stream_of_consciousness_index']:.2f}",
                              help="Tire, üç nokta ve cümle varyansı kombinasyonu (Modernizm göstergesi).")
                with k2:
                    st.metric("Sözcük Çeşitliliği (TTR)", f"{text_metrics['ttr']:.3f}",
                              help="Vocabulary Richness (Benzersiz kelime / Toplam kelime).")
                    st.metric("-ly Zarf Yoğunluğu", f"%{text_metrics['suffix_ly_ratio']*100:.2f}",
                              help="Viktorya ve Romantizm dönemi betimleme göstergesi.")
                    st.metric("Ortalama Cümle Uzunluğu", f"{text_metrics['avg_sentence_length_words']:.1f} kelime")

                st.info(
                    f"💡 **NLP Değerlendirmesi:** Metin toplam **{int(text_metrics['total_tokens'])}** kelime "
                    f"ve **{int(text_metrics['total_sentences'])}** cümledir. "
                    f"Burrows' Delta mesafesine göre metin en çok **{delta_results[0][0]}** üslubuna yaklaşmaktadır "
                    f"(Delta: {delta_results[0][1]})."
                )

                # 5. Adli Yapay Zeka (LLM) vs. Gerçek Yazar Doğrulaması
                forensics = AdvancedForensicsEngine(clf, extractor)
                ai_analysis = forensics.detect_ai_imitation(text_input)
                forensic_report_md = forensics.generate_forensic_report_markdown(
                    text=text_input,
                    predicted_era=predicted_era,
                    top_authors=top_authors,
                    ai_analysis=ai_analysis,
                )
                st.session_state["last_query_text"] = text_input

                st.markdown("---")
                st.markdown("### 🤖 Yapay Zeka (LLM) vs. Gerçek Yazar Doğrulaması")

                ai_badge_color = "#10b981" if ai_analysis["human_authenticity_pct"] >= 55.0 else "#ef4444"
                st.markdown(
                    f"""
                    <div class="ai-detector-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
                            <span style="font-size: 1.15rem; font-weight: 700; color: #f8fafc;">Adli Stilometrik Karar:</span>
                            <span style="background: {ai_badge_color}; color: #ffffff; padding: 4px 14px; border-radius: 20px; font-size: 0.9rem; font-weight: 700;">
                                {ai_analysis['verdict']}
                            </span>
                        </div>
                        <div style="font-size: 0.95rem; color: #cbd5e1; margin-bottom: 14px;">
                            Metnin sözcüksel varyansı (Burstiness), cümle boyutu dalgalanması ve Hapax Legomena (nadir tekil kelime)
                            entropisi analiz edilerek sentetik makine üretimi (ChatGPT/LLM pastiche) olup olmadığı doğrulanmıştır.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_ai1, col_ai2, col_ai3, col_ai4 = st.columns(4)
                with col_ai1:
                    st.metric("🌿 Organik İnsan Üslubu", f"%{ai_analysis['human_authenticity_pct']:.1f}")
                with col_ai2:
                    st.metric("🤖 LLM Taklit Riski", f"%{ai_analysis['ai_probability_pct']:.1f}")
                with col_ai3:
                    st.metric("📈 Cümle Ritim Varyansı (CV)", f"{ai_analysis['sentence_variation_cv']:.2f}",
                              help="İnsan yazarlarında cümle uzunluğu dalgalanması (CV) çok daha yüksektir.")
                with col_ai4:
                    st.metric("💎 Hapax Legomena Oranı", f"{ai_analysis['hapax_ratio']:.3f}",
                              help="Metindeki tekil/nadir sözcük zenginliği.")

                st.markdown("---")
                col_rep_btn, col_rep_info = st.columns([1.3, 2])
                with col_rep_btn:
                    st.download_button(
                        label="📥 Resmi TÜBİTAK Adli Stilometri Raporunu İndir (.md)",
                        data=forensic_report_md,
                        file_name=f"TUBITAK_Adli_Rapor_{best_author.replace(' ', '_')}.md",
                        mime="text/markdown",
                        type="primary",
                        use_container_width=True,
                    )
                with col_rep_info:
                    st.caption(
                        "📄 Bu resmi adli rapor; çift katmanlı hiyerarşik sınıflandırma sonuçlarını, "
                        "Burrows' Delta z-skorlarını ve AI pastiche analizlerini 'Developed by Bayram Akpamukçu' "
                        "resmi imzasıyla belgelendirir."
                    )

                # 6. Prosodi & Hece Ritmi ve 6 Boyutlu Edebi Atmosfer Radarı
                prosody_analyzer = ProsodyAndAtmosphereAnalyzer()
                meter_info = prosody_analyzer.analyze_poetic_meter(text_input)
                atmosphere_scores = prosody_analyzer.analyze_literary_atmosphere(text_input)

                st.markdown("---")
                st.markdown("### 🕸️ 6 Boyutlu Edebi Atmosfer Radarı & Prosodi Analizi")

                col_rad, col_pros = st.columns([1.1, 1], gap="medium")
                with col_rad:
                    st.markdown("#### 🧭 Edebi Atmosfer Radarı (Spider Chart)")
                    fig_radar = plot_atmosphere_radar(
                        atmosphere_scores,
                        title=f"Metnin Edebi Atmosfer Profili ({best_author} Eşleşmesi)",
                        label1="Test Metni",
                    )
                    st.pyplot(fig_radar)
                    plt.close(fig_radar)

                with col_pros:
                    st.markdown("#### 🎭 Prosodi, Vezin & Hece Ritmi")
                    st.markdown(
                        f"""
                        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; margin-bottom: 12px;">
                            <div style="font-size: 0.82rem; color: #64748b; font-weight: 700; text-transform: uppercase;">Tespit Edilen Vezin Eğilimi</div>
                            <div style="font-size: 1.05rem; font-weight: 800; color: #1e293b; margin-top: 4px;">{meter_info['poetic_form_tendency']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    p1, p2 = st.columns(2)
                    with p1:
                        st.metric("Toplam Hece Sayısı", f"{meter_info['total_syllables']}")
                        st.metric("10 Hece (Blank Verse) %", f"%{meter_info['iambic_pentameter_pct']:.1f}",
                                  help="Shakespeare ve Milton blank verse dizeleri ~10 hece (Iambic Pentameter) kalıbındadır.")
                    with p2:
                        st.metric("Ortalama Hece / Dize", f"{meter_info['avg_syllables_per_line']:.1f}")
                        st.metric("Ritim Düzenlilik İndeksi", f"{meter_info['meter_regularity_index']:.1f}",
                                  help="Hece varyasyonuna göre metnin ritmik kararlılığı (0-100).")

                    # Atmosfer Dağılımı Tablosu
                    st.markdown("##### 📊 6 Atmosfer Boyutu Puanları")
                    df_atmos = pd.DataFrame([
                        {"Atmosfer Boyutu": k, "Puan": f"%{v:.1f}"}
                        for k, v in atmosphere_scores.items()
                    ])
                    st.dataframe(df_atmos, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------------------
# SEKME 2: EDEBİYAT TARİHİ & KÜLLİYAT GEZGİNİ
# ------------------------------------------------------------------------------
with tab_catalog:
    st.markdown("### 📚 7 Edebi Dönem ve 37 Kanonik Yazar Kataloğu")
    st.markdown(
        "Bu atlas, İngiliz ve Amerikan edebiyatının Orta Çağ'dan 20. yüzyıl ortasına kadar uzanan "
        "en kapsamlı stilometri külliyatlarından biridir. Her yazarın Project Gutenberg kayıtları ve "
        "temsil ettiği üslup özellikleri aşağıda listelenmiştir."
    )

    filter_era = st.selectbox(
        "Döneme Göre Filtrele:",
        options=["Tüm Dönemleri Göster"] + LITERARY_ERAS,
        index=0,
    )

    catalog_rows = []
    for auth_id, entry in CANON_CATALOG.items():
        if filter_era != "Tüm Dönemleri Göster" and entry.era != filter_era:
            continue
        catalog_rows.append({
            "Yazar": entry.name,
            "Edebi Dönem": entry.era,
            "Yüzyıl / Tarih": entry.century_span,
            "Kanonik Eser": entry.canonical_work,
            "Gutenberg ID": entry.gutenberg_id,
            "Üslup Karakteristiği": entry.stylistic_profile,
        })

    df_catalog = pd.DataFrame(catalog_rows)
    st.dataframe(df_catalog, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🔄 Project Gutenberg Canlı Senkronizasyon Aracı")
    col_sync_btn, col_sync_info = st.columns([1, 2])
    with col_sync_btn:
        sync_btn = st.button("🌐 Gutenberg Tam Metinlerini Eşitle (İsteğe Bağlı)")
    with col_sync_info:
        st.caption(
            "Varsayılan olarak sistem, çevrimdışı yüksek doğruluklu yerleşik pasaj külliyatını kullanır. "
            "Butona tıklandığında çevrimiçi Gutenberg kütüphanesinden tam metinler taranır."
        )

    if sync_btn:
        with st.spinner("Gutenberg arşivine bağlanılıyor ve metinler güncelleniyor..."):
            data_dir = Path(__file__).resolve().parent / "data" / "raw"
            # 2 örnek yazar için deneme senkronizasyonu
            sample_entry = CANON_CATALOG["shakespeare"]
            p = sync_author_corpus(data_dir, sample_entry, try_download=True)
            st.success(f"✓ Gutenberg eşitlemesi tamamlandı. Külliyat yolu: `{p.name}`")


# ------------------------------------------------------------------------------
# SEKME 3: ÜSLUP EVRİMİ & YAZAR KARŞILAŞTIRMA
# ------------------------------------------------------------------------------
with tab_comparison:
    st.markdown("### ⚖️ İki Kanonik Yazarın Üslup Karşılaştırması")
    st.markdown(
        "Farklı veya aynı dönemden iki yazar seçerek sözdizimsel, morfolojik ve arkaik "
        "özniteliklerini yan yana karşılaştırın (Örn: *Shakespeare vs. Marlowe* veya *Dickens vs. Hemingway*)."
    )

    all_author_names = sorted(list(set(atlas_info["authors_list"])))
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        author1 = st.selectbox("1. Yazar:", options=all_author_names, index=all_author_names.index("William Shakespeare") if "William Shakespeare" in all_author_names else 0)
    with col_a2:
        default_idx2 = all_author_names.index("Ernest Hemingway") if "Ernest Hemingway" in all_author_names else 1
        author2 = st.selectbox("2. Yazar:", options=all_author_names, index=default_idx2)

    # İki yazarın ortalama özniteliklerini hesapla
    feat_extractor = StylometricFeatureExtractor()
    auth1_entry = next((e for e in CANON_CATALOG.values() if e.name == author1), None)
    auth2_entry = next((e for e in CANON_CATALOG.values() if e.name == author2), None)

    if auth1_entry and auth2_entry:
        text1 = " ".join(auth1_entry.sample_passages)
        text2 = " ".join(auth2_entry.sample_passages)

        f1 = feat_extractor.extract_features(text1)
        f2 = feat_extractor.extract_features(text2)

        fig_comp = plot_author_stylistic_comparison(
            author1_name=author1,
            features1=f1,
            author2_name=author2,
            features2=f2,
            title=f"Stilometri Karşılaştırması: {author1} vs. {author2}",
        )
        st.pyplot(fig_comp)
        plt.close(fig_comp)

        # Karşılaştırmalı Tablo
        st.markdown("#### 📋 Sayısal Göstergeler Tablosu")
        comp_data = [
            {"Metrik": "Edebi Dönem", author1: auth1_entry.era, author2: auth2_entry.era},
            {"Metrik": "Ortalama Cümle Uzunluğu (Kelime)", author1: f"{f1['avg_sentence_length_words']:.1f}", author2: f"{f2['avg_sentence_length_words']:.1f}"},
            {"Metrik": "Sözcük Çeşitliliği (TTR)", author1: f"{f1['ttr']:.3f}", author2: f"{f2['ttr']:.3f}"},
            {"Metrik": "Arkaik Sözcük Oranı", author1: f"%{f1['archaic_total_ratio']*100:.2f}", author2: f"%{f2['archaic_total_ratio']*100:.2f}"},
            {"Metrik": "-ly Zarf Yoğunluğu", author1: f"%{f1['suffix_ly_ratio']*100:.2f}", author2: f"%{f2['suffix_ly_ratio']*100:.2f}"},
            {"Metrik": "-ing Süreklilik Oranı", author1: f"%{f1['suffix_ing_ratio']*100:.2f}", author2: f"%{f2['suffix_ing_ratio']*100:.2f}"},
            {"Metrik": "Bilinç Akışı İndeksi", author1: f"{f1['stream_of_consciousness_index']:.2f}", author2: f"{f2['stream_of_consciousness_index']:.2f}"},
            {"Metrik": "Noktalı Virgül (;) Oranı", author1: f"{f1['punct_semicolon_ratio']:.3f}", author2: f"{f2['punct_semicolon_ratio']:.3f}"},
        ]
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

        # 6 Boyutlu Edebi Atmosfer Karşılaştırma Radarı (Spider Chart)
        prosody_comp = ProsodyAndAtmosphereAnalyzer()
        atmos1 = prosody_comp.analyze_literary_atmosphere(text1)
        atmos2 = prosody_comp.analyze_literary_atmosphere(text2)
        meter1 = prosody_comp.analyze_poetic_meter(text1)
        meter2 = prosody_comp.analyze_poetic_meter(text2)

        st.markdown("---")
        st.markdown("#### 🕸️ 6 Boyutlu Edebi Atmosfer Radarı Karşılaştırması (Spider Chart)")

        col_c_rad, col_c_pros = st.columns([1.1, 1], gap="medium")
        with col_c_rad:
            fig_rad_comp = plot_atmosphere_radar(
                atmosphere_scores=atmos1,
                comparison_scores=atmos2,
                label1=author1,
                label2=author2,
                title=f"Edebi Atmosfer Karşılaştırması: {author1} vs. {author2}",
            )
            st.pyplot(fig_rad_comp)
            plt.close(fig_rad_comp)

        with col_c_pros:
            st.markdown("#### 🎭 Prosodi ve Hece Vezin Karşılaştırması")
            pros_data = [
                {"Metrik": "Vezin / Ritim Eğilimi", author1: meter1["poetic_form_tendency"], author2: meter2["poetic_form_tendency"]},
                {"Metrik": "Ortalama Hece / Dize", author1: f"{meter1['avg_syllables_per_line']:.1f}", author2: f"{meter2['avg_syllables_per_line']:.1f}"},
                {"Metrik": "10 Hece (Iambic Pentameter) %", author1: f"%{meter1['iambic_pentameter_pct']:.1f}", author2: f"%{meter2['iambic_pentameter_pct']:.1f}"},
                {"Metrik": "Ritim Düzenlilik İndeksi", author1: f"{meter1['meter_regularity_index']:.1f}", author2: f"{meter2['meter_regularity_index']:.1f}"},
                {"Metrik": "Baskın Edebi Atmosfer", author1: max(atmos1.items(), key=lambda x: x[1])[0], author2: max(atmos2.items(), key=lambda x: x[1])[0]},
            ]
            st.dataframe(pd.DataFrame(pros_data), use_container_width=True, hide_index=True)


# ------------------------------------------------------------------------------
# SEKME 4: MODEL PERFORMANSI & BİRİM TESTLER
# ------------------------------------------------------------------------------
with tab_performance:
    st.markdown("### 📊 Çift Katmanlı Model Performansı ve QA Doğrulaması")

    # Üst KPI Kartları
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-val">%{atlas_info['era_cv']['mean_accuracy']*100:.1f}</div>
                <div class="kpi-lbl">Katman 1 (Dönem) CV Doğruluğu</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-val">%{atlas_info['eval_results']['accuracy']*100:.1f}</div>
                <div class="kpi-lbl">Katman 2 (Yazar) Test Doğruluğu</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-val">{atlas_info['author_cv']['mean_f1_macro']:.3f}</div>
                <div class="kpi-lbl">Katman 2 Macro F1 Skoru</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-val">{atlas_info['feature_count']}</div>
                <div class="kpi-lbl">Toplam Stilometrik Boyut</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    col_cm, col_imp = st.columns([1, 1], gap="medium")
    with col_cm:
        st.markdown("#### 🧩 Katman 2 Karmaşıklık Matrisi (Seaborn Heatmap)")
        y_test_pred = clf.pipeline.predict(X_test)
        fig_cm = plot_confusion_matrix(
            y_true=y_auth_test,
            y_pred=y_test_pred,
            classes=sorted(list(set(y_auth_test))),
            title="Kanonik Yazarlar Test Kümesi Karmaşıklık Matrisi",
        )
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with col_imp:
        st.markdown("#### 🏆 Yazarları En Çok Ayırt Eden 12 Öznitelik")
        df_imp = clf.get_feature_importances(top_n=12)
        fig_imp = plot_feature_importance(df_imp, top_n=12, title="En Ayırt Edici Göstergeler (|SVM Katsayısı|)")
        st.pyplot(fig_imp)
        plt.close(fig_imp)

    st.markdown("---")
    st.markdown("#### 🧪 TÜBİTAK Birim Testleri Doğrulama Konsolu")
    test_run_btn = st.button("Birim Testleri Doğrudan Çalıştır (tests/ Suite)")
    if test_run_btn:
        with st.spinner("unittest discover -s tests çalıştırılıyor..."):
            loader = unittest.TestLoader()
            suite = loader.discover("tests")
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            if result.wasSuccessful():
                st.success(f"✓ TÜM BİRİM TESTLER BAŞARIYLA GEÇTİ! ({result.testsRun} test çalıştırıldı, 0 hata, 0 başarısızlık).")
            else:
                st.error(f"Testler tamamlandı: {len(result.failures)} başarısızlık, {len(result.errors)} hata.")


# ------------------------------------------------------------------------------
# SEKME 5: ADLİ STİLOMETRİ & 2B ÜSLUP GALAKSİSİ
# ------------------------------------------------------------------------------
with tab_forensics:
    st.markdown("### 🕵️‍♂️ İleri Düzey Adli Stilometri & 2B Üslup Galaksisi")
    st.markdown(
        "Bu modül, adli edebiyat araştırmalarında (Computational Forensics) kullanılan iki ileri düzey yöntemi sunar: "
        "**Rolling Stylometry** (sahne/bölüm bazlı ortak yazarlık ayrıştırması) ve **2B PCA Üslup Galaksisi** "
        "(70+ boyutlu üslup uzayının 2 boyuta izdüşümü)."
    )

    st.markdown("---")
    st.markdown("#### 🔬 Alt Bölüm A: Ortak Yazarlık Tarayıcısı (Rolling Stylometry)")
    st.markdown(
        "Kayan pencere (Sliding Window) tekniği, özellikle ortak yazıldığı bilinen veya şüphelenilen tiyatro ve romanlarda "
        "(Örn: *William Shakespeare* ile *Christopher Marlowe* veya *John Fletcher* işbirlikleri) "
        "metin ilerledikçe üslup hakimiyetinin hangi yazara ait olduğunu adım adım izler."
    )

    all_authors_sorted = sorted(list(set(atlas_info["authors_list"])))
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        shak_idx = all_authors_sorted.index("William Shakespeare") if "William Shakespeare" in all_authors_sorted else 0
        roll_author_a = st.selectbox(
            "1. Yazar Adayı:",
            options=all_authors_sorted,
            index=shak_idx,
            key="roll_a_select",
        )
    with col_r2:
        marl_idx = all_authors_sorted.index("Christopher Marlowe") if "Christopher Marlowe" in all_authors_sorted else 1
        roll_author_b = st.selectbox(
            "2. Yazar Adayı:",
            options=all_authors_sorted,
            index=marl_idx,
            key="roll_b_select",
        )

    # Örnek çok sahneli dramatik metin
    default_collaborative_text = (
        "Was this the face that launch'd a thousand ships, and burnt the topless towers of Ilium? "
        "Sweet Helen, make me immortal with a kiss! Her lips suck forth my soul; see where it flies! "
        "Come, Helen, come, give me my soul again. Here will I dwell, for heaven is in these lips, "
        "and all is dross that is not Helena. I will be Paris, and for love of thee, "
        "instead of Troy, shall Wittenberg be sack'd; and I will combat with weak Menelaus, "
        "and wear thy colours on my plumed crest; yea, I will wound Achilles in the heel, "
        "and then return to Helen for a kiss. O, thou art fairer than the evening air "
        "clad in the beauty of a thousand stars; brighter art thou than flaming Jupiter "
        "when he appear'd to hapless Semele; more lovely than the monarch of the sky "
        "in wanton Arethusa's azur'd arms; and none but thou shalt be my paramour! "
        "To be, or not to be, that is the question: whether 'tis nobler in the mind to suffer "
        "the slings and arrows of outrageous fortune, or to take arms against a sea of troubles "
        "and by opposing end them. To die—to sleep, no more; and by a sleep to say we end "
        "the heart-ache and the thousand natural shocks that flesh is heir to: 'tis a consummation "
        "devoutly to be wish'd. To die, to sleep; to sleep, perchance to dream—ay, there's the rub: "
        "for in that sleep of death what dreams may come, when we have shuffled off this mortal coil, "
        "must give us pause. There's the respect that makes calamity of so long life."
    )

    roll_text_input = st.text_area(
        "İncelenecek Çok Bölümlü / Ortak Edebi Metin:",
        value=default_collaborative_text,
        height=140,
        key="roll_text_area",
    )

    col_w1, col_w2, col_w3 = st.columns([1, 1, 1])
    with col_w1:
        win_size = st.slider("Pencere Büyüklüğü (Kelime):", min_value=15, max_value=80, value=35, step=5)
    with col_w2:
        step_sz = st.slider("Kaydırma Adımı (Step Size):", min_value=10, max_value=40, value=20, step=5)
    with col_w3:
        st.write("")
        st.write("")
        run_roll_btn = st.button("📈 Rolling Stylometry Analizini Çalıştır", type="primary", use_container_width=True)

    if run_roll_btn or roll_text_input:
        with st.spinner("Kayan pencere ile üslup dalgalanması hesaplanıyor..."):
            forensics_roll = AdvancedForensicsEngine(clf, StylometricFeatureExtractor())
            segments = forensics_roll.rolling_stylometry(
                roll_text_input,
                author_a=roll_author_a,
                author_b=roll_author_b,
                window_words=win_size,
                step_words=step_sz,
            )

        if segments:
            fig_roll = plot_rolling_stylometry(
                segments,
                author_a=roll_author_a,
                author_b=roll_author_b,
                title=f"Ortak Yazarlık Üslup Dağılımı: {roll_author_a} vs. {roll_author_b}",
            )
            st.pyplot(fig_roll)
            plt.close(fig_roll)

            st.markdown("##### 📑 Bölüm Bazlı Üslup Dağılımı Tablosu")
            df_segs = pd.DataFrame(segments)
            df_segs = df_segs.rename(columns={
                "segment_id": "Segment #",
                "word_range": "Kelime Aralığı",
                "preview": "Metin Önizlemesi",
                roll_author_a: f"{roll_author_a} (%)",
                roll_author_b: f"{roll_author_b} (%)",
                "dominant_author": "Baskın Yazar",
            })
            st.dataframe(df_segs, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🌌 Alt Bölüm B: 2B Edebiyat Galaksisi (PCA Haritası)")
    st.markdown(
        "70+ boyutlu stilometrik öznitelik uzayı (TTR, sözcük boyutu, arkaik oranlar, ek frekansları, noktalama entropisi), "
        "**Temel Bileşen Analizi (PCA)** ile 2 boyuta (PC1 ve PC2) indirgenmiştir. "
        "Aşağıdaki harita, 37 kanonik yazarın üslup merkezlerini edebi dönem renkleriyle gösterir. "
        "Sekme 1'de analiz edilen metin varsa büyük kırmızı bir yıldızla ('★ Sizin Metniniz') uzaya yerleştirilir."
    )

    query_from_tab1 = st.session_state.get("last_query_text", "")
    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        if query_from_tab1:
            st.success(f"📍 **Canlı Test Metni Yüklendi:** \"{query_from_tab1[:85].strip()}...\" (Kırmızı yıldızla işaretleniyor)")
        else:
            st.info("💡 Sekme 1'de bir metin analiz ettiğinizde, o metnin koordinatları otomatik olarak bu galaksiye yansıtılacaktır.")
    with col_q2:
        recalc_pca_btn = st.button("🌌 Galaksiyi Güncelle / Yenile", use_container_width=True)

    with st.spinner("70+ boyutlu stilometri matrisi PCA ile 2B uzaya izdüşürülüyor..."):
        forensics_pca = AdvancedForensicsEngine(clf, StylometricFeatureExtractor())
        pca_data = forensics_pca.compute_stylometric_galaxy_pca(
            X_corpus=atlas_info["df_X"],
            y_authors=list(atlas_info["y_authors"]),
            y_eras=list(atlas_info["y_eras"]),
            query_text=query_from_tab1,
        )

    fig_galaxy = plot_stylometric_galaxy(
        pca_data,
        title="37 Kanonik Yazar ve Metninizin 2B Üslup Galaksisi (PCA Haritası)",
    )
    st.pyplot(fig_galaxy)
    plt.close(fig_galaxy)

    with st.expander("🔍 37 Yazarın PCA Merkez Koordinatları (Centroids)"):
        st.dataframe(pca_data["centroids"], use_container_width=True, hide_index=True)


# ------------------------------------------------------------------------------
# SEKME 6: EDEBİ SOY AĞACI & ETKİ AĞI
# ------------------------------------------------------------------------------
with tab_network:
    st.markdown("### 🌳 37 Kanonik Yazarın Edebi Soy Ağacı & Etki Ağı")
    st.markdown(
        "Hesaplamalı edebiyat ve filogenetik stilometri (**Phylogenetic Stylometry**), yazarlar arasındaki kronolojik ve "
        "tarihsel mesafelerin ötesine geçerek **sözdizimsel, sözcüksel ve morfolojik genetik akrabalıkları** ortaya koyar. "
        "Aşağıdaki soy ağacı (Dendrogram) ve üslup ikizleri analizi; 600 yıllık İngiliz ve Amerikan edebiyatındaki "
        "gizli edebi miras bağlarını ve yüzyıllar arası üslup akrabalıklarını açığa çıkarır."
    )

    # 1. Hiyerarşik Kümeleme ve Soy Ağacı
    st.markdown("---")
    st.markdown("#### 🌿 Alt Bölüm A: Hiyerarşik Edebi Soy Ağacı (Phylogenetic Dendrogram)")
    st.caption("Average Linkage ve Cosine mesafesi ile 70+ stilometrik boyut üzerinden kümelenmiş 37 kanonik yazar.")

    col_link1, col_link2 = st.columns([2, 1])
    with col_link1:
        linkage_method = st.selectbox(
            "Hiyerarşik Kümeleme Bağlantı Metodu (Linkage Method):",
            options=["average", "complete", "single"],
            format_func=lambda x: {
                "average": "Average Linkage (UPGMA - Önerilen Dengeli Kümeleme)",
                "complete": "Complete Linkage (En Uzak Mesafe Kümelemesi)",
                "single": "Single Linkage (En Yakın Komşu Kümelemesi)",
            }[x],
            index=0,
            key="linkage_select",
        )
    with col_link2:
        distance_metric_choice = st.selectbox(
            "Mesafe Metriği:",
            options=["cosine", "burrows_delta"],
            format_func=lambda x: "Cosine Mesafesi (Açısal Üslup Yakınlığı)" if x == "cosine" else "Burrows' Delta (Manhattan Z-Skor)",
            index=0,
            key="phylogeny_metric_select",
        )

    with st.spinner("37 yazarın çift yönlü mesafe matrisi ve edebi soy ağacı hesaplanıyor..."):
        phylogeny_engine = LiteraryPhylogenyEngine(
            X_corpus=atlas_info["df_X"],
            y_authors=list(atlas_info["y_authors"]),
            y_eras=list(atlas_info["y_eras"]),
            metric=distance_metric_choice,
        )
        linkage_mat = phylogeny_engine.compute_linkage_matrix(method=linkage_method)

    fig_dendro = plot_literary_dendrogram(
        linkage_matrix=linkage_mat,
        authors=phylogeny_engine.authors_,
        eras=[phylogeny_engine.eras_map_.get(a, "") for a in phylogeny_engine.authors_],
        title="37 Kanonik Yazarın Edebi Soy Ağacı (Phylogenetic Stylometry Dendrogram)",
    )
    st.pyplot(fig_dendro)
    plt.close(fig_dendro)

    # 2. Tarihsel Üslup İkizleri (Stylistic Soulmates)
    st.markdown("---")
    st.markdown("#### 🎭 Alt Bölüm B: Tarihsel Üslup İkizleri (Cross-Era Stylistic Soulmates)")
    st.markdown(
        "Her yazarın tüm edebiyat tarihindeki (hem kendi çağından hem de farklı yüzyıllardan) "
        "en yakın üslup akrabalarını keşfedin (Örn: *Jonathan Swift* ➔ *George Orwell*, *Shakespeare* ➔ *Marlowe*)."
    )

    col_target_auth, col_top_k = st.columns([3, 1])
    with col_target_auth:
        selected_author = st.selectbox(
            "Üslup İkizleri İncelenecek Yazar:",
            options=phylogeny_engine.authors_,
            index=phylogeny_engine.authors_.index("Jonathan Swift") if "Jonathan Swift" in phylogeny_engine.authors_ else 0,
            key="soulmate_auth_select",
        )
    with col_top_k:
        soulmates_k = st.selectbox("Gösterilecek İkiz Sayısı:", options=[3, 4, 5], index=0, key="soulmates_k_select")

    soulmates = phylogeny_engine.get_stylistic_soulmates(selected_author, top_k=soulmates_k)

    target_era = phylogeny_engine.eras_map_.get(selected_author, "Bilinmeyen")
    st.markdown(f"##### 🎯 **{selected_author}** ({target_era}) İçin Tespit Edilen En Yakın {len(soulmates)} Üslup İkizi:")

    cols = st.columns(len(soulmates))
    for idx, (col_s, sm) in enumerate(zip(cols, soulmates), 1):
        with col_s:
            badge_color = "#3b82f6" if "Dönemdaş" in sm["relationship_type"] else "#8b5cf6"
            st.markdown(
                f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); min-height: 260px;">
                    <div style="font-size: 0.8rem; font-weight: 700; color: #64748b; text-transform: uppercase;">İkiz #{idx}</div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a; margin-top: 4px;">{sm['author']}</div>
                    <div style="font-size: 0.85rem; color: #475569; margin-bottom: 8px;">🏛️ {sm['era']}</div>
                    <div style="background: {badge_color}15; color: {badge_color}; border: 1px solid {badge_color}40; padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; display: inline-block; margin-bottom: 10px;">
                        {sm['relationship_type']}
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.88rem;">
                        <span style="color: #64748b;">Akrabalık Skoru:</span>
                        <strong style="color: #16a34a;">%{sm['similarity_pct']}</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px; font-size: 0.88rem;">
                        <span style="color: #64748b;">Üslup Mesafesi:</span>
                        <code style="color: #0f172a;">{sm['distance']}</code>
                    </div>
                    <div style="font-size: 0.8rem; color: #334155; border-top: 1px dashed #cbd5e1; padding-top: 8px; line-height: 1.4;">
                        💡 <b>Üslup Köprüsü:</b> {sm['stylistic_insight']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 3. Yazarlar Arası Mesafe Isı Haritası
    st.markdown("---")
    st.markdown("#### 🗺️ Alt Bölüm C: Üslup Akrabalık ve Çift Yönlü Mesafe Isı Haritası (Heatmap)")

    col_filter_era, col_heat_info = st.columns([2, 1])
    with col_filter_era:
        filter_heat_era = st.selectbox(
            "Döneme Göre Odaklan (Opsiyonel):",
            options=["Tüm 37 Yazarı Göster (Global Matris)"] + LITERARY_ERAS,
            index=0,
            key="heat_era_select",
        )
    with col_heat_info:
        st.write("")
        st.caption("Koyu/canlı renkler birbirine üslupça en yakın yazarları temsil eder.")

    if filter_heat_era == "Tüm 37 Yazarı Göster (Global Matris)":
        heat_authors = phylogeny_engine.authors_
    else:
        heat_authors = [a for a in phylogeny_engine.authors_ if phylogeny_engine.eras_map_.get(a) == filter_heat_era]

    if heat_authors:
        fig_heat = plot_author_distance_heatmap(
            distance_matrix=phylogeny_engine.distance_df_,
            authors=heat_authors,
            title=f"Yazarlar Arası Üslup Akrabalık Matrisi ({filter_heat_era})",
        )
        st.pyplot(fig_heat)
        plt.close(fig_heat)
    else:
        st.info("Bu filtreye uygun yazar bulunamadı.")


# ==============================================================================
# Sayfa Altbilgisi (Footer)
# ==============================================================================
st.markdown(
    """
    <div class="dev-footer">
        🏛️ Developed by Bayram Akpamukçu | TÜBİTAK Computational Literary Studies & Stylometry Project
    </div>
    """,
    unsafe_allow_html=True,
)

