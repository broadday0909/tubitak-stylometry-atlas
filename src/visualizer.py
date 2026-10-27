"""
Stilometri Görselleştirme Modülü (Visualizer)
==============================================
Proje: TÜBİTAK - Stilometri ile Edebi Eserlerde Yazar Tespiti ve Üslup Analizi

Bu modül, makine öğrenmesi modellerinin performansını, karmaşıklık matrisini (confusion matrix)
ve ayırt edici stilometrik öznitelik katsayılarını (feature importances / SVM coefficients)
akademik ve estetik grafikler halinde görselleştiren Matplotlib/Seaborn fonksiyonlarını içerir.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix

# Akademik ve modern görsel tema ayarları
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def plot_confusion_matrix(
    y_true: Union[Sequence[str], np.ndarray],
    y_pred: Union[Sequence[str], np.ndarray],
    classes: Sequence[str],
    title: str = "Karmaşıklık Matrisi (Confusion Matrix)",
    cmap: str = "Blues",
) -> Figure:
    """
    Modelin sınıflandırma doğruluğunu ve hatalarını Seaborn ısı haritası (heatmap)
    ile görselleştirir.
    
    Args:
        y_true (Sequence[str] | np.ndarray): Gerçek yazar etiketleri.
        y_pred (Sequence[str] | np.ndarray): Model tarafından tahmin edilen etiketler.
        classes (Sequence[str]): Sınıf isimleri listesi.
        title (str): Grafik başlığı.
        cmap (str): Isı haritası renk paleti (Varsayılan: 'Blues').
        
    Returns:
        Figure: Streamlit veya dosya kaydı için Matplotlib Figure nesnesi.
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Yazar isimlerini etiketler için kısalt/düzenle (uzun isimler için)
    display_labels = [c.replace(" ", "\n") if len(c) > 15 else c for c in classes]

    fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=120)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        cbar=True,
        square=True,
        xticklabels=display_labels,
        yticklabels=display_labels,
        linewidths=1.2,
        linecolor="#ffffff",
        annot_kws={"size": 13, "weight": "bold"},
        ax=ax,
    )

    ax.set_title(title, fontsize=13, weight="bold", pad=15, color="#1e293b")
    ax.set_xlabel("Tahmin Edilen Yazar", fontsize=11, weight="semibold", labelpad=10, color="#334155")
    ax.set_ylabel("Gerçek Yazar", fontsize=11, weight="semibold", labelpad=10, color="#334155")
    ax.tick_params(axis="both", which="major", labelsize=9.5)
    
    fig.tight_layout()
    return fig


def plot_feature_importance(
    coefs: Union[pd.DataFrame, pd.Series, np.ndarray, Sequence[float]],
    feature_names: Optional[Sequence[str]] = None,
    top_n: int = 12,
    title: str = "Yazarları En Çok Ayırt Eden Stilometrik Öznitelikler",
    color: str = "#2563eb",
) -> Figure:
    """
    Linear SVM veya Random Forest modelinin en ayırt edici bulduğu öznitelikleri
    yatay çubuk grafik (horizontal bar chart) ile görselleştirir.
    
    Args:
        coefs: Öznitelik katsayıları, önem skorları veya önceden hazırlanmış
            (feature, importance) DataFrame'i.
        feature_names (Optional[Sequence[str]]): Öznitelik isimleri listesi.
        top_n (int): Gösterilecek en etkili öznitelik sayısı (Varsayılan: 12).
        title (str): Grafik başlığı.
        color (str): Çubukların ana rengi.
        
    Returns:
        Figure: Matplotlib Figure nesnesi.
    """
    if isinstance(coefs, pd.DataFrame):
        # Eğer DataFrame geldiyse (örn: clf.get_feature_importances() çıktısı)
        col_names = coefs.columns
        feat_col = col_names[0]
        val_col = col_names[1]
        df_top = coefs.head(top_n).copy()
        names = df_top[feat_col].tolist()
        values = df_top[val_col].tolist()
    elif isinstance(coefs, pd.Series):
        s_sorted = coefs.abs().sort_values(ascending=False).head(top_n)
        names = s_sorted.index.tolist()
        values = s_sorted.values.tolist()
    else:
        values_arr = np.array(coefs)
        if values_arr.ndim > 1:
            values_arr = np.mean(np.abs(values_arr), axis=0)
        else:
            values_arr = np.abs(values_arr)

        names_list = list(feature_names) if feature_names is not None else [f"feat_{i}" for i in range(len(values_arr))]
        sorted_idx = np.argsort(values_arr)[::-1][:top_n]
        names = [names_list[i] for i in sorted_idx]
        values = [values_arr[i] for i in sorted_idx]

    # Okunabilirlik için grafik hazırlığı (ters sıralama: en büyük yukarıda)
    names = names[::-1]
    values = values[::-1]

    # Öznitelik isimlerini Türkçe açıklamalarla zenginleştir
    pretty_names = [_prettify_feature_name(n) for n in names]

    fig, ax = plt.subplots(figsize=(8.0, max(4.5, top_n * 0.4)), dpi=120)
    y_pos = np.arange(len(pretty_names))

    # Gradyan/estetik renk paleti
    colors = sns.color_palette("Blues_r", n_colors=len(values))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="#1e3a8a", height=0.65, alpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pretty_names, fontsize=9.5, color="#1e293b")
    ax.set_xlabel("Mutlak Önem Skoru / Katsayı Büyüklüğü (|katsayı|)", fontsize=10.5, weight="semibold", color="#334155")
    ax.set_title(title, fontsize=12.5, weight="bold", pad=15, color="#0f172a")

    # Çubukların ucuna değerleri yazdır
    max_val = max(values) if values else 1.0
    for bar, val in zip(bars, values):
        ax.text(
            val + (max_val * 0.015),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=8.5,
            weight="bold",
            color="#1e293b",
        )

    ax.set_xlim(0, max_val * 1.16)
    sns.despine(top=True, right=True)
    fig.tight_layout()
    return fig


def plot_probability_bars(probabilities: dict[str, float]) -> Figure:
    """
    Tahmin edilen sınıf olasılıklarını renkli yatay çubuk grafik olarak çizer.
    
    Args:
        probabilities (dict[str, float]): {yazar: olasilik} sözlüğü.
        
    Returns:
        Figure: Matplotlib Figure nesnesi.
    """
    sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    authors = [item[0].replace(" ", "\n") if len(item[0]) > 18 else item[0] for item in sorted_items][::-1]
    probs = [item[1] * 100 for item in sorted_items][::-1]

    fig, ax = plt.subplots(figsize=(6.0, 3.2), dpi=120)
    palette = sns.color_palette("mako", n_colors=len(probs))
    bars = ax.barh(authors, probs, color=palette, edgecolor="#0f172a", height=0.55)

    ax.set_xlim(0, 105)
    ax.set_xlabel("Model Güven / Olasılık Oranı (%)", fontsize=9.5, weight="semibold", color="#334155")
    ax.set_title("Yazar Tahmini Olasılık Dağılımı", fontsize=11, weight="bold", pad=10, color="#0f172a")

    for bar, p in zip(bars, probs):
        ax.text(
            p + 2.0,
            bar.get_y() + bar.get_height() / 2,
            f"%{p:.1f}",
            va="center",
            ha="left",
            fontsize=9.0,
            weight="bold",
            color="#0f172a",
        )

    sns.despine(top=True, right=True)
    fig.tight_layout()
    return fig


def plot_author_stylistic_comparison(
    author1_name: str,
    features1: Dict[str, float],
    author2_name: str,
    features2: Dict[str, float],
    metrics_to_compare: Optional[List[Tuple[str, str]]] = None,
    title: str = "Kanonik Yazarlar Üslup Karşılaştırması",
) -> Figure:
    """
    İki farklı yazarın temel stilometrik ve edebi parmak izlerini
    yan yana gruplanmış çubuk grafik (grouped bar chart) ile karşılaştırır.
    
    Args:
        author1_name (str): 1. Yazarın adı (Örn: Shakespeare).
        features1 (Dict[str, float]): 1. Yazarın ortalama öznitelikleri.
        author2_name (str): 2. Yazarın adı (Örn: Marlowe).
        features2 (Dict[str, float]): 2. Yazarın ortalama öznitelikleri.
        metrics_to_compare: (öznitelik_anahtarı, etiket) demetleri listesi.
        title (str): Grafik başlığı.
        
    Returns:
        Figure: Matplotlib Figure nesnesi.
    """
    if metrics_to_compare is None:
        metrics_to_compare = [
            ("ttr", "Sözcük Çeşitliliği (TTR)"),
            ("archaic_total_ratio", "Arkaik Sözcük Oranı"),
            ("suffix_eth_ratio", "-eth Arkaik Fiil"),
            ("suffix_ly_ratio", "-ly Zarf Yoğunluğu"),
            ("suffix_ing_ratio", "-ing Süreklilik"),
            ("suffix_tion_sion_ratio", "-tion Soyut İsim"),
            ("stream_of_consciousness_index", "Bilinç Akışı İndeksi"),
            ("punct_semicolon_ratio", "Noktalı Virgül (;)"),
            ("dialogue_quote_ratio", "Diyalog (Tırnak)"),
        ]

    labels = [m[1] for m in metrics_to_compare]
    vals1 = [features1.get(m[0], 0.0) for m in metrics_to_compare]
    vals2 = [features2.get(m[0], 0.0) for m in metrics_to_compare]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=120)
    ax.bar(x - width / 2, vals1, width, label=author1_name, color="#2563eb", edgecolor="#1e3a8a", alpha=0.9)
    ax.bar(x + width / 2, vals2, width, label=author2_name, color="#e11d48", edgecolor="#9f1239", alpha=0.9)

    ax.set_ylabel("Öznitelik Değeri / Oranı", fontsize=10.5, weight="semibold", color="#334155")
    ax.set_title(title, fontsize=12.5, weight="bold", pad=15, color="#0f172a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9.5, color="#1e293b")
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=10)

    sns.despine(top=True, right=True)
    fig.tight_layout()
    return fig


def _prettify_feature_name(raw_name: str) -> str:
    """Teknik öznitelik isimlerini anlaşılır Türkçe etiketlere çevirir."""
    mapping = {
        "punct_semicolon_ratio": "Noktalı Virgül (;) Sıklığı",
        "punct_comma_ratio": "Virgül (,) Sıklığı",
        "punct_question_ratio": "Soru İşareti (?) Sıklığı",
        "punct_exclamation_ratio": "Ünlem İşareti (!) Sıklığı",
        "punct_total_ratio": "Toplam Noktalama Oranı",
        "avg_sentence_length_words": "Ort. Cümle Uzunluğu (Kelime)",
        "avg_word_length_chars": "Ort. Kelime Uzunluğu (Karakter)",
        "ttr": "Sözcük Çeşitliliği (TTR)",
        "root_ttr": "Kök TTR (Guiraud R İndeksi)",
        "hapax_legomena_ratio": "Hapax Oranı (Tekil Kelimeler)",
        "long_word_ratio": "Uzun Kelime Oranı (≥7 harf)",
        "short_word_ratio": "Kısa Kelime Oranı (≤3 harf)",
        "fw_total_ratio": "Toplam İşlevsel Kelime Oranı",
        "archaic_total_ratio": "Toplam Arkaik Sözcük Oranı",
        "suffix_eth_ratio": "-eth Arkaik Fiil Çekimi",
        "suffix_est_ratio": "-est Arkaik Fiil Çekimi",
        "suffix_ly_ratio": "-ly Zarf Yoğunluğu",
        "suffix_ing_ratio": "-ing Süreklilik Oranı",
        "suffix_tion_sion_ratio": "-tion/-sion Soyut İsimler",
        "suffix_total_ratio": "Toplam Morfolojik Ek Yoğunluğu",
        "stream_of_consciousness_index": "Bilinç Akışı İndeksi (Modernizm)",
        "poetic_line_break_ratio": "Dize / Satır Sonu Sıklığı",
        "poetic_line_capitalization_ratio": "Dize Başı Büyük Harf Oranı",
        "dialogue_quote_ratio": "Diyalog (Tırnak) Yoğunluğu",
        "sentence_length_variance_cv": "Cümle Boyutu Varyasyonu (CV)",
    }
    if raw_name in mapping:
        return mapping[raw_name]
    if raw_name.startswith("archaic_") and raw_name.endswith("_ratio"):
        word = raw_name[8:-6]
        return f"Arkaik: '{word}'"
    if raw_name.startswith("fw_") and raw_name.endswith("_ratio"):
        word = raw_name[3:-6]
        return f"İşlevsel: '{word}'"
    if raw_name.startswith("punct_") and raw_name.endswith("_ratio"):
        p = raw_name[6:-6]
        return f"Noktalama: {p}"
    return raw_name
