"""
TÜBİTAK Evrensel İngiliz Edebiyatı Stilometri Atlası
Modül: İleri Düzey Adli Stilometri (Rolling Stylometry, AI Detector & 2D PCA Galaxy)
Developed by Bayram Akpamukçu
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class AdvancedForensicsEngine:
    """
    Ortak yazarlık taraması (Rolling Stylometry), Yapay Zeka (LLM) taklit tespiti
    ve 2B PCA Üslup Uzayı izdüşümü gerçekleştiren ileri düzey analiz motoru.
    
    Developed by Bayram Akpamukçu
    """

    def __init__(self, classifier: Any, extractor: Any):
        self.classifier = classifier
        self.extractor = extractor

    def rolling_stylometry(
        self,
        text: str,
        author_a: str,
        author_b: str,
        window_words: int = 45,
        step_words: int = 25,
    ) -> List[Dict[str, Any]]:
        """
        Uzun bir metni kayan pencerelerle (sliding window) tarayarak her bölümde
        hangi yazarın üslubunun baskın olduğunu (Örn: Shakespeare vs. Marlowe) hesaplar.
        
        Args:
            text (str): Analiz edilecek çok sahneli veya uzun metin.
            author_a (str): 1. Yazar adayı (Örn: "William Shakespeare").
            author_b (str): 2. Yazar adayı (Örn: "Christopher Marlowe").
            window_words (int): Kayan pencere büyüklüğü (kelime sayısı).
            step_words (int): Pencere kaydırma adımı (kelime sayısı).
            
        Returns:
            List[Dict[str, Any]]: Her segment için baskınlık yüzdeleri ve önizleme.
        """
        words = text.split()
        if not words:
            return []

        if len(words) < window_words:
            window_words = max(10, len(words))
            step_words = window_words

        segments: List[Dict[str, Any]] = []
        segment_idx = 1

        total_authors_count = (
            len(getattr(self.classifier, "author_profiles_", {}))
            if hasattr(self.classifier, "author_profiles_") and self.classifier.author_profiles_
            else 50
        )

        for start in range(0, max(1, len(words) - window_words + 1), step_words):
            chunk_words = words[start : start + window_words]
            if len(chunk_words) < 8:
                continue
            chunk_text = " ".join(chunk_words)

            # Burrows' Delta z-skor mesafesini hesapla
            delta_res = self.classifier.compute_burrows_delta(
                chunk_text, top_k=max(50, total_authors_count)
            )

            # Liste veya dict formatını sözlüğe çevir
            if isinstance(delta_res, list):
                delta_map = {item[0]: float(item[1]) for item in delta_res}
            elif isinstance(delta_res, dict):
                delta_map = {k: float(v) for k, v in delta_res.items()}
            else:
                delta_map = {}

            delta_a = delta_map.get(author_a, 1.5)
            delta_b = delta_map.get(author_b, 1.5)

            # Sayısal taşmayı ve sıfıra yuvarlanmayı önleyen kararlı olasılık dönüşümü
            min_d = min(delta_a, delta_b)
            # İki yazar arasındaki farkı ölçeklendirerek kararlı softmax hesapla
            diff_scale = max(1.0, (delta_a + delta_b) / 10.0) if (delta_a + delta_b) > 20.0 else 1.0
            sim_a = float(np.exp(-(delta_a - min_d) / diff_scale))
            sim_b = float(np.exp(-(delta_b - min_d) / diff_scale))
            total = sim_a + sim_b if (sim_a + sim_b) > 0 else 1.0

            score_a = round((sim_a / total) * 100, 1)
            score_b = round(100.0 - score_a, 1)

            segments.append({
                "segment_id": segment_idx,
                "word_range": f"{start + 1}-{start + len(chunk_words)}",
                "preview": chunk_text[:65] + ("..." if len(chunk_text) > 65 else ""),
                author_a: score_a,
                author_b: score_b,
                "dominant_author": author_a if score_a >= score_b else author_b,
            })
            segment_idx += 1

        return segments

    def detect_ai_imitation(self, text: str) -> Dict[str, Any]:
        """
        Metnin gerçek bir edebi yazar mı yoksa LLM (ChatGPT vb.) tarafından üretilmiş
        mekanik bir taklit (pastiche) mi olduğunu 'Burstiness' ve leksikal entropi ile ölçer.
        
        Args:
            text (str): İncelenecek edebi veya sentetik metin.
            
        Returns:
            Dict[str, Any]: human_authenticity_pct, ai_probability_pct, burstiness_index,
                           sentence_variation_cv, hapax_ratio, verdict.
        """
        features = self.extractor.extract_features(text)
        sentences = self.extractor.preprocessor.tokenize_sentences(text)
        sent_lengths = [
            len(self.extractor.preprocessor.tokenize_words(s))
            for s in sentences
            if s.strip()
        ]

        if len(sent_lengths) >= 2:
            mu = float(np.mean(sent_lengths))
            sigma = float(np.std(sent_lengths))
            # Burstiness Formülü: B = (sigma - mu) / (sigma + mu) -> İnsan yazımı yüksek varyans gösterir
            burstiness = (sigma - mu) / (sigma + mu) if (sigma + mu) > 0 else 0.0
            cv_sent = sigma / mu if mu > 0 else 0.0
        else:
            burstiness = -0.3
            cv_sent = 0.20

        hapax_ratio = float(features.get("hapax_legomena_ratio", 0.5))
        punct_variety = sum(
            1 for k, v in features.items() if k.startswith("punct_") and v > 0
        )

        # İnsan edebi dehası: Yüksek cümle boyutu dalgalanması (cv_sent), zengin hapax ve noktalama çeşitliliği
        human_score = (
            min(1.0, cv_sent / 0.65) * 0.45
            + min(1.0, hapax_ratio / 0.75) * 0.35
            + min(1.0, punct_variety / 4.0) * 0.20
        )
        human_authenticity_pct = round(float(np.clip(human_score * 100, 12.0, 98.5)), 1)
        ai_probability_pct = round(100.0 - human_authenticity_pct, 1)

        verdict = (
            "Gerçek İnsan Edebi Üslubu (Organic Literary Genius)"
            if human_authenticity_pct >= 55.0
            else "Yapay Zeka Taklidi Şüphesi (Synthetic LLM Pastiche)"
        )

        return {
            "human_authenticity_pct": human_authenticity_pct,
            "ai_probability_pct": ai_probability_pct,
            "burstiness_index": round(burstiness, 3),
            "sentence_variation_cv": round(cv_sent, 3),
            "hapax_ratio": round(hapax_ratio, 3),
            "verdict": verdict,
        }

    def compute_stylometric_galaxy_pca(
        self,
        X_corpus: pd.DataFrame,
        y_authors: List[str],
        y_eras: List[str],
        query_text: str = "",
    ) -> Dict[str, Any]:
        """
        70+ boyutlu stilometri matrisini PCA ile 2 boyuta (PC1, PC2) indirgeyerek
        37 yazarın üslup haritasını ve aranan metnin koordinatını çıkarır.
        
        Args:
            X_corpus (pd.DataFrame): Külliyatın stilometrik öznitelik matrisi.
            y_authors (List[str]): Her satırın yazar etiketi.
            y_eras (List[str]): Her satırın edebi dönem etiketi.
            query_text (str): Koordinatı hesaplanacak sorgu metni (opsiyonel).
            
        Returns:
            Dict[str, Any]: centroids DataFrame'i, query_point koordinatı ve explained_variance.
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_corpus.values)

        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)

        df_map = pd.DataFrame({
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "Author": list(y_authors),
            "Era": list(y_eras),
        })
        # Yazar merkezlerini (centroids) al
        author_centroids = df_map.groupby(["Author", "Era"], as_index=False)[["PC1", "PC2"]].mean()

        query_point: Optional[Dict[str, float]] = None
        if query_text and query_text.strip():
            q_features = self.extractor.extract_features(query_text)
            q_df = pd.DataFrame([q_features])[X_corpus.columns]
            q_scaled = scaler.transform(q_df.values)
            q_coord = pca.transform(q_scaled)[0]
            query_point = {
                "PC1": round(float(q_coord[0]), 3),
                "PC2": round(float(q_coord[1]), 3),
            }

        return {
            "centroids": author_centroids,
            "query_point": query_point,
            "explained_variance": [
                round(float(v * 100), 1) for v in pca.explained_variance_ratio_
            ],
        }

    def generate_forensic_report_markdown(
        self,
        text: str,
        predicted_era: str,
        top_authors: List[Any],
        ai_analysis: Dict[str, Any],
    ) -> str:
        """
        Analiz sonuçlarını resmi bir 'TÜBİTAK Adli Stilometri Raporu' olarak üretir.
        
        Args:
            text (str): Analiz edilen metin.
            predicted_era (str): Katman 1'de tahmin edilen edebi dönem.
            top_authors (List[Any]): En olası aday yazarlar (tuple veya dict listesi).
            ai_analysis (Dict[str, Any]): detect_ai_imitation çıktısı.
            
        Returns:
            str: Markdown formatında adli analiz raporu.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Top authors ayrıştırma
        parsed_top: List[Dict[str, Any]] = []
        for item in top_authors:
            if isinstance(item, tuple):
                parsed_top.append({
                    "author": str(item[0]),
                    "probability": float(item[1]),
                    "delta": "-",
                })
            elif isinstance(item, dict):
                parsed_top.append({
                    "author": str(item.get("author", "Unknown")),
                    "probability": float(item.get("probability", 0.0)),
                    "delta": item.get("delta", "-"),
                })
            else:
                parsed_top.append({"author": str(item), "probability": 0.0, "delta": "-"})

        top1 = parsed_top[0] if parsed_top else {"author": "Unknown", "probability": 0.0, "delta": "-"}
        clean_text_preview = text[:280].strip().replace("\n", " ")

        report = f"""# 📜 TÜBİTAK ADLİ STİLOMETRİ VE YAZAR TESPİT RAPORU
**Sistem:** Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası  
**Geliştirici:** Bayram Akpamukçu  
**Rapor No:** TÜBİTAK-CSL-{datetime.now().strftime("%Y%m%d%H%M%S")}  
**Tarih / Saat:** {timestamp}  

---

## 1. İNCELENEN METİN ÖZETİ
> "{clean_text_preview}..."

- **Toplam Karakter Uzunluğu:** {len(text)} karakter
- **Yaklaşık Kelime Sayısı:** {len(text.split())} kelime

---

## 2. ÇİFT KATMANLI HİYERARŞİK TESPİT SONUCU
- **Tahmin Edilen Edebi Dönem (Katman 1):** `{predicted_era}`
- **Birincil Yazar Eşleşmesi (Katman 2):** **{top1['author']}** (Model Güveni: **%{round(top1['probability'] * 100, 1)}**)

### En Yakın 5 Üslup Adayı (Kanonik Sıralama):
"""
        for idx, item in enumerate(parsed_top[:5], 1):
            delta_str = f" | Burrows' Delta: `{item['delta']}`" if item.get("delta") != "-" else ""
            report += f"{idx}. **{item['author']}** — Olasılık: `%{round(item['probability'] * 100, 1)}`{delta_str}\n"

        report += f"""
---

## 3. YAPAY ZEKA (LLM) vs. GERÇEK YAZAR DOĞRULAMASI
- **Adli Karar:** **{ai_analysis.get('verdict', 'N/A')}**
- **Organik İnsan Üslubu Skoru:** `%{ai_analysis.get('human_authenticity_pct', 0.0)}`
- **Yapay Zeka (LLM) Taklit Riski:** `%{ai_analysis.get('ai_probability_pct', 0.0)}`
- **Cümle Ritim Dalgalanması (Burstiness CV):** `{ai_analysis.get('sentence_variation_cv', 0.0)}`
- **Hapax Legomena (Tekil Kelime Zenginliği):** `{ai_analysis.get('hapax_ratio', 0.0)}`
- **Burstiness İndeksi:** `{ai_analysis.get('burstiness_index', 0.0)}`

---

## 4. ADLİ STİLOMETRİ VE METODOLOJİ BİLGİSİ
Bu rapor; John Burrows'un klasik Delta (Z-Score Manhattan Distance) algoritması,
Support Vector Classifier (Linear Kernel), 70+ sözdizimsel-morfolojik öznitelik çıkarımı
ve leksikal entropi (Burstiness) analizleri birleştirilerek oluşturulmuştur.

---
*Bu rapor **Developed by Bayram Akpamukçu | TÜBİTAK Computational Literary Studies Engine** tarafından dijital olarak sertifikalandırılmıştır.*
"""
        return report
