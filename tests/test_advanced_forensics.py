"""
İleri Düzey Adli Stilometri Birim Testleri (test_advanced_forensics.py)
====================================================================
Proje: TÜBİTAK - Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası
Developed by Bayram Akpamukçu
"""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.advanced_forensics import AdvancedForensicsEngine
from src.classifier import StylometryClassifier
from src.feature_extractor import StylometricFeatureExtractor
from src.visualizer import plot_rolling_stylometry, plot_stylometric_galaxy


class TestAdvancedForensics(unittest.TestCase):
    """Adli stilometri, AI tespiti, PCA galaksisi ve raporlayıcı testleri."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.extractor = StylometricFeatureExtractor()

        # Basit test külliyatı oluştur
        texts_a = [
            "Whan that Aprille with his shoures soote hath perced the droghte of March to the roote.",
            "And bathed every veyne in swich licour of which vertu engendred is the flour; eek soote.",
            "Thanne longen folk to goon on pilgrimages, and palmeres for to seken straunge strondes.",
        ]
        texts_b = [
            "Was this the face that launch'd a thousand ships, and burnt the topless towers of Ilium?",
            "Sweet Helen, make me immortal with a kiss! Her lips suck forth my soul; see where it flies!",
            "Come, Helen, come, give me my soul again. Here will I dwell, for heaven is in these lips.",
        ]

        data = []
        authors = []
        eras = []
        for t in texts_a:
            data.append(cls.extractor.extract_features(t))
            authors.append("Geoffrey Chaucer")
            eras.append("Middle English")
        for t in texts_b:
            data.append(cls.extractor.extract_features(t))
            authors.append("Christopher Marlowe")
            eras.append("Renaissance")

        cls.df_X = pd.DataFrame(data)
        cls.authors = authors
        cls.eras = eras

        cls.classifier = StylometryClassifier(model_type="svm", cv_folds=2, random_state=42, feature_extractor=cls.extractor)
        cls.classifier.train_hierarchical(cls.df_X, cls.authors, cls.eras)
        cls.forensics = AdvancedForensicsEngine(cls.classifier, cls.extractor)

    def test_rolling_stylometry_segments_structure(self) -> None:
        """Kayan pencere ortak yazarlık analizi çıktı şemasını ve toplam skorları test et."""
        test_text = (
            "Was this the face that launch'd a thousand ships, and burnt the topless towers of Ilium? "
            "Sweet Helen, make me immortal with a kiss! Her lips suck forth my soul; see where it flies! "
            "Whan that Aprille with his shoures soote the droghte of March hath perced to the roote. "
            "And bathed every veyne in swich licour of which vertu engendred is the flour."
        )
        segments = self.forensics.rolling_stylometry(
            text=test_text,
            author_a="Christopher Marlowe",
            author_b="Geoffrey Chaucer",
            window_words=20,
            step_words=10,
        )

        self.assertGreater(len(segments), 0)
        for seg in segments:
            self.assertIn("segment_id", seg)
            self.assertIn("word_range", seg)
            self.assertIn("preview", seg)
            self.assertIn("Christopher Marlowe", seg)
            self.assertIn("Geoffrey Chaucer", seg)
            self.assertIn("dominant_author", seg)
            # Olasılık toplamı ~100 olmalı
            total_score = seg["Christopher Marlowe"] + seg["Geoffrey Chaucer"]
            self.assertAlmostEqual(total_score, 100.0, delta=0.5)

    def test_detect_ai_imitation(self) -> None:
        """AI vs Human tespit motorunun Burstiness ve Hapax oranlarını doğru hesapladığını test et."""
        # Yüksek stilistik varyasyona sahip edebi metin
        human_text = (
            "To be, or not to be, that is the question: whether 'tis nobler in the mind to suffer "
            "the slings and arrows of outrageous fortune, or to take arms against a sea of troubles "
            "and by opposing end them. To die—to sleep, no more; and by a sleep to say we end "
            "the heart-ache and the thousand natural shocks that flesh is heir to! 'Tis done."
        )
        res = self.forensics.detect_ai_imitation(human_text)

        self.assertIn("human_authenticity_pct", res)
        self.assertIn("ai_probability_pct", res)
        self.assertIn("burstiness_index", res)
        self.assertIn("sentence_variation_cv", res)
        self.assertIn("hapax_ratio", res)
        self.assertIn("verdict", res)
        self.assertAlmostEqual(res["human_authenticity_pct"] + res["ai_probability_pct"], 100.0, places=1)
        self.assertGreater(res["human_authenticity_pct"], 50.0)

        # Tekdüze sentetik metin testi (düşük varyans)
        synthetic_text = "This is a short book. That is a small tree. Here is a blue bird. There is a cold lake."
        res_syn = self.forensics.detect_ai_imitation(synthetic_text)
        self.assertLess(res_syn["human_authenticity_pct"], res["human_authenticity_pct"])

    def test_compute_stylometric_galaxy_pca(self) -> None:
        """70+ boyutlu öznitelik uzayının PCA ile 2B koordinatlara dönüştüğünü doğrula."""
        query = "Sweet Helen, make me immortal with a kiss!"
        pca_data = self.forensics.compute_stylometric_galaxy_pca(
            X_corpus=self.df_X,
            y_authors=self.authors,
            y_eras=self.eras,
            query_text=query,
        )

        self.assertIn("centroids", pca_data)
        self.assertIn("query_point", pca_data)
        self.assertIn("explained_variance", pca_data)

        centroids = pca_data["centroids"]
        self.assertIsInstance(centroids, pd.DataFrame)
        self.assertIn("PC1", centroids.columns)
        self.assertIn("PC2", centroids.columns)
        self.assertIn("Author", centroids.columns)
        self.assertIn("Era", centroids.columns)
        self.assertEqual(len(centroids), 2)  # Chaucer ve Marlowe

        q_point = pca_data["query_point"]
        self.assertIsNotNone(q_point)
        self.assertIn("PC1", q_point)
        self.assertIn("PC2", q_point)

    def test_generate_forensic_report_markdown(self) -> None:
        """Bayram Akpamukçu imzalı adli stilometri raporunun üretildiğini doğrula."""
        top_authors = [("Christopher Marlowe", 0.92), ("William Shakespeare", 0.05)]
        ai_data = {
            "human_authenticity_pct": 88.5,
            "ai_probability_pct": 11.5,
            "sentence_variation_cv": 0.45,
            "hapax_ratio": 0.62,
            "burstiness_index": 0.12,
            "verdict": "Gerçek İnsan Edebi Üslubu (Organic Literary Genius)",
        }
        report = self.forensics.generate_forensic_report_markdown(
            text="Was this the face that launch'd a thousand ships, and burnt the topless towers of Ilium?",
            predicted_era="Renaissance",
            top_authors=top_authors,
            ai_analysis=ai_data,
        )

        self.assertIn("TÜBİTAK ADLİ STİLOMETRİ VE YAZAR TESPİT RAPORU", report)
        self.assertIn("Bayram Akpamukçu", report)
        self.assertIn("Christopher Marlowe", report)
        self.assertIn("Renaissance", report)
        self.assertIn("Organic Literary Genius", report)

    def test_forensic_visualizations(self) -> None:
        """Rolling Stylometry ve Galaxy grafiklerinin hatasız Figure döndürdüğünü test et."""
        segments = [
            {"segment_id": 1, "word_range": "1-20", "Author A": 75.0, "Author B": 25.0},
            {"segment_id": 2, "word_range": "11-30", "Author A": 40.0, "Author B": 60.0},
        ]
        fig_roll = plot_rolling_stylometry(segments, "Author A", "Author B")
        self.assertIsInstance(fig_roll, Figure)

        pca_mock = {
            "centroids": pd.DataFrame({
                "PC1": [1.2, -0.8],
                "PC2": [0.5, -0.3],
                "Author": ["Author A", "Author B"],
                "Era": ["Renaissance", "Victorian"],
            }),
            "query_point": {"PC1": 1.0, "PC2": 0.4},
            "explained_variance": [45.2, 22.8],
        }
        fig_galaxy = plot_stylometric_galaxy(pca_mock)
        self.assertIsInstance(fig_galaxy, Figure)


if __name__ == "__main__":
    unittest.main()
