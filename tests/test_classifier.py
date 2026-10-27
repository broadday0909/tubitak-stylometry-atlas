"""
Çift Katmanlı Sınıflandırıcı ve Burrows' Delta Testleri (test_classifier.py)
=============================================================================
Proje: TÜBİTAK - Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası
"""

from __future__ import annotations

import unittest
import warnings
import numpy as np
import pandas as pd

from src.classifier import StylometryClassifier
from src.feature_extractor import StylometricFeatureExtractor

warnings.filterwarnings("ignore", category=FutureWarning)


class TestStylometryClassifier(unittest.TestCase):
    """StylometryClassifier çift katmanlı sınıflandırma ve Burrows' Delta testleri."""

    def setUp(self) -> None:
        """Her testten önce küçük çok sınıflı ve çok dönemli bir sentetik veri kümesi oluşturur."""
        self.rng = np.random.RandomState(42)
        self.feature_names = [f"feat_{i}" for i in range(10)]

        # 3 Yazar, 2 Dönem
        # Yazar 1 & 2 -> Renaissance; Yazar 3 -> Modernism
        samples_1 = self.rng.normal(loc=2.0, scale=0.4, size=(6, 10))
        samples_2 = self.rng.normal(loc=1.2, scale=0.4, size=(6, 10))
        samples_3 = self.rng.normal(loc=-1.5, scale=0.4, size=(6, 10))

        X_mat = np.vstack([samples_1, samples_2, samples_3])
        self.df_X = pd.DataFrame(X_mat, columns=self.feature_names)
        self.s_y_author = pd.Series(["Shakespeare"] * 6 + ["Marlowe"] * 6 + ["Woolf"] * 6, name="author")
        self.s_y_era = pd.Series(["Renaissance"] * 12 + ["Modernism"] * 6, name="era")

    def test_train_hierarchical(self) -> None:
        """Çift katmanlı (Dönem ve Yazar) eğitimin hatasız tamamlanması testi."""
        clf = StylometryClassifier(model_type="svm", cv_folds=2, random_state=42)
        results = clf.train_hierarchical(self.df_X, self.s_y_author, self.s_y_era)

        self.assertTrue(clf.is_fitted)
        self.assertTrue(clf.is_era_fitted)
        self.assertEqual(len(clf.classes_), 3)
        self.assertEqual(len(clf.era_classes_), 2)
        self.assertIn("author_layer", results)
        self.assertIn("era_layer", results)

    def test_predict_era(self) -> None:
        """Katman 1: Edebi dönem tahmini ve olasılıkları testi."""
        clf = StylometryClassifier(model_type="svm", cv_folds=2, random_state=42)
        clf.train_hierarchical(self.df_X, self.s_y_author, self.s_y_era)

        # Renaissance örneği
        sample_ren = self.df_X.iloc[0]
        era, era_probs = clf.predict_era(sample_ren)
        self.assertIn(era, ["Renaissance", "Modernism"])
        self.assertAlmostEqual(sum(era_probs.values()), 1.0, places=4)

    def test_predict_top_authors(self) -> None:
        """Katman 2: En olası ilk K yazar adayının olasılıklarıyla sıralanması testi."""
        clf = StylometryClassifier(model_type="svm", cv_folds=2, random_state=42)
        clf.train_hierarchical(self.df_X, self.s_y_author, self.s_y_era)

        sample = self.df_X.iloc[0]
        top_2 = clf.predict_top_authors(sample, top_k=2)
        self.assertEqual(len(top_2), 2)
        self.assertGreaterEqual(top_2[0][1], top_2[1][1])

        # Dönem filtresi testi
        top_ren = clf.predict_top_authors(sample, era_filter="Renaissance", top_k=2)
        for auth, prob in top_ren:
            self.assertIn(auth, ["Shakespeare", "Marlowe"])

    def test_compute_burrows_delta(self) -> None:
        """Burrows' Delta Z-skor Manhattan mesafesinin matematiksel testi."""
        clf = StylometryClassifier(model_type="svm", cv_folds=2, random_state=42)
        clf.train_hierarchical(self.df_X, self.s_y_author, self.s_y_era)

        sample = self.df_X.iloc[0]
        delta_list = clf.compute_burrows_delta(sample, top_k=3)

        self.assertEqual(len(delta_list), 3)
        # delta_list: [(yazar, delta_mesafesi, benzerlik_orani), ...]
        first_author, first_delta, first_sim = delta_list[0]
        second_author, second_delta, second_sim = delta_list[1]

        # Mesafe pozitif olmalı
        self.assertGreaterEqual(first_delta, 0.0)
        # En yakın yazarın mesafesi 2. sıradakinden küçük veya eşit olmalı
        self.assertLessEqual(first_delta, second_delta)
        # Benzerlik skoru 0 ile 100 arasında olmalı
        self.assertGreater(first_sim, 0.0)
        self.assertLessEqual(first_sim, 100.0)

    def test_predict_proba_sum_to_one(self) -> None:
        """Sınıf olasılıklarının toplamının tam 1.0 olması testi."""
        clf = StylometryClassifier(model_type="svm", cv_folds=2, random_state=42)
        clf.train(self.df_X, self.s_y_author)

        probs = clf.predict_proba(self.df_X.iloc[0])
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=5)

    def test_unfitted_model_error(self) -> None:
        """Eğitilmemiş model hatası testi."""
        clf = StylometryClassifier()
        with self.assertRaises(RuntimeError):
            clf.predict(self.df_X)

    def test_hierarchical_conditioning_consistency(self) -> None:
        """Hiyerarşik koşullu tahminin dönem-yazar tutarlılığını sağlaması testi."""
        clf = StylometryClassifier(model_type="svm", cv_folds=2, random_state=42)
        clf.train_hierarchical(self.df_X, self.s_y_author, self.s_y_era)

        # Renaissance özelliklerine sahip örnek
        sample_ren = self.df_X.iloc[0]
        pred_era, era_probs = clf.predict_era(sample_ren)

        # Hiyerarşik tahmin sonuçları
        top_hierarchical = clf.predict_top_authors(sample_ren, top_k=3, use_hierarchical=True)
        top_raw = clf.predict_top_authors(sample_ren, top_k=3, use_hierarchical=False)

        top_h_author, top_h_prob = top_hierarchical[0]

        # 1. En güçlü yazar adayı, tahmin edilen dönemin yazarı olmalı
        self.assertEqual(clf.author_to_era_map_[top_h_author], pred_era)

        # 2. Hiyerarşik olasılıkların toplamı 1.0 olmalı
        all_hierarchical = clf.predict_top_authors(sample_ren, top_k=len(clf.classes_), use_hierarchical=True)
        total_prob = sum(prob for _, prob in all_hierarchical)
        self.assertAlmostEqual(total_prob, 1.0, places=4)

        # 3. predict() metodu use_hierarchical=True ile dönemle tutarlı yazarı döndürmeli
        pred_author = clf.predict(sample_ren, use_hierarchical=True)
        self.assertEqual(pred_author, top_h_author)
        self.assertEqual(clf.author_to_era_map_[str(pred_author)], pred_era)


if __name__ == "__main__":
    unittest.main()
