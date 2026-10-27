"""
Edebi Soy Ağacı ve Etki Ağı Birim Testleri (test_literary_network.py)
===================================================================
Proje: TÜBİTAK - Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası
Developed by Bayram Akpamukçu
"""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.feature_extractor import StylometricFeatureExtractor
from src.literary_network import LiteraryPhylogenyEngine
from src.visualizer import plot_author_distance_heatmap, plot_literary_dendrogram


class TestLiteraryNetwork(unittest.TestCase):
    """Edebi soy ağacı (Dendrogram), mesafe matrisi ve üslup ikizleri testleri."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.extractor = StylometricFeatureExtractor()

        # 4 yazarlı örnek sentetik veri seti
        passages = {
            "Geoffrey Chaucer": [
                "Whan that Aprille with his shoures soote the droghte of March hath perced to the roote.",
                "And bathed every veyne in swich licour of which vertu engendred is the flour; eek soote.",
            ],
            "Christopher Marlowe": [
                "Was this the face that launch'd a thousand ships, and burnt the topless towers of Ilium?",
                "Sweet Helen, make me immortal with a kiss! Her lips suck forth my soul; see where it flies!",
            ],
            "William Shakespeare": [
                "To be, or not to be, that is the question: whether 'tis nobler in the mind to suffer.",
                "The slings and arrows of outrageous fortune, or to take arms against a sea of troubles.",
            ],
            "Jonathan Swift": [
                "My father had a small estate in Nottinghamshire; I was the third of five sons.",
                "He sent me to Emanuel College in Cambridge, at fourteen years old, where I resided.",
            ],
        }

        eras_dict = {
            "Geoffrey Chaucer": "Middle English",
            "Christopher Marlowe": "Renaissance",
            "William Shakespeare": "Renaissance",
            "Jonathan Swift": "Enlightenment",
        }

        data = []
        authors = []
        eras = []
        for auth, texts in passages.items():
            for t in texts:
                feats = cls.extractor.extract_features(t)
                data.append(feats)
                authors.append(auth)
                eras.append(eras_dict[auth])

        cls.df_X = pd.DataFrame(data)
        cls.authors = authors
        cls.eras = eras

        cls.engine = LiteraryPhylogenyEngine(
            X_corpus=cls.df_X,
            y_authors=cls.authors,
            y_eras=cls.eras,
            metric="cosine",
        )

    def test_distance_matrix_properties(self) -> None:
        """Mesafe matrisinin simetrik ve köşegenlerinin tam 0.0 olduğunu test et."""
        dist_df = self.engine.distance_df_
        self.assertIsNotNone(dist_df)
        self.assertEqual(dist_df.shape[0], dist_df.shape[1])
        self.assertEqual(len(dist_df), 4)

        # 1. Köşegenlerin 0 olması
        for auth in dist_df.index:
            self.assertAlmostEqual(dist_df.loc[auth, auth], 0.0, places=6)

        # 2. Simetri testi: D_ij == D_ji
        matrix_vals = dist_df.values
        np.testing.assert_allclose(matrix_vals, matrix_vals.T, atol=1e-7)

    def test_get_stylistic_soulmates(self) -> None:
        """Seçilen yazarın kendisi hariç en yakın üslup ikizlerinin sıralı döndüğünü test et."""
        soulmates = self.engine.get_stylistic_soulmates("William Shakespeare", top_k=2)

        self.assertEqual(len(soulmates), 2)
        # Kendisi sonuçlar arasında olmamalı
        soulmate_names = [s["author"] for s in soulmates]
        self.assertNotIn("William Shakespeare", soulmate_names)

        # Mesafeye göre artan sırada olmalı
        self.assertLessEqual(soulmates[0]["distance"], soulmates[1]["distance"])

        # Gerekli anahtarlar mevcut olmalı
        for s in soulmates:
            self.assertIn("author", s)
            self.assertIn("distance", s)
            self.assertIn("similarity_pct", s)
            self.assertIn("era", s)
            self.assertIn("relationship_type", s)
            self.assertIn("stylistic_insight", s)
            self.assertGreaterEqual(s["similarity_pct"], 0.0)
            self.assertLessEqual(s["similarity_pct"], 100.0)

    def test_compute_linkage_matrix_shape(self) -> None:
        """Hiyerarşik kümeleme linkage matrisinin (N-1, 4) boyutunda olduğunu test et."""
        linkage_mat = self.engine.compute_linkage_matrix(method="average")

        n_authors = len(self.engine.authors_)
        self.assertEqual(linkage_mat.shape, (n_authors - 1, 4))
        # Mesafelerin pozitif olduğunu doğrula
        self.assertTrue(np.all(linkage_mat[:, 2] >= 0.0))

    def test_visualizer_dendrogram_and_heatmap(self) -> None:
        """Dendrogram ve Mesafe Isı Haritası fonksiyonlarının Figure nesnesi ürettiğini test et."""
        linkage_mat = self.engine.compute_linkage_matrix(method="average")
        fig_dendro = plot_literary_dendrogram(
            linkage_matrix=linkage_mat,
            authors=self.engine.authors_,
            eras=[self.engine.eras_map_.get(a, "") for a in self.engine.authors_],
        )
        self.assertIsInstance(fig_dendro, Figure)

        fig_heat = plot_author_distance_heatmap(
            distance_matrix=self.engine.distance_df_,
            authors=self.engine.authors_,
        )
        self.assertIsInstance(fig_heat, Figure)

    def test_burrows_delta_metric_option(self) -> None:
        """Burrows' Delta (cityblock Z-skor) metriğinin de hatasız çalıştığını test et."""
        engine_delta = LiteraryPhylogenyEngine(
            X_corpus=self.df_X,
            y_authors=self.authors,
            y_eras=self.eras,
            metric="burrows_delta",
        )
        self.assertIsNotNone(engine_delta.distance_df_)
        matrix_vals = engine_delta.distance_df_.values
        np.testing.assert_allclose(matrix_vals, matrix_vals.T, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
