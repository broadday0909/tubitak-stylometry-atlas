"""
Gelişmiş Stilometrik Öznitelik Çıkarıcı Birim Testleri (test_feature_extractor.py)
=================================================================================
Proje: TÜBİTAK - Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası
"""

from __future__ import annotations

import math
import unittest
import pandas as pd

from src.feature_extractor import StylometricFeatureExtractor


class TestStylometricFeatureExtractor(unittest.TestCase):
    """Evrensel öznitelik çıkarıcının matematiksel ve edebi metrik testleri."""

    def setUp(self) -> None:
        self.extractor = StylometricFeatureExtractor()

    def test_mathematical_metrics_controlled_text(self) -> None:
        """Standart metin üzerinden matematiksel metriklerin doğrulanması."""
        sample_text = "Hava güzeldi ve güneş açtı. Hava sıcaktı ve rüzgâr dindi."
        features = self.extractor.extract_features(sample_text)

        self.assertEqual(features["total_sentences"], 2.0)
        self.assertEqual(features["total_tokens"], 10.0)
        self.assertEqual(features["total_unique_tokens"], 8.0)
        self.assertAlmostEqual(features["avg_sentence_length_words"], 5.0, places=4)
        self.assertAlmostEqual(features["ttr"], 0.80, places=4)
        self.assertAlmostEqual(features["fw_ve_ratio"], 0.20, places=4)

    def test_archaic_words_extraction(self) -> None:
        """Orta Çağ ve Rönesans arkaik sözcük ve zamir oranlarının (thou, thee, hath) testi."""
        archaic_text = "Thou art my guide, and thee I follow; for he hath spoken and doth know."
        features = self.extractor.extract_features(archaic_text)

        # thou, thee, hath, doth, art -> 5 arkaik kelime
        self.assertGreater(features["archaic_total_ratio"], 0.0)
        self.assertGreater(features["archaic_thou_ratio"], 0.0)
        self.assertGreater(features["archaic_thee_ratio"], 0.0)
        self.assertGreater(features["archaic_hath_ratio"], 0.0)
        self.assertGreater(features["archaic_doth_ratio"], 0.0)

    def test_morphological_suffixes(self) -> None:
        """Morfolojik eklerin (-eth, -est, -ly, -ing, -tion) doğru yakalanması testi."""
        morpho_text = "He maketh silence gently, thinking deeply of salvation and perfection."
        features = self.extractor.extract_features(morpho_text)

        # maketh (-eth), gently (-ly), deeply (-ly), thinking (-ing), salvation (-tion), perfection (-tion)
        self.assertGreater(features["suffix_eth_ratio"], 0.0)
        self.assertGreater(features["suffix_ly_ratio"], 0.0)
        self.assertGreater(features["suffix_ing_ratio"], 0.0)
        self.assertGreater(features["suffix_tion_sion_ratio"], 0.0)
        self.assertGreater(features["suffix_total_ratio"], 0.0)

    def test_stream_of_consciousness_index(self) -> None:
        """Modernist bilinç akışı göstergelerinin (tire, üç nokta, parantez) testi."""
        modernist_text = "What a morning—fresh as if issued to children on a beach... (feeling as she did) solemn; looking at the flowers."
        features = self.extractor.extract_features(modernist_text)

        self.assertGreater(features["stream_of_consciousness_index"], 0.0)
        self.assertGreater(features["dash_frequency_ratio"], 0.0)
        self.assertGreater(features["ellipsis_frequency_ratio"], 0.0)
        self.assertGreater(features["parentheses_frequency_ratio"], 0.0)

    def test_poetic_vs_prose_metrics(self) -> None:
        """Şiir satır sonu ve diyalog tırnak işaretlerinin testi."""
        poetic_text = "Shall I compare thee to a summer's day?\nThou art more lovely and more temperate:\nRough winds do shake the darling buds of May."
        features = self.extractor.extract_features(poetic_text)

        self.assertGreater(features["poetic_line_break_ratio"], 0.0)
        self.assertAlmostEqual(features["poetic_line_capitalization_ratio"], 1.0, places=2)

    def test_empty_text_edge_case(self) -> None:
        """Boş metin durumunda sıfır değerler dönmesi testi."""
        features = self.extractor.extract_features("")
        self.assertEqual(features["total_tokens"], 0.0)
        self.assertEqual(features["archaic_total_ratio"], 0.0)
        self.assertEqual(features["stream_of_consciousness_index"], 0.0)
        self.assertEqual(features["ttr"], 0.0)


if __name__ == "__main__":
    unittest.main()
