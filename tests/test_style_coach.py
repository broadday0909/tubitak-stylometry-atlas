"""
Üslup Koçu ve Reçete Motoru Birim Testleri (test_style_coach.py)
==============================================================
Proje: TÜBİTAK - Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası
Developed by Bayram Akpamukçu
"""

from __future__ import annotations

import unittest
import pandas as pd

from src.classifier import StylometryClassifier
from src.feature_extractor import StylometricFeatureExtractor
from src.style_coach import StylometricCoach


class TestStyleCoach(unittest.TestCase):
    """Üslup karşılaştırma, reçete üretimi ve işaretçi tespiti birim testleri."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.extractor = StylometricFeatureExtractor()

        # Basit 2 yazarlı eğitim külliyatı
        passages = {
            "William Shakespeare": [
                "To be, or not to be, that is the question: whether 'tis nobler in the mind to suffer.",
                "The slings and arrows of outrageous fortune, or to take arms against a sea of troubles.",
                "O mighty king, behold the royal crown upon thy sovereign throne; thou hath seen destiny.",
            ],
            "George Orwell": [
                "It was a bright cold day in April, and the clocks were striking thirteen.",
                "Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind.",
                "The ministry of truth contained three thousand rooms above ground level and corresponding ramifications.",
            ],
        }

        data = []
        authors = []
        eras = []
        for auth, texts in passages.items():
            for t in texts:
                feats = cls.extractor.extract_features(t)
                data.append(feats)
                authors.append(auth)
                eras.append("Renaissance" if "Shakespeare" in auth else "Modernism")

        cls.df_X = pd.DataFrame(data)
        cls.authors = authors
        cls.eras = eras

        cls.classifier = StylometryClassifier(model_type="svm", cv_folds=2, random_state=42, feature_extractor=cls.extractor)
        cls.classifier.train_hierarchical(cls.df_X, cls.authors, cls.eras)
        cls.coach = StylometricCoach(cls.classifier, cls.extractor)

    def test_compare_with_target_author(self) -> None:
        """Kullanıcı metni ile hedef yazar arasındaki sapma ve uyum skorunu doğrula."""
        user_text = (
            "This is a very short text. It is very simple. There are no big words here. "
            "I write quickly and simply."
        )
        res = self.coach.compare_with_target_author(user_text, target_author="William Shakespeare")

        self.assertIn("target_author", res)
        self.assertIn("similarity_pct", res)
        self.assertIn("top_deviations", res)
        self.assertGreaterEqual(res["similarity_pct"], 10.0)
        self.assertLessEqual(res["similarity_pct"], 100.0)

        # En büyük 5 sapmanın yapısını doğrula
        deviations = res["top_deviations"]
        self.assertLessEqual(len(deviations), 5)
        for dev in deviations:
            self.assertIn("feature", dev)
            self.assertIn("friendly_name", dev)
            self.assertIn("user_value", dev)
            self.assertIn("target_value", dev)
            self.assertIn("z_diff", dev)
            self.assertIn("direction", dev)

    def test_generate_prescription(self) -> None:
        """Adım adım somut reçetelerin üretildiğini doğrula."""
        user_text = (
            "The quick brown fox jumps over the lazy dog. He ran quickly, silently, and smoothly; "
            "however, the weather was terribly cold."
        )
        prescriptions = self.coach.generate_prescription(user_text, target_author="George Orwell")

        self.assertGreaterEqual(len(prescriptions), 2)
        for p in prescriptions:
            self.assertIn("category", p)
            self.assertIn("action", p)
            self.assertIn("recommendation", p)
            self.assertIn("impact", p)
            self.assertGreater(len(p["recommendation"]), 15)

    def test_highlight_stylistic_markers(self) -> None:
        """Aşırı uzun cümle, zarf (-ly) ve arkaik kelimelerin doğru yakalandığını test et."""
        # 30+ kelimelik uzun cümle ve işaretçiler
        long_sent = (
            "To be or not to be that is the eternal question whether it is truly nobler in the troubled mind to suffer "
            "the cruel slings and painful arrows of outrageous fortune or to take valiant arms against a vast endless sea of "
            "calamitous troubles and by opposing them to end our mortal strife."
        )
        short_sent = "He spoke slowly and dreadfully; thou hath seen the truth."
        test_text = f"{long_sent}. {short_sent}"

        markers = self.coach.highlight_stylistic_markers(test_text)

        self.assertEqual(markers["long_sentence_count"], 1)
        self.assertEqual(len(markers["long_sentences"]), 1)
        self.assertGreaterEqual(markers["long_sentences"][0]["word_count"], 30)

        # -ly zarfları
        self.assertIn("slowly", markers["adverbs_ly"])
        self.assertIn("dreadfully", markers["adverbs_ly"])

        # Arkaik kelimeler
        self.assertIn("thou", markers["archaic_words"])
        self.assertIn("hath", markers["archaic_words"])

        # Noktalı virgül
        self.assertEqual(markers["semicolon_count"], 1)


if __name__ == "__main__":
    unittest.main()
