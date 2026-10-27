"""
TÜBİTAK Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası
Birim Testleri: Akademik Veri İhracı ve Çoklu Dosya Analizcisi (test_academic_exporter.py)
Developed by Bayram Akpamukçu
"""

import unittest
import pandas as pd
import numpy as np

from src.academic_exporter import AcademicExporter
from src.classifier import StylometryClassifier
from src.feature_extractor import StylometricFeatureExtractor


class TestAcademicExporter(unittest.TestCase):
    """AcademicExporter sınıfı birim testleri."""

    def setUp(self):
        self.extractor = StylometricFeatureExtractor()
        self.exporter = AcademicExporter(extractor=self.extractor)

    def test_export_corpus_summary_csv(self):
        """Külliyat stilometrik matrisinin CSV olarak doğru ihraç edildiğini test et."""
        df_features = pd.DataFrame({
            "avg_sentence_length_words": [12.5, 24.1, 18.0],
            "ttr": [0.65, 0.45, 0.58],
            "suffix_ly_ratio": [0.03, 0.08, 0.04],
        })
        authors = ["William Shakespeare", "Charles Dickens", "George Orwell"]
        eras = ["Renaissance & Early Modern", "Victorian Era", "Modernism"]

        csv_str = self.exporter.export_corpus_summary_csv(df_features, authors, eras)

        self.assertIsInstance(csv_str, str)
        self.assertIn("Sample_ID,Author,Era,avg_sentence_length_words,ttr,suffix_ly_ratio", csv_str)
        self.assertIn("William Shakespeare", csv_str)
        self.assertIn("Charles Dickens", csv_str)
        self.assertIn("George Orwell", csv_str)
        self.assertIn("SAMPLE_0001", csv_str)
        self.assertIn("SAMPLE_0003", csv_str)

    def test_export_latex_results_table(self):
        """Akademik LaTeX sonuç tablosunun hatasız oluşturulduğunu test et."""
        cv_results = {
            "era_layer": {
                "mean_accuracy": 1.0,
                "std_accuracy": 0.0,
                "mean_f1_macro": 1.0,
                "mean_precision_macro": 1.0,
                "mean_recall_macro": 1.0,
            },
            "author_layer": {
                "mean_accuracy": 0.948,
                "std_accuracy": 0.021,
                "mean_f1_macro": 0.942,
                "mean_precision_macro": 0.951,
                "mean_recall_macro": 0.940,
            },
        }
        test_metrics = {
            "accuracy": 0.962,
            "classification_report_dict": {
                "macro avg": {
                    "precision": 0.965,
                    "recall": 0.955,
                    "f1-score": 0.958,
                }
            },
        }

        latex_str = self.exporter.export_latex_results_table(
            cv_results=cv_results,
            test_metrics=test_metrics,
            caption="Test Tablosu",
            label="tab:test_metrics",
        )

        self.assertIsInstance(latex_str, str)
        self.assertIn("\\begin{table}", latex_str)
        self.assertIn("\\end{table}", latex_str)
        self.assertIn("\\caption{Test Tablosu}", latex_str)
        self.assertIn("\\label{tab:test_metrics}", latex_str)
        self.assertIn("Developed by Bayram Akpamukçu", latex_str)
        self.assertIn("Katman 1: Edebi Dönem", latex_str)
        self.assertIn("Katman 2: Kanonik Yazar", latex_str)
        self.assertIn("100.0\\%", latex_str)
        self.assertIn("94.8\\%", latex_str)

    def test_batch_analyze_files(self):
        """Yüklenen çoklu dosyaların toplu özet tablosunun oluşturulduğunu test et."""
        uploaded_files = {
            "hamlet_scene.txt": (
                "To be, or not to be, that is the question: whether 'tis nobler in the mind to suffer "
                "the slings and arrows of outrageous fortune, or to take arms against a sea of troubles."
            ),
            "dickens_intro.md": (
                "It was the best of times, it was the worst of times, it was the age of wisdom, "
                "it was the age of foolishness, it was the epoch of belief, it was the season of light."
            ),
        }

        df_batch = self.exporter.batch_analyze_files(uploaded_files)

        self.assertIsInstance(df_batch, pd.DataFrame)
        self.assertEqual(len(df_batch), 2)
        self.assertIn("Dosya Adı", df_batch.columns)
        self.assertIn("Kelime Sayısı", df_batch.columns)
        self.assertIn("Sözcük Çeşitliliği (TTR)", df_batch.columns)
        self.assertIn("Ort. Cümle Boyutu (Kelime)", df_batch.columns)
        self.assertEqual(df_batch.iloc[0]["Dosya Adı"], "hamlet_scene.txt")
        self.assertEqual(df_batch.iloc[1]["Dosya Adı"], "dickens_intro.md")
        self.assertGreater(df_batch.iloc[0]["Kelime Sayısı"], 10)

    def test_orwell_vs_woolf_batch_distinction(self):
        """Modernist yazarlar George Orwell ve Virginia Woolf'un ayırt edildiğini test et."""
        orwell_text = (
            "It was a bright cold day in April, and the clocks were striking thirteen. "
            "Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, "
            "slipped quickly through the glass doors of Victory Mansions."
        )
        woolf_text = (
            "Mrs. Dalloway said she would buy the flowers herself. For Lucy had her work cut out for her. "
            "And then, thought Clarissa Dalloway, what a morning—fresh as if issued to children on a beach. "
            "What a lark! What a plunge!"
        )

        orwell_feats = self.extractor.extract_features(orwell_text)
        woolf_feats = self.extractor.extract_features(woolf_text)

        # Orwell doğrudan ve saydam sentaks, Woolf ise parçalanmış/lirik sentaks sergilemelidir
        self.assertGreater(orwell_feats["clarity_directness_index"], woolf_feats["clarity_directness_index"])
        self.assertGreater(woolf_feats["stream_of_consciousness_index"], orwell_feats["stream_of_consciousness_index"])
        self.assertGreater(orwell_feats["political_lexicon_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
