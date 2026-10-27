"""
Metin Ön İşleme ve Tokenizasyon Birim Testleri (test_preprocessor.py)
=====================================================================
Proje: TÜBİTAK - Stilometri ile Edebi Eserlerde Yazar Tespiti ve Üslup Analizi
"""

from __future__ import annotations

import unittest

from src.preprocessor import TextPreprocessor, turkish_lower


class TestTextPreprocessor(unittest.TestCase):
    """TextPreprocessor sınıfı ve Türkçe yardımcı fonksiyonlarının birim testleri."""

    def setUp(self) -> None:
        """Her testten önce çalıştırılacak kurulum."""
        self.preprocessor = TextPreprocessor()

    def test_turkish_lower(self) -> None:
        """Türkçe karakterlere özgü küçük harfe dönüştürme (I->ı, İ->i) testleri."""
        self.assertEqual(turkish_lower("IŞIK"), "ışık")
        self.assertEqual(turkish_lower("İSTANBUL"), "istanbul")
        self.assertEqual(turkish_lower("Ilık Bir İlkbahar"), "ılık bir ilkbahar")
        self.assertEqual(turkish_lower("ÇĞİÖŞÜ"), "çğiöşü")
        self.assertEqual(turkish_lower(""), "")

    def test_tokenize_sentences_basic(self) -> None:
        """Standart cümle bitiş noktalarının doğru ayrıştırılması testi."""
        text = "Akdeniz'in dalgaları kıyıları dövüyordu. İhtiyar sahile bakıyordu."
        sentences = self.preprocessor.tokenize_sentences(text)
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0], "Akdeniz'in dalgaları kıyıları dövüyordu.")
        self.assertEqual(sentences[1], "İhtiyar sahile bakıyordu.")

    def test_tokenize_sentences_variety_punctuations(self) -> None:
        """Soru işareti, ünlem ve üç nokta gibi farklı cümle sonlandırıcılar testi."""
        text = "Bu gelen kimdi? Derhal dur! Yolun sonu görünmüyor..."
        sentences = self.preprocessor.tokenize_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertTrue(sentences[0].endswith("?"))
        self.assertTrue(sentences[1].endswith("!"))
        self.assertTrue(sentences[2].endswith("..."))

    def test_tokenize_sentences_abbreviations(self) -> None:
        """Kısaltmaların (Prof., Dr., vb.) cümleyi yanlış bölmemesi testi."""
        text = "Prof. Dr. Ahmet Bey toplantıya katıldı. Konuyla ilgili raporu sundu."
        sentences = self.preprocessor.tokenize_sentences(text)
        self.assertEqual(len(sentences), 2)
        self.assertTrue(sentences[0].startswith("Prof. Dr."))

    def test_tokenize_sentences_empty(self) -> None:
        """Boş metin durumunda boş liste dönmesi testi."""
        self.assertEqual(self.preprocessor.tokenize_sentences(""), [])
        self.assertEqual(self.preprocessor.tokenize_sentences("    "), [])

    def test_tokenize_words(self) -> None:
        """Kelime tokenizasyonu ve Türkçe karakterlerin korunması testi."""
        text = "Çiçekler, ağaçlar ve ılık yağmur..."
        tokens = self.preprocessor.tokenize_words(text, lower=True)
        expected = ["çiçekler", "ağaçlar", "ve", "ılık", "yağmur"]
        self.assertEqual(tokens, expected)

    def test_tokenize_words_apostrophe(self) -> None:
        """Kesme işaretiyle ayrılan eklerin yönetimi testi."""
        text = "İstanbul'un Boğazı"
        tokens_stripped = self.preprocessor.tokenize_words(text, lower=True, keep_apostrophe=False)
        self.assertEqual(tokens_stripped, ["istanbul", "boğazı"])

    def test_extract_punctuations(self) -> None:
        """Metin içerisindeki noktalama işaretlerinin sırasıyla yakalanması testi."""
        text = "Gözlerini açtı; deniz dalgalıydı, rüzgâr sertti! Neden durdu?..."
        puncts = self.preprocessor.extract_punctuations(text)
        self.assertIn(";", puncts)
        self.assertIn(",", puncts)
        self.assertIn("!", puncts)
        self.assertIn("?", puncts)
        self.assertIn("...", puncts)


if __name__ == "__main__":
    unittest.main()
