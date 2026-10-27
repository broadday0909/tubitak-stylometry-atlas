"""
Kanon Kataloğu ve Gutenberg Modülü Birim Testleri (test_canon_and_gutenberg.py)
=============================================================================
Proje: TÜBİTAK - Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası
"""

from __future__ import annotations

import unittest
from src.canon_catalog import CANON_CATALOG, LITERARY_ERAS, get_all_eras, get_authors_by_era
from src.gutenberg_fetcher import clean_gutenberg_text


class TestCanonAndGutenberg(unittest.TestCase):
    """Kanon kataloğu bütünlüğü ve Gutenberg metin temizleme testleri."""

    def test_eras_coverage(self) -> None:
        """7 ana dönemin eksiksiz tanımlandığını doğrula."""
        eras = get_all_eras()
        self.assertEqual(len(eras), 7)
        self.assertIn("Middle English", eras)
        self.assertIn("Renaissance & Early Modern", eras)
        self.assertIn("Enlightenment & 18th Century", eras)
        self.assertIn("Romanticism", eras)
        self.assertIn("Victorian Era", eras)
        self.assertIn("American Classics", eras)
        self.assertIn("Modernism", eras)

    def test_canon_authors_count_and_fields(self) -> None:
        """En az 35 kanonik yazar olduğunu ve gerekli alanların dolu olduğunu doğrula."""
        self.assertGreaterEqual(len(CANON_CATALOG), 35)

        for author_id, entry in CANON_CATALOG.items():
            self.assertTrue(entry.name, f"{author_id} ismi boş!")
            self.assertIn(entry.era, LITERARY_ERAS, f"{entry.name} dönemi geçersiz: {entry.era}")
            self.assertGreater(entry.gutenberg_id, 0, f"{entry.name} Gutenberg ID geçersiz!")
            self.assertTrue(entry.canonical_work, f"{entry.name} kanonik eseri boş!")
            self.assertGreaterEqual(len(entry.sample_passages), 1, f"{entry.name} örnek pasajı yok!")

    def test_authors_by_era(self) -> None:
        """Dönem bazlı yazar sorgusunun doğru çalıştığını doğrula."""
        victorian_authors = get_authors_by_era("Victorian Era")
        names = [a.name for a in victorian_authors]
        self.assertIn("Charles Dickens", names)
        self.assertIn("Oscar Wilde", names)
        self.assertIn("Charlotte Brontë", names)

    def test_clean_gutenberg_text(self) -> None:
        """Gutenberg başlık ve lisans şablonlarının temizlenmesi testi."""
        raw_mock = (
            "The Project Gutenberg eBook of Sample Work\n"
            "Author: Famous Writer\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE WORK ***\n"
            "Chapter 1\nIt was a dark and stormy night.\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE WORK ***\n"
            "End of the Project Gutenberg EBook\n"
        )
        cleaned = clean_gutenberg_text(raw_mock)
        self.assertEqual(cleaned, "Chapter 1\nIt was a dark and stormy night.")
        self.assertNotIn("START OF THE PROJECT GUTENBERG", cleaned)
        self.assertNotIn("END OF THE PROJECT GUTENBERG", cleaned)


if __name__ == "__main__":
    unittest.main()
