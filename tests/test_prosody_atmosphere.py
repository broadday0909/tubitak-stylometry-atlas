"""
Prosodi ve 6 Boyutlu Edebi Atmosfer Birim Testleri (test_prosody_atmosphere.py)
=============================================================================
Proje: TÜBİTAK - Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası
Developed by Bayram Akpamukçu
"""

from __future__ import annotations

import unittest
from matplotlib.figure import Figure

from src.prosody_atmosphere import ProsodyAndAtmosphereAnalyzer
from src.visualizer import plot_atmosphere_radar


class TestProsodyAndAtmosphere(unittest.TestCase):
    """Hece sayımı, vezin/ritim analizi, 6D atmosfer radarı birim testleri."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.analyzer = ProsodyAndAtmosphereAnalyzer()

    def test_count_syllables_rules(self) -> None:
        """İngilizce kelimelerde hece sayma kurallarının doğrulanması."""
        self.assertEqual(self.analyzer.count_syllables("cat"), 1)
        self.assertEqual(self.analyzer.count_syllables("the"), 1)
        self.assertEqual(self.analyzer.count_syllables("question"), 2)
        self.assertEqual(self.analyzer.count_syllables("nobler"), 2)
        self.assertEqual(self.analyzer.count_syllables("fortune"), 2)
        self.assertEqual(self.analyzer.count_syllables("beautiful"), 3)
        self.assertEqual(self.analyzer.count_syllables("monarch"), 2)
        self.assertEqual(self.analyzer.count_syllables("agony"), 3)
        self.assertEqual(self.analyzer.count_syllables("examination"), 5)
        # Kenar durumlar
        self.assertEqual(self.analyzer.count_syllables(""), 0)
        self.assertEqual(self.analyzer.count_syllables("a"), 1)

    def test_analyze_poetic_meter_blank_verse(self) -> None:
        """Shakespeare'in ünlü Hamlet tiradında Blank Verse (10 heceli vezin) testi."""
        hamlet_lines = (
            "To be, or not to be, that is the question:\n"
            "Whether 'tis nobler in the mind to suffer\n"
            "The slings and arrows of outrageous fortune,\n"
            "Or to take arms against a sea of troubles,\n"
            "And by opposing end them."
        )
        res = self.analyzer.analyze_poetic_meter(hamlet_lines)

        self.assertIn("avg_syllables_per_line", res)
        self.assertIn("iambic_pentameter_pct", res)
        self.assertIn("meter_regularity_index", res)
        self.assertIn("poetic_form_tendency", res)

        # Ortalama hece sayısı ~10-11 olmalı
        self.assertGreaterEqual(res["avg_syllables_per_line"], 8.5)
        self.assertLessEqual(res["avg_syllables_per_line"], 11.5)

        # Iambic pentameter oranı en az %50 olmalı
        self.assertGreaterEqual(res["iambic_pentameter_pct"], 50.0)
        self.assertIn("Blank Verse", res["poetic_form_tendency"])

    def test_analyze_literary_atmosphere_gothic(self) -> None:
        """Poe ve Gotik edebiyat temalı metinde Gothic & Macabre baskınlığı testi."""
        gothic_text = (
            "True! nervous very dreadfully nervous I had been; but why will you say that I am mad? "
            "The disease had sharpened my senses. Above all was the sense of hearing acute. "
            "I heard the beating of the hideous heart in the dark tomb of death and horror. "
            "The pale phantom and the dread shadow of the grave haunted my tormented soul with gloomy terror."
        )
        scores = self.analyzer.analyze_literary_atmosphere(gothic_text)

        self.assertEqual(len(scores), 6)
        self.assertIn("Gothic & Macabre", scores)
        self.assertGreater(scores["Gothic & Macabre"], 45.0)

        # Gotik skor diğer atmosferlerden belirgin şekilde yüksek olmalı
        self.assertGreater(scores["Gothic & Macabre"], scores["Victorian Social Realism"])

    def test_analyze_literary_atmosphere_courtly(self) -> None:
        """Shakespeare ve Rönesans saray/kader metninde Courtly & Archaic Majesty testi."""
        courtly_text = (
            "O mighty king, behold the royal crown upon thy sovereign throne! "
            "Thou art the noble monarch of this realm, and by thy sacred decree shall the kingdom triumph. "
            "Wherefore dost thou fear the treason of the sword when destiny hath ordained thy reign?"
        )
        scores = self.analyzer.analyze_literary_atmosphere(courtly_text)

        self.assertGreater(scores["Courtly & Archaic Majesty"], 50.0)
        self.assertGreater(scores["Courtly & Archaic Majesty"], scores["Modernist Psychological Flux"])

    def test_analyze_literary_atmosphere_romantic(self) -> None:
        """Wordsworth ve Romantik doğa/yücelik metninde Romantic & Pastoral Sublime testi."""
        romantic_text = (
            "I wandered lonely as a cloud that floats on high o'er vales and hills, "
            "when all at once I saw a crowd, a host of golden daffodils beside the lake, "
            "beneath the trees fluttering and dancing in the breeze. The sublime beauty of nature "
            "filled my solitary soul with immortal harmony and radiant spring ecstasy."
        )
        scores = self.analyzer.analyze_literary_atmosphere(romantic_text)

        self.assertGreater(scores["Romantic & Pastoral Sublime"], 50.0)

    def test_get_atmosphere_profile(self) -> None:
        """Tam atmosfer profilinin baskın kategori ve meter özetini ürettiğini doğrula."""
        sample = "The king hath decreed the doom of the treasonous lord upon the throne."
        profile = self.analyzer.get_atmosphere_profile(sample)

        self.assertIn("meter", profile)
        self.assertIn("atmosphere", profile)
        self.assertIn("dominant_atmosphere", profile)
        self.assertIn("dominant_score", profile)

    def test_plot_atmosphere_radar(self) -> None:
        """Radar grafiğinin tekil ve karşılaştırmalı modda Figure döndürdüğünü test et."""
        scores1 = {
            "Gothic & Macabre": 75.0,
            "Romantic & Pastoral Sublime": 20.0,
            "Courtly & Archaic Majesty": 15.0,
            "Satirical & Political Sharpness": 30.0,
            "Victorian Social Realism": 25.0,
            "Modernist Psychological Flux": 40.0,
        }
        fig_single = plot_atmosphere_radar(scores1, title="Test Radar Tekil")
        self.assertIsInstance(fig_single, Figure)

        scores2 = {
            "Gothic & Macabre": 10.0,
            "Romantic & Pastoral Sublime": 85.0,
            "Courtly & Archaic Majesty": 20.0,
            "Satirical & Political Sharpness": 15.0,
            "Victorian Social Realism": 35.0,
            "Modernist Psychological Flux": 50.0,
        }
        fig_comp = plot_atmosphere_radar(scores1, comparison_scores=scores2, label1="Poe", label2="Wordsworth")
        self.assertIsInstance(fig_comp, Figure)


if __name__ == "__main__":
    unittest.main()
