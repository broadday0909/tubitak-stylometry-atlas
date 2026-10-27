"""
TÜBİTAK Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası
Modül: Prosodi/Vezin Analizi ve 6 Boyutlu Edebi Atmosfer Radarı
Developed by Bayram Akpamukçu
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ProsodyAndAtmosphereAnalyzer:
    """
    İngilizce edebi metinlerin prosodi (hece/ritim düzeni, Iambic Pentameter
    / Blank Verse eğilimi) ve 6 boyutlu edebi atmosferini (Gothic, Romantic Sublime,
    Courtly Majesty, Satirical Sharpness, Victorian Realism, Modernist Flux) analiz eden motor.

    Developed by Bayram Akpamukçu
    """

    # 6 Edebi Atmosfer Sözlüğü (Kanonik Tematik Kelimeler ve Kökler)
    ATMOSPHERE_LEXICONS: Dict[str, List[str]] = {
        "Gothic & Macabre": [
            "death", "dark", "darkness", "gloom", "gloomy", "terror", "horror",
            "dread", "tomb", "grave", "crypt", "phantom", "ghost", "shadow",
            "corpse", "blood", "bleed", "bloody", "doom", "doomed", "agony",
            "pale", "ghastly", "sinister", "demon", "devil", "sorrow", "weep",
            "weeping", "haunt", "haunted", "decay", "sepulchre", "melancholy",
            "night", "mourn", "mournful", "abyss", "nightmare", "cursed", "grim",
            "scream", "chill", "chilling", "eerie", "morbid", "vault", "shroud",
            "coffin", "murder", "wicked", "fear", "spectre", "spectral",
        ],
        "Romantic & Pastoral Sublime": [
            "nature", "soul", "sublime", "beauty", "meadow", "forest", "green",
            "mountain", "hill", "valley", "flower", "bloom", "divine", "breeze",
            "wind", "stream", "brook", "river", "solitude", "solitary", "immortal",
            "ecstasy", "celestial", "wander", "wandering", "harmony", "spring",
            "bird", "song", "heavens", "heavenly", "dawn", "dusk", "sunset",
            "wild", "wilderness", "passion", "delight", "sacred", "radiant",
            "serenity", "autumn", "summer", "ocean", "sea", "pure", "grove",
            "woods", "sunshine", "tranquil", "tranquility", "enchant", "enchanted",
        ],
        "Courtly & Archaic Majesty": [
            "king", "queen", "prince", "monarch", "crown", "throne", "majesty",
            "royal", "lord", "noble", "nobility", "reign", "empire", "kingdom",
            "sword", "destiny", "fate", "treason", "glory", "honour", "honor",
            "knight", "conquest", "sovereignty", "sovereign", "realm", "sceptre",
            "command", "triumph", "valiant", "battle", "warrior", "decree",
            "thou", "thee", "thy", "thine", "hath", "doth", "art", "dost",
            "'tis", "'twas", "behold", "hither", "whither", "wherefore", "shall",
        ],
        "Satirical & Political Sharpness": [
            "party", "state", "government", "law", "laws", "minister", "power",
            "tyrant", "tyranny", "reason", "folly", "satire", "sarcastic", "irony",
            "corrupt", "corruption", "citizen", "freedom", "liberty", "empire",
            "rule", "ruler", "traitor", "hypocrisy", "hypocrite", "fool", "foolish",
            "court", "parliament", "public", "abuse", "authority", "decree",
            "censor", "propaganda", "system", "absurd", "ridicule", "money",
            "faction", "rebellion", "oppression", "plot", "policy", "bribe",
        ],
        "Victorian Social Realism": [
            "society", "marriage", "marry", "fortune", "estate", "gentleman",
            "gentlewoman", "lady", "money", "pound", "debt", "family", "father",
            "mother", "daughter", "son", "duty", "respectable", "respectability",
            "propriety", "manners", "class", "neighbourhood", "virtue", "pride",
            "prejudice", "income", "inheritance", "poverty", "workhouse", "mill",
            "street", "london", "domestic", "reputation", "modest", "governess",
            "parlour", "carriage", "clergyman", "scandal", "decorum", "tenant",
        ],
        "Modernist Psychological Flux": [
            "consciousness", "time", "memory", "thought", "mind", "moment",
            "clock", "strike", "striking", "pulse", "stream", "sensation",
            "interior", "fragment", "fragmented", "echo", "shadow", "drift",
            "drifting", "wave", "waves", "perception", "sense", "awareness",
            "silence", "rhythm", "flash", "instant", "fleeting", "illusion",
            "monologue", "reverie", "subconscious", "flow", "passing", "blur",
            "vibration", "whisper", "fleeting", "dissolve", "interval",
        ],
    }

    def count_syllables(self, word: str) -> int:
        """
        İngilizce bir kelimenin hece sayısını sesli harf kümeleri, çift sesliler
        ve sessiz 'e' kuralları üzerinden hesaplar.
        
        Args:
            word (str): İncelenecek kelime.
            
        Returns:
            int: Hesaplanmış hece sayısı (en az 1).
        """
        w = word.lower().strip()
        w = re.sub(r"[^a-z]", "", w)
        if not w:
            return 0
        if len(w) <= 3:
            return 1

        # Sessiz 'e' sonlarını temizle (le, ee, oe hariç)
        if w.endswith("e") and not w.endswith(("le", "ee", "oe")):
            w = w[:-1]

        # Düzenli 'ed' geçmiş zaman takısını temizle (ted, ded hariç ses vermez)
        if w.endswith("ed") and not w.endswith(("ted", "ded")) and len(w) > 4:
            w = w[:-2]

        # Ardışık sesli harf kümelerini say (a, e, i, o, u, y)
        matches = re.findall(r"[aeiouy]+", w)
        count = len(matches)

        # Kelime sonundaki sessiz harf + le yapısı (örn: little, bottle, noble)
        if word.lower().endswith("le") and len(w) >= 2 and w[-2] not in "aeiouy":
            count = max(count, 1)

        return max(1, count)

    def analyze_poetic_meter(self, text: str) -> Dict[str, Any]:
        """
        Metnin hece sayısını, dize/cümle başına hece dağılımını, 10 heceli
        Iambic Pentameter (Blank Verse) oranını ve ritim düzenliliğini hesaplar.
        
        Args:
            text (str): Analiz edilecek edebi metin.
            
        Returns:
            Dict[str, Any]: avg_syllables_per_line, iambic_pentameter_pct,
                           meter_regularity_index, poetic_form_tendency vb.
        """
        clean_text = text.strip()
        if not clean_text:
            return {
                "total_syllables": 0,
                "total_eval_units": 0,
                "avg_syllables_per_line": 0.0,
                "iambic_pentameter_pct": 0.0,
                "meter_regularity_index": 0.0,
                "poetic_form_tendency": "Boş Metin",
            }

        # Dize ayrıştırma: Yeni satırlar varsa doğrudan satırları kullan,
        # yoksa doğal sentaktik/ritmik parçalara (klauszlara) ayır.
        raw_lines = [l.strip() for l in clean_text.split("\n") if l.strip()]
        if len(raw_lines) >= 3:
            eval_units = raw_lines
        else:
            clauses = [
                c.strip()
                for c in re.split(r"[.,;!?—\n]+", clean_text)
                if len(c.strip().split()) >= 3
            ]
            eval_units = clauses if len(clauses) >= 2 else [clean_text]

        syllable_counts: List[int] = []
        for unit in eval_units:
            words = re.findall(r"[A-Za-z']+", unit)
            s_count = sum(self.count_syllables(w) for w in words)
            if s_count > 0:
                syllable_counts.append(s_count)

        if not syllable_counts:
            syllable_counts = [1]

        total_syllables = sum(syllable_counts)
        avg_syllables = float(np.mean(syllable_counts))
        std_syllables = float(np.std(syllable_counts))

        # Iambic Pentameter / Blank Verse: 9-11 hece aralığı
        pentameter_matches = sum(1 for s in syllable_counts if 9 <= s <= 11)
        pentameter_pct = round((pentameter_matches / len(syllable_counts)) * 100.0, 1)

        # Ritim Düzenlilik İndeksi (Düşük varyasyon = Yüksek ritmik düzenlilik)
        cv = std_syllables / avg_syllables if avg_syllables > 0 else 1.0
        if cv < 1.0:
            regularity = round(float(np.clip((1.0 - cv) * 100.0, 12.0, 98.0)), 1)
        else:
            regularity = round(float(np.clip(100.0 / (1.0 + cv), 12.0, 98.0)), 1)

        # Edebi Form Eğilimi
        if pentameter_pct >= 40.0 or (9.0 <= avg_syllables <= 11.5 and regularity >= 55.0):
            tendency = "Rönesans Blank Verse / Iambic Pentameter (Shakespeare & Milton)"
        elif avg_syllables <= 8.5 and regularity >= 50.0:
            tendency = "Lirik Balad / Kısa Şiirsel Dize (Romantic Ballad / Lyric Meter)"
        elif avg_syllables >= 16.0:
            tendency = "Geniş Anlatı / Viktorya & Modernist Düz Yazı (Prose Stream)"
        else:
            tendency = "Dengeli Klasik Edebi Anlatı (Rhythmic Classical Prose)"

        return {
            "total_syllables": total_syllables,
            "total_eval_units": len(syllable_counts),
            "avg_syllables_per_line": round(avg_syllables, 1),
            "iambic_pentameter_pct": pentameter_pct,
            "meter_regularity_index": regularity,
            "poetic_form_tendency": tendency,
        }

    def analyze_literary_atmosphere(self, text: str) -> Dict[str, float]:
        """
        Metni 6 edebi atmosfer boyutunda (0.0 - 100.0) puanlar.
        
        Boyutlar:
        1. Gothic & Macabre
        2. Romantic & Pastoral Sublime
        3. Courtly & Archaic Majesty
        4. Satirical & Political Sharpness
        5. Victorian Social Realism
        6. Modernist Psychological Flux
        
        Returns:
            Dict[str, float]: 6 atmosfer kategorisinin 0-100 ölçeğinde puanları.
        """
        clean_text = text.lower()
        words = re.findall(r"[a-z']+", clean_text)
        n_words = max(1, len(words))

        scores: Dict[str, float] = {}

        for category, lexicon in self.ATMOSPHERE_LEXICONS.items():
            matches = 0
            for w in words:
                clean_w = w.strip("'")
                if clean_w in lexicon:
                    matches += 1
                elif any(clean_w.startswith(stem) and len(clean_w) <= len(stem) + 3 for stem in lexicon if len(stem) >= 4):
                    matches += 0.85

            # Oran hesaplama: Kelime başına tematik anahtar kelime yoğunluğu
            ratio = matches / n_words

            # Doygunluk eğrisi ile 0-100 puanlama: Puan = 100 * (1 - exp(-18 * ratio))
            score_val = round(float(np.clip(100.0 * (1.0 - np.exp(-18.0 * ratio)), 0.0, 100.0)), 1)
            scores[category] = score_val

        return scores

    def get_atmosphere_profile(self, text: str) -> Dict[str, Any]:
        """
        Prosodi ve 6 boyutlu atmosfer skorlarını birleştirip özet edebi profil üretir.
        """
        meter = self.analyze_poetic_meter(text)
        atmosphere = self.analyze_literary_atmosphere(text)

        dominant_cat = max(atmosphere.items(), key=lambda item: item[1])

        return {
            "meter": meter,
            "atmosphere": atmosphere,
            "dominant_atmosphere": dominant_cat[0],
            "dominant_score": dominant_cat[1],
        }
