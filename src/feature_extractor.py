"""
Evrensel Edebi ve Stilometrik Öznitelik Çıkarıcı (Universal Stylometric Feature Extractor)
========================================================================================
Proje: TÜBİTAK - Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası
(Chaucer'dan Modernizm'e Tüm Dönemler ve Kanonik Yazarlar)

Bu modül, Orta Çağ İngilizcesi (Middle English), Rönesans, Aydınlanma, Romantizm,
Viktorya, Amerikan Klasikleri ve Modernizm dönemlerine ait metinlerden kronolojik
ve üslup ayırt edici sayısal öznitelikleri çıkarır.

Öznitelik Kategorileri:
1. Temel Boyut & Sentaktik Metrikler (Cümle boyutu, kelime karakter uzunluğu)
2. Sözcük Zenginliği (TTR, Guiraud R-TTR, Hapax Legomena)
3. Arkaik Kelime ve Zamirler (thou, thee, thy, hath, doth, 'tis, o'er vb.)
4. Morfolojik Ekler (-eth, -est, -ly, -ing, -tion/-sion)
5. Bilinç Akışı İndeksi (Stream of Consciousness: tireler, üç noktalar, cümle varyansı)
6. Şiir vs. Düz Yazı Metrikleri (Satır sonu, dize büyük harfi, diyalog tırnakları)
7. Kanonik İşlevsel Kelimeler (Most Frequent Function Words)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.preprocessor import TextPreprocessor, turkish_lower


class StylometricFeatureExtractor:
    """
    Evrensel Edebi Stilometrik Öznitelik Çıkarıcı Sınıfı.
    
    Orta Çağ'dan Modernizm'e kadar tüm edebi metinleri sözdizimsel, morfolojik,
    sözcüksel ve kronolojik boyutlarda analiz eder.
    """

    # 1. Arkaik İngilizce Zamir, Fiil ve Sözcükleri (Orta Çağ & Rönesans ayrımı)
    DEFAULT_ARCHAIC_WORDS: List[str] = [
        "thou", "thee", "thy", "thine", "ye", "hath", "doth", "art",
        "wilt", "shalt", "didst", "couldst", "wherefore", "whence",
        "whither", "hitherto", "whilst", "'tis", "'twas", "o'er", "thro'",
        "ere", "quoth", "nay", "yea", "forsooth", "anon", "oft"
    ]

    # 2. Standart İngilizce ve Türkçe İşlevsel Kelimeler (Function Words / Stopwords)
    DEFAULT_FUNCTION_WORDS: List[str] = [
        # İngilizce çekirdek bağlaç, edat ve zamirler
        "the", "and", "of", "to", "in", "that", "with", "for", "as", "be",
        "at", "by", "from", "upon", "but", "yet", "though", "whereas",
        "which", "who", "whom", "what", "when", "where", "why", "how",
        "so", "no", "not", "all", "any", "some", "every", "if", "or",
        "then", "there", "now", "here", "very", "too", "it", "his", "her",
        # Türkçe temel bağlaç ve edatlar (geriye dönük uyumluluk)
        "ve", "ile", "de", "da", "ki", "ama", "fakat", "ancak", "çünkü",
        "için", "gibi", "kadar", "daha", "en", "çok", "bu", "şu", "o", "bir"
    ]

    # 3. Noktalama İşaretleri
    DEFAULT_PUNCTUATION_MARKS: List[str] = [
        ",", ";", ".", ":", "!", "?", "...", "-", "—", '"', "'", "(", ")"
    ]

    def __init__(
        self,
        preprocessor: Optional[TextPreprocessor] = None,
        function_words: Optional[Sequence[str]] = None,
        archaic_words: Optional[Sequence[str]] = None,
        punctuation_marks: Optional[Sequence[str]] = None,
    ) -> None:
        """Sınıfı başlatır ve kelime listelerini hazırlar."""
        self.preprocessor = preprocessor or TextPreprocessor()

        fw_src = function_words if function_words is not None else self.DEFAULT_FUNCTION_WORDS
        self.function_words = [turkish_lower(w.strip()) for w in fw_src]

        aw_src = archaic_words if archaic_words is not None else self.DEFAULT_ARCHAIC_WORDS
        self.archaic_words = [turkish_lower(w.strip()) for w in aw_src]

        pm_src = punctuation_marks if punctuation_marks is not None else self.DEFAULT_PUNCTUATION_MARKS
        self.punctuation_marks = list(pm_src)

    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Verilen metinden evrensel edebi ve stilometrik öznitelikleri çıkarır.
        
        Args:
            text (str): Analiz edilecek edebi pasaj.
            
        Returns:
            Dict[str, float]: Öznitelik adı -> sayısal değer eşlemesi.
        """
        features: Dict[str, float] = {}

        if not text or not text.strip():
            return self._fill_empty_features()

        # Tokenizasyon
        raw_lines = text.splitlines()
        sentences = self.preprocessor.tokenize_sentences(text)
        words = self.preprocessor.tokenize_words(text, lower=True, keep_apostrophe=True)
        raw_punctuations = self.preprocessor.extract_punctuations(text)

        total_sentences = len(sentences)
        total_tokens = len(words)
        unique_tokens = len(set(words))
        total_puncts = len(raw_punctuations)

        features["total_sentences"] = float(total_sentences)
        features["total_tokens"] = float(total_tokens)
        features["total_unique_tokens"] = float(unique_tokens)
        features["total_punctuations"] = float(total_puncts)

        if total_tokens == 0:
            return self._fill_empty_features()

        # ----------------------------------------------------------------------
        # 1. Cümle Düzeyi Metrikler
        # ----------------------------------------------------------------------
        sentence_word_counts = []
        for s in sentences:
            s_tokens = self.preprocessor.tokenize_words(s, lower=True, keep_apostrophe=True)
            if s_tokens:
                sentence_word_counts.append(len(s_tokens))

        if sentence_word_counts:
            mean_sent_len = float(np.mean(sentence_word_counts))
            std_sent_len = float(np.std(sentence_word_counts))
            features["avg_sentence_length_words"] = mean_sent_len
            features["std_sentence_length_words"] = std_sent_len
            features["min_sentence_length_words"] = float(np.min(sentence_word_counts))
            features["max_sentence_length_words"] = float(np.max(sentence_word_counts))
            # Varyasyon Katsayısı (CV = std / mean)
            cv_sent = (std_sent_len / mean_sent_len) if mean_sent_len > 0 else 0.0
            features["sentence_length_variance_cv"] = float(cv_sent)
        else:
            mean_sent_len, std_sent_len, cv_sent = 0.0, 0.0, 0.0
            features["avg_sentence_length_words"] = 0.0
            features["std_sentence_length_words"] = 0.0
            features["min_sentence_length_words"] = 0.0
            features["max_sentence_length_words"] = 0.0
            features["sentence_length_variance_cv"] = 0.0

        # ----------------------------------------------------------------------
        # 2. Kelime ve Karakter Düzeyi Metrikler
        # ----------------------------------------------------------------------
        word_lengths = [len(w) for w in words]
        features["avg_word_length_chars"] = float(np.mean(word_lengths))
        features["std_word_length_chars"] = float(np.std(word_lengths))
        features["long_word_ratio"] = float(sum(1 for w in words if len(w) >= 7) / total_tokens)
        features["short_word_ratio"] = float(sum(1 for w in words if len(w) <= 3) / total_tokens)

        # ----------------------------------------------------------------------
        # 3. Sözcük Zenginliği (Lexical Richness)
        # ----------------------------------------------------------------------
        features["ttr"] = float(unique_tokens / total_tokens)
        features["root_ttr"] = float(unique_tokens / math.sqrt(total_tokens))

        word_counts = Counter(words)
        hapax_count = sum(1 for c in word_counts.values() if c == 1)
        features["hapax_legomena_ratio"] = float(hapax_count / total_tokens)

        # ----------------------------------------------------------------------
        # 4. Arkaik Kelime ve Zamir Oranları (Middle English & Renaissance)
        # ----------------------------------------------------------------------
        total_archaic_count = 0
        for aw in self.archaic_words:
            c = word_counts.get(aw, 0)
            total_archaic_count += c
            slug = aw.replace("'", "")
            features[f"archaic_{slug}_ratio"] = float(c / total_tokens)

        features["archaic_total_ratio"] = float(total_archaic_count / total_tokens)

        # ----------------------------------------------------------------------
        # 5. Morfolojik Ek Analizi (Morphological Suffixes)
        # ----------------------------------------------------------------------
        # -eth: arkaik fiil (hath, maketh, doth, sayeth)
        eth_count = sum(1 for w in words if len(w) > 4 and w.endswith("eth"))
        # -est: arkaik fiil (knowest, didst, couldst)
        est_count = sum(1 for w in words if len(w) > 4 and (w.endswith("est") or w.endswith("dst")))
        # -ly: zarf yoğunluğu (Dickens, Austen)
        ly_count = sum(1 for w in words if len(w) > 3 and w.endswith("ly"))
        # -ing: eylem/durum sürekliliği (Woolf, Joyce)
        ing_count = sum(1 for w in words if len(w) > 4 and w.endswith("ing"))
        # -tion / -sion: Aydınlanma soyut isimleri (Swift, Johnson)
        tion_sion_count = sum(1 for w in words if len(w) > 5 and (w.endswith("tion") or w.endswith("sion")))

        features["suffix_eth_ratio"] = float(eth_count / total_tokens)
        features["suffix_est_ratio"] = float(est_count / total_tokens)
        features["suffix_ly_ratio"] = float(ly_count / total_tokens)
        features["suffix_ing_ratio"] = float(ing_count / total_tokens)
        features["suffix_tion_sion_ratio"] = float(tion_sion_count / total_tokens)
        features["suffix_total_ratio"] = float((eth_count + est_count + ly_count + ing_count + tion_sion_count) / total_tokens)

        # ----------------------------------------------------------------------
        # 6. Bilinç Akışı İndeksi (Stream of Consciousness Index - Modernism)
        # ----------------------------------------------------------------------
        dash_count = text.count("—") + text.count("--")
        ellipsis_count = text.count("...") + text.count("…")
        parentheses_count = text.count("(") + text.count(")")
        semicolon_count = text.count(";")

        # Parçalanmış sentaks göstergesi
        fragmented_syntax = (dash_count + ellipsis_count + parentheses_count + semicolon_count) / total_tokens
        # Cümle uzunluğu varyansı ile birleştirilmiş Modernist İndeks
        features["stream_of_consciousness_index"] = float((fragmented_syntax * 10.0) + cv_sent)
        features["dash_frequency_ratio"] = float(dash_count / total_tokens)
        features["ellipsis_frequency_ratio"] = float(ellipsis_count / total_tokens)
        features["parentheses_frequency_ratio"] = float(parentheses_count / total_tokens)

        # 6b. Orwellian Saydamlık ve Politik Dağarcık İndeksi (Window-Pane Clarity vs. Stream of Consciousness)
        political_satirical_words = {
            "political", "party", "state", "leader", "power", "rule", "truth", "lies", "war", "peace",
            "freedom", "slavery", "ignorance", "regime", "propaganda", "rebellion", "comrade", "society",
            "citizen", "control", "speech", "doublethink", "thoughtcrime", "ministry", "brother", "hate",
            "equal", "animal", "pig", "deceit", "liberty", "law", "order", "clocks", "thirteen"
        }
        political_count = sum(1 for w in words if w in political_satirical_words)
        features["political_lexicon_ratio"] = float(political_count / total_tokens)
        clarity_factor = 1.0 / (1.0 + (fragmented_syntax * 8.0) + (features["suffix_ly_ratio"] * 5.0))
        features["clarity_directness_index"] = float(clarity_factor)

        # ----------------------------------------------------------------------
        # 7. Şiir vs. Düz Yazı Metrikleri (Poetic vs. Prose)
        # ----------------------------------------------------------------------
        valid_lines = [line.strip() for line in raw_lines if line.strip()]
        features["poetic_line_break_ratio"] = float(len(valid_lines) / total_tokens) if total_tokens else 0.0

        capitalized_line_starts = sum(1 for line in valid_lines if line and line[0].isupper())
        features["poetic_line_capitalization_ratio"] = float(capitalized_line_starts / len(valid_lines)) if valid_lines else 0.0

        # Diyalog (tırnak) sıklığı
        quote_marks = text.count('"') + text.count("“") + text.count("”") + text.count("«") + text.count("»")
        features["dialogue_quote_ratio"] = float(quote_marks / total_tokens)

        # ----------------------------------------------------------------------
        # 8. Noktalama İşaretleri Kullanım Sıklıkları
        # ----------------------------------------------------------------------
        punct_counter = Counter(raw_punctuations)
        features["punct_total_ratio"] = float(total_puncts / total_tokens)

        for p_mark in self.punctuation_marks:
            c = punct_counter.get(p_mark, 0)
            if p_mark == "...":
                c += punct_counter.get("…", 0)
            elif p_mark in ('"', "'"):
                if p_mark == '"':
                    c += punct_counter.get("“", 0) + punct_counter.get("”", 0)
                elif p_mark == "'":
                    c += punct_counter.get("’", 0) + punct_counter.get("‘", 0)

            slug = self._get_punct_slug(p_mark)
            features[f"punct_{slug}_ratio"] = float(c / total_tokens)

        # ----------------------------------------------------------------------
        # 9. Standart İşlevsel Kelime Oranları (Function Words)
        # ----------------------------------------------------------------------
        total_fw_count = 0
        for fw in self.function_words:
            c = word_counts.get(fw, 0)
            total_fw_count += c
            features[f"fw_{fw}_ratio"] = float(c / total_tokens)

        features["fw_total_ratio"] = float(total_fw_count / total_tokens)

        return features

    def extract_features_series(self, text: str, name: Optional[str] = None) -> pd.Series:
        """Öznitelikleri Pandas Series olarak döndürür."""
        return pd.Series(self.extract_features(text), name=name)

    def extract_features_dataframe(
        self,
        corpus: Dict[str, str] | Sequence[Tuple[str, str]]
    ) -> pd.DataFrame:
        """Çoklu metinler için öznitelik matrisi üretir."""
        records: List[Dict[str, Any]] = []
        items = corpus.items() if isinstance(corpus, dict) else corpus
        for doc_id, text in items:
            row = self.extract_features(text)
            row["doc_id"] = doc_id
            records.append(row)

        df = pd.DataFrame(records)
        if "doc_id" in df.columns:
            df.set_index("doc_id", inplace=True)
        return df

    def _get_punct_slug(self, punct: str) -> str:
        """Noktalama işaretini okunabilir anahtara çevirir."""
        slug_map = {
            ",": "comma",
            ";": "semicolon",
            ".": "period",
            ":": "colon",
            "!": "exclamation",
            "?": "question",
            "...": "ellipsis",
            "…": "ellipsis",
            "-": "hyphen",
            "–": "dash",
            "—": "em_dash",
            '"': "double_quote",
            "'": "single_quote",
            "(": "open_parenthesis",
            ")": "close_parenthesis",
        }
        if punct in slug_map:
            return slug_map[punct]
        return "_".join(f"u{ord(c):04x}" for c in punct)

    def _fill_empty_features(self) -> Dict[str, float]:
        """Boş metin durumunda tüm öznitelikleri sıfırlar."""
        empty: Dict[str, float] = {
            "total_sentences": 0.0,
            "total_tokens": 0.0,
            "total_unique_tokens": 0.0,
            "total_punctuations": 0.0,
            "avg_sentence_length_words": 0.0,
            "std_sentence_length_words": 0.0,
            "min_sentence_length_words": 0.0,
            "max_sentence_length_words": 0.0,
            "sentence_length_variance_cv": 0.0,
            "avg_word_length_chars": 0.0,
            "std_word_length_chars": 0.0,
            "long_word_ratio": 0.0,
            "short_word_ratio": 0.0,
            "ttr": 0.0,
            "root_ttr": 0.0,
            "hapax_legomena_ratio": 0.0,
            "archaic_total_ratio": 0.0,
            "suffix_eth_ratio": 0.0,
            "suffix_est_ratio": 0.0,
            "suffix_ly_ratio": 0.0,
            "suffix_ing_ratio": 0.0,
            "suffix_tion_sion_ratio": 0.0,
            "suffix_total_ratio": 0.0,
            "stream_of_consciousness_index": 0.0,
            "dash_frequency_ratio": 0.0,
            "ellipsis_frequency_ratio": 0.0,
            "parentheses_frequency_ratio": 0.0,
            "poetic_line_break_ratio": 0.0,
            "poetic_line_capitalization_ratio": 0.0,
            "dialogue_quote_ratio": 0.0,
            "political_lexicon_ratio": 0.0,
            "clarity_directness_index": 0.0,
            "punct_total_ratio": 0.0,
            "fw_total_ratio": 0.0,
        }

        for aw in self.archaic_words:
            slug = aw.replace("'", "")
            empty[f"archaic_{slug}_ratio"] = 0.0

        for p_mark in self.punctuation_marks:
            slug = self._get_punct_slug(p_mark)
            empty[f"punct_{slug}_ratio"] = 0.0

        for fw in self.function_words:
            empty[f"fw_{fw}_ratio"] = 0.0

        return empty
