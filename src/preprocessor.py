"""
Metin Ön İşleme ve Tokenizasyon Modülü (Text Preprocessor)
==========================================================
Proje: TÜBİTAK - Stilometri ile Edebi Eserlerde Yazar Tespiti ve Üslup Analizi

Bu modül, Türkçe edebi metinlerin ön işleme, temizleme ve tokenizasyon (cümle, kelime,
noktalama) adımlarını gerçekleştiren sınıfları ve fonksiyonları içerir.
Özellikle Türkçe karakter duyarlılığı (I/ı, İ/i) ve edebi metinlerin noktalama
çeşitliliği göz önünde bulundurularak tasarlanmıştır.
"""

from __future__ import annotations

import re
import string
from typing import List, Tuple, Pattern


def turkish_lower(text: str) -> str:
    """
    Türkçe karakter kurallarına uygun biçimde metni küçük harfe dönüştürür.
    
    Standart Python `str.lower()` metodu Türkçe 'I' harfini 'i' veya noktalı 'i'
    yaparak bozulmalara yol açabilir. Bu fonksiyon 'I' -> 'ı' ve 'İ' -> 'i'
    dönüşümlerini güvenli şekilde gerçekleştirir.
    
    Args:
        text (str): Küçük harfe dönüştürülecek kaynak metin.
        
    Returns:
        str: Türkçe kurallarına uygun küçük harfli metin.
    """
    char_map = {
        "I": "ı",
        "İ": "i",
    }
    for upper_char, lower_char in char_map.items():
        text = text.replace(upper_char, lower_char)
    return text.lower()


class TextPreprocessor:
    """
    Türkçe edebi metinler için gelişmiş ön işleme ve tokenizasyon sınıfı.
    
    Attributes:
        abbreviations (set[str]): Cümle bölme aşamasında yanlış bölünmeyi
            önlemek için kullanılan yaygın Türkçe kısaltmalar.
        sentence_regex (Pattern): Cümle sınırlarını belirleyen derlenmiş düzenli ifade.
        word_regex (Pattern): Türkçe karakterleri de içeren kelime yakalama ifadesi.
    """

    DEFAULT_ABBREVIATIONS = {
        # Türkçe kısaltmalar
        "dr", "prof", "doç", "yrd", "av", "sf", "s", "mad", "vb", "vs", "bkz",
        "cad", "sok", "mah", "apt", "no", "alb", "gen", "yrb", "binb", "yzb",
        # İngilizce kanonik kısaltmalar
        "mr", "mrs", "ms", "dr", "prof", "rev", "capt", "col", "gen", "sir",
        "lord", "lady", "st", "etc", "viz", "ie", "eg", "al", "vol", "no", "chap"
    }

    # İngilizce arkaik kesmeler ('tis, 'twas, o'er, thro') ve Türkçe karakterleri içeren kelime deseni
    WORD_PATTERN = r"(?:'tis|'twas|o'er|thro'|[a-zA-ZçğıöşüÇĞİÖŞÜâîûÂÎÛ0-9]+(?:'[a-zA-ZçğıöşüÇĞİÖŞÜâîûÂÎÛ0-9]+)?)"
    
    # Noktalama işaretleri (üç nokta, tireler ve özel tırnak işaretleri dahil)
    PUNCT_PATTERN = r"(?:\.\.\.|[\.,;:!\?\-\–\—'\"“”‘’\(\)\[\]\{\}\/\\«»…])"

    def __init__(self, custom_abbreviations: set[str] | None = None) -> None:
        """
        TextPreprocessor sınıfını başlatır.
        
        Args:
            custom_abbreviations (set[str] | None): Cümle bölücünün dikkate alacağı
                özel kısaltma kümesi. Belirtilmezse varsayılanlar kullanılır.
        """
        self.abbreviations = set(self.DEFAULT_ABBREVIATIONS)
        if custom_abbreviations:
            self.abbreviations.update({turkish_lower(a).rstrip(".") for a in custom_abbreviations})

        self.word_regex: Pattern = re.compile(self.WORD_PATTERN, flags=re.UNICODE)
        self.punct_regex: Pattern = re.compile(self.PUNCT_PATTERN, flags=re.UNICODE)

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Metindeki gereksiz boşlukları, satır sonlarını ve kontrol karakterlerini normalize eder.
        
        Args:
            text (str): Temizlenecek ham metin.
            
        Returns:
            str: Normalize edilmiş metin.
        """
        if not text:
            return ""
        # Satır sonlarını boşluğa çevir ve birden fazla boşluğu tek boşluğa indir
        normalized = re.sub(r"[\r\n\t]+", " ", text)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Metni mantıksal cümlelerine ayırır (Sentence Tokenization).
        
        Türkçe edebi metinlerde sıkça görülen diyaloglar, üç noktalar (...),
        ünlem-soru kombinasyonları (!?, ?!) ve kısaltmaları gözeterek bölme yapar.
        
        Args:
            text (str): Cümlelerine ayrılacak metin.
            
        Returns:
            List[str]: Ayrıştırılmış cümlelerin listesi.
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            return []

        # Kısaltmaların noktalarını geçici bir özel sembol ile koru (Örn: Prof. Dr. -> Prof<PRD> Dr<PRD>)
        protected = cleaned
        for abbr in self.abbreviations:
            # Büyük/küçük harf duyarsız eşleştirme ile abbr + '.' sonrasını koru
            pattern = rf"\b({abbr})\."
            protected = re.sub(pattern, r"\1__DOT__", protected, flags=re.IGNORECASE)

        # Cümle sonlandırıcı işaretleri belirle: '.', '!', '?', '...'
        potential_sentences = re.split(r"(?<=[.!?…])\s+(?=[A-ZÇĞİÖŞÜ\"“'«])", protected)

        refined_sentences: List[str] = []
        for sent in potential_sentences:
            sent = sent.strip()
            if sent:
                # Korunan noktaları geri yükle
                restored = sent.replace("__DOT__", ".")
                refined_sentences.append(restored)

        return refined_sentences

    def tokenize_words(
        self,
        text: str,
        lower: bool = True,
        keep_apostrophe: bool = False
    ) -> List[str]:
        """
        Metni kelime tokenlarına ayırır (Word Tokenization).
        Noktalama işaretlerini kelimelerden arındırır.
        
        Args:
            text (str): Tokenize edilecek metin veya cümle.
            lower (bool): Kelimelerin küçük harfe çevrilip çevrilmeyeceği (Varsayılan: True).
            keep_apostrophe (bool): Kesme işaretli eklerin korunup korunmayacağı.
            
        Returns:
            List[str]: Kelime tokenları listesi.
        """
        if not text:
            return []

        target_text = turkish_lower(text) if lower else text
        tokens = self.word_regex.findall(target_text)

        if not keep_apostrophe:
            tokens = [tok.split("'")[0] for tok in tokens if tok]

        return [tok for tok in tokens if tok]

    def extract_punctuations(self, text: str) -> List[str]:
        """
        Metin içerisindeki tüm noktalama işaretlerini sırasıyla yakalar.
        Üç nokta (...) tek bir sembol olarak gruplanır.
        
        Args:
            text (str): İncelenecek metin.
            
        Returns:
            List[str]: Yakalanan noktalama işaretleri listesi.
        """
        if not text:
            return []
        return self.punct_regex.findall(text)

    def extract_characters(self, text: str, only_alphanumeric: bool = False) -> List[str]:
        """
        Metindeki karakterleri ayrıştırır (ortalama kelime boyutu veya karakter frekansı için).
        
        Args:
            text (str): İncelenecek metin.
            only_alphanumeric (bool): Sadece harf ve rakamların alınıp alınmayacağı.
            
        Returns:
            List[str]: Karakter listesi.
        """
        if only_alphanumeric:
            return [ch for ch in text if ch.isalnum()]
        return [ch for ch in text if not ch.isspace()]
