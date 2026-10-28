"""
Derlem ve Veri Seti Oluşturucu Modülü (Corpus & Dataset Builder)
================================================================
Proje: TÜBİTAK - Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası

Bu modül, edebi metinleri stilometrik bloklara böler ve hem Yazar hem de Edebi Dönem
etiketlerini içeren yapılandırılmış (X, y_author, y_era) veri setleri üretir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from src.canon_catalog import CANON_CATALOG
from src.feature_extractor import StylometricFeatureExtractor


def chunk_text_by_words(
    text: str,
    chunk_size: int = 80,
    overlap: int = 35,
    min_chunks: int = 4,
) -> List[str]:
    """
    Metni kelime tabanlı kayan pencere bloklarına böler.
    Her yazarın eğitim ve test setinde yer alabilmesi için min_chunks (varsayılan 4)
    blok üretilmesini garanti eder.
    """
    words = text.split()
    if not words:
        return []

    if len(words) < min_chunks:
        return [text]

    # Dinamik pencere: Metin min_chunks adet standart blok üretmek için kısaysa
    eff_chunk_size = chunk_size
    eff_overlap = overlap

    if len(words) < chunk_size * min_chunks:
        eff_chunk_size = max(20, len(words) // min_chunks)
        eff_overlap = max(5, eff_chunk_size // 2)

    step = max(1, eff_chunk_size - eff_overlap)
    chunks: List[str] = []

    for i in range(0, len(words), step):
        chunk_words = words[i : i + eff_chunk_size]
        if len(chunk_words) < max(12, eff_chunk_size // 3) and chunks:
            break
        chunks.append(" ".join(chunk_words))

    # Hâlâ min_chunks'tan az blok varsa metni eşit dilimlere böl
    if len(chunks) < min_chunks:
        seg_len = len(words) // min_chunks
        chunks = []
        for c_idx in range(min_chunks):
            start_i = c_idx * seg_len
            end_i = (c_idx + 1) * seg_len if c_idx < min_chunks - 1 else len(words)
            chunks.append(" ".join(words[start_i:end_i]))

    return chunks


def build_stylometry_dataset(
    corpus_dict: Dict[str, Union[str, Tuple[str, str]]],
    extractor: Optional[StylometricFeatureExtractor] = None,
    chunk_size: int = 100,
    overlap: int = 30,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Geriye dönük uyumlu (X, y_author) veri kümesi üretir.
    corpus_dict: {author: text} veya {author: (era, text)}
    """
    feat_extractor = extractor or StylometricFeatureExtractor()
    feature_rows: List[Dict[str, float]] = []
    author_labels: List[str] = []

    for author_name, val in corpus_dict.items():
        raw_text = val[1] if isinstance(val, tuple) else val
        text_chunks = chunk_text_by_words(raw_text, chunk_size=chunk_size, overlap=overlap)
        for idx, chunk in enumerate(text_chunks):
            features = feat_extractor.extract_features(chunk)
            feature_rows.append(features)
            author_labels.append(author_name)

    df_X = pd.DataFrame(feature_rows)
    s_y = pd.Series(author_labels, name="author")

    row_ids = [f"{author_labels[i]}_chunk_{i}" for i in range(len(author_labels))]
    df_X.index = pd.Index(row_ids, name="sample_id")
    s_y.index = df_X.index

    return df_X, s_y


def build_hierarchical_stylometry_dataset(
    corpus_dict: Dict[str, Tuple[str, str]],
    extractor: Optional[StylometricFeatureExtractor] = None,
    chunk_size: int = 100,
    overlap: int = 30,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Çift katmanlı sınıflandırma için (X, y_author, y_era) veri setini üretir.
    
    Args:
        corpus_dict (Dict[str, Tuple[str, str]]): {author_name: (era_name, raw_text)}
        
    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.Series]: (df_X, s_y_author, s_y_era)
    """
    feat_extractor = extractor or StylometricFeatureExtractor()
    feature_rows: List[Dict[str, float]] = []
    author_labels: List[str] = []
    era_labels: List[str] = []

    for author_name, (era_name, raw_text) in corpus_dict.items():
        text_chunks = chunk_text_by_words(raw_text, chunk_size=chunk_size, overlap=overlap)
        for idx, chunk in enumerate(text_chunks):
            features = feat_extractor.extract_features(chunk)
            feature_rows.append(features)
            author_labels.append(author_name)
            era_labels.append(era_name)

    df_X = pd.DataFrame(feature_rows)
    s_y_author = pd.Series(author_labels, name="author")
    s_y_era = pd.Series(era_labels, name="era")

    row_ids = [f"{author_labels[i]}_chunk_{i}" for i in range(len(author_labels))]
    df_X.index = pd.Index(row_ids, name="sample_id")
    s_y_author.index = df_X.index
    s_y_era.index = df_X.index

    return df_X, s_y_author, s_y_era
