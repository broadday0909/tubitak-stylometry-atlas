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
    chunk_size: int = 100,
    overlap: int = 30,
) -> List[str]:
    """Metni kelime tabanlı kayan pencere bloklarına böler."""
    words = text.split()
    if not words:
        return []

    if len(words) <= chunk_size:
        return [text]

    step = max(1, chunk_size - overlap)
    chunks: List[str] = []

    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        if len(chunk_words) < max(20, chunk_size // 3) and chunks:
            break
        chunks.append(" ".join(chunk_words))

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
