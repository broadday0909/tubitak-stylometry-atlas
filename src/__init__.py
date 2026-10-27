"""
TÜBİTAK Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası Çekirdek Paketi
===========================================================================
"""

from src.canon_catalog import CANON_CATALOG, LITERARY_ERAS, AuthorEntry, get_all_eras, get_authors_by_era
from src.classifier import StylometryClassifier
from src.corpus_builder import (
    build_hierarchical_stylometry_dataset,
    build_stylometry_dataset,
    chunk_text_by_words,
)
from src.feature_extractor import StylometricFeatureExtractor
from src.gutenberg_fetcher import clean_gutenberg_text, download_gutenberg_text, load_universal_canon_corpus
from src.preprocessor import TextPreprocessor, turkish_lower
from src.visualizer import (
    plot_author_stylistic_comparison,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_probability_bars,
)

__all__ = [
    "CANON_CATALOG",
    "LITERARY_ERAS",
    "AuthorEntry",
    "get_all_eras",
    "get_authors_by_era",
    "clean_gutenberg_text",
    "download_gutenberg_text",
    "load_universal_canon_corpus",
    "TextPreprocessor",
    "turkish_lower",
    "StylometricFeatureExtractor",
    "StylometryClassifier",
    "chunk_text_by_words",
    "build_stylometry_dataset",
    "build_hierarchical_stylometry_dataset",
    "plot_confusion_matrix",
    "plot_feature_importance",
    "plot_probability_bars",
    "plot_author_stylistic_comparison",
]
