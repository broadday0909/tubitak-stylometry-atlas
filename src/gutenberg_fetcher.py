"""
Project Gutenberg İndirici ve Külliyat Yöneticisi (Gutenberg Fetcher)
====================================================================
Proje: TÜBİTAK - Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası

Bu modül, çevrimiçi ortamda Project Gutenberg arşivinden kanonik eserlerin tam metinlerini
indirip Gutenberg lisans/başlık kısımlarını temizleyerek `data/raw/` dizinine kaydeder.
İnternet bağlantısı olmadığı (çevrimdışı) durumlarda ise `canon_catalog.py` içerisindeki
zengin yerleşik pasajları (built-in corpus) otomatik olarak devreye sokar.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.canon_catalog import CANON_CATALOG, AuthorEntry


def clean_gutenberg_text(text: str) -> str:
    """
    Project Gutenberg başlık, lisans ve yasal uyarı metinlerini ayıklayarak
    yalnızca eserin asıl edebi içeriğini bırakır.
    
    Args:
        text (str): Gutenberg'den indirilen ham metin.
        
    Returns:
        str: Temizlenmiş edebi metin.
    """
    if not text:
        return ""

    # Başlangıç işaretçisi desenleri
    start_patterns = [
        r"\*\*\*\s*START OF TH(?:IS|E) PROJECT GUTENBERG EBOOK[^\*]*\*\*\*",
        r"\*\*\*\s*START OF THIS PROJECT GUTENBERG[^\*]*\*\*\*",
        r"START OF THE PROJECT GUTENBERG EBOOK",
    ]
    # Bitiş işaretçisi desenleri
    end_patterns = [
        r"\*\*\*\s*END OF TH(?:IS|E) PROJECT GUTENBERG EBOOK[^\*]*\*\*\*",
        r"\*\*\*\s*END OF THIS PROJECT GUTENBERG[^\*]*\*\*\*",
        r"END OF THE PROJECT GUTENBERG EBOOK",
    ]

    cleaned = text
    for sp in start_patterns:
        match = re.search(sp, cleaned, flags=re.IGNORECASE)
        if match:
            cleaned = cleaned[match.end():]
            break

    for ep in end_patterns:
        match = re.search(ep, cleaned, flags=re.IGNORECASE)
        if match:
            cleaned = cleaned[:match.start()]
            break

    return cleaned.strip()


def download_gutenberg_text(book_id: int, timeout_sec: int = 6) -> Optional[str]:
    """
    Belirtilen Gutenberg kitap ID'sini web üzerinden indirmeyi dener.
    
    Args:
        book_id (int): Project Gutenberg eser kimlik numarası.
        timeout_sec (int): İstek zaman aşımı süresi (saniye).
        
    Returns:
        Optional[str]: Temizlenmiş metin içeriği veya başarısız olursa None.
    """
    candidate_urls = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TubitakStylometryResearch/2.0"}

    for url in candidate_urls:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                raw_bytes = resp.read()
                # Encoding denemeleri
                for enc in ("utf-8", "latin-1", "cp1252"):
                    try:
                        raw_text = raw_bytes.decode(enc)
                        cleaned = clean_gutenberg_text(raw_text)
                        if len(cleaned.split()) >= 100:
                            return cleaned
                    except UnicodeDecodeError:
                        continue
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            continue

    return None


def sync_author_corpus(
    data_dir: Path,
    author_entry: AuthorEntry,
    try_download: bool = False,
    max_words: int = 5000,
) -> Path:
    """
    Yazar için yerel diskte dosya oluşturur veya senkronize eder.
    Varsayılan olarak çevrimdışı yerleşik pasajları kullanır, istenirse Gutenberg'i dener.
    
    Args:
        data_dir (Path): Ham metinlerin kaydedileceği dizin.
        author_entry (AuthorEntry): Yazar künye kaydı.
        try_download (bool): İnternetten indirme denenip denenmeyeceği.
        max_words (int): İndirilen metinden alınacak maksimum kelime sayısı.
        
    Returns:
        Path: Oluşturulan veya var olan yerel dosya yolu.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{author_entry.era.lower().replace(' ', '_').replace('&', 'and')}__{author_entry.author_id}.txt"
    file_path = data_dir / slug

    catalog_content = "\n\n".join(author_entry.sample_passages)

    # Dosya zaten varsa ve katalog içeriğinden küçük değilse yeniden oluşturma
    if file_path.exists() and file_path.stat().st_size >= len(catalog_content.encode("utf-8")) - 10:
        return file_path

    content = ""
    # Çevrimiçi indirme talep edildiyse dene
    if try_download:
        online_text = download_gutenberg_text(author_entry.gutenberg_id, timeout_sec=4)
        if online_text:
            words = online_text.split()[:max_words]
            content = " ".join(words)

    # Çevrimiçi alınamadıysa yerleşik edebi pasajları birleştir
    if not content:
        content = catalog_content

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return file_path


def load_universal_canon_corpus(
    data_dir: Path,
    try_download: bool = False,
    selected_eras: Optional[List[str]] = None,
) -> Dict[str, Tuple[str, str]]:
    """
    Tüm kanonik yazarların metinlerini hazırlar ve döndürür.
    
    Returns:
        Dict[str, Tuple[str, str]]: {author_name: (era, full_text)} sözlüğü.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    corpus: Dict[str, Tuple[str, str]] = {}

    for author_id, entry in CANON_CATALOG.items():
        if selected_eras and entry.era not in selected_eras:
            continue

        file_path = sync_author_corpus(data_dir, entry, try_download=try_download)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    corpus[entry.name] = (entry.era, text)
        except Exception:
            # Fallback to direct memory
            text = "\n\n".join(entry.sample_passages)
            corpus[entry.name] = (entry.era, text)

    return corpus
