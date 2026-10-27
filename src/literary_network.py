"""
TÜBİTAK Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası
Modül: Edebi Soy Ağacı ve Yazarlar Arası Etki Ağı (Literary Phylogeny & Network)
Developed by Bayram Akpamukçu
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

from src.canon_catalog import CANON_CATALOG, AuthorEntry


class LiteraryPhylogenyEngine:
    """
    37 kanonik yazarın üslup mesafelerini (pairwise distance matrix),
    hiyerarşik edebi soy ağacını (Phylogenetic Dendrogram) ve
    yüzyıllar arası üslup ikizlerini (Cross-Era Stylistic Soulmates)
    hesaplayan adli ve hesaplamalı edebiyat motoru.

    Developed by Bayram Akpamukçu
    """

    def __init__(
        self,
        X_corpus: Optional[pd.DataFrame] = None,
        y_authors: Optional[Union[List[str], pd.Series]] = None,
        y_eras: Optional[Union[List[str], pd.Series]] = None,
        metric: str = "cosine",
    ):
        self.metric = metric
        self.distance_df_: Optional[pd.DataFrame] = None
        self.authors_: List[str] = []
        self.eras_map_: Dict[str, str] = {}
        self.centroids_: Optional[pd.DataFrame] = None
        self.linkage_matrix_: Optional[np.ndarray] = None

        # Yazar -> Dönem haritasını CANON_CATALOG'dan önceden yükle
        for entry in CANON_CATALOG.values():
            self.eras_map_[entry.name] = entry.era

        if X_corpus is not None and y_authors is not None:
            self.compute_author_distance_matrix(X_corpus, y_authors, y_eras=y_eras, metric=metric)

    def compute_author_distance_matrix(
        self,
        X_corpus: pd.DataFrame,
        y_authors: Union[List[str], pd.Series],
        y_eras: Optional[Union[List[str], pd.Series]] = None,
        metric: str = "cosine",
    ) -> pd.DataFrame:
        """
        Külliyattaki 70+ stilometrik öznitelik matrisini kullanarak her yazarın
        merkez noktasını (centroid) çıkarır ve tüm yazarlar arasındaki çift yönlü
        (pairwise) mesafe matrisini hesaplar.
        
        Args:
            X_corpus (pd.DataFrame): Stilometri öznitelik matrisi.
            y_authors: Her satıra karşılık gelen yazar etiketi.
            y_eras: Opsiyonel edebi dönem etiketleri.
            metric (str): Mesafe metriği ('cosine', 'burrows_delta' / 'cityblock', 'euclidean').
            
        Returns:
            pd.DataFrame: 37x37 simetrik ve köşegeni 0 olan mesafe matrisi.
        """
        self.metric = metric
        df_merged = X_corpus.copy()
        df_merged["__Author__"] = list(y_authors)

        if y_eras is not None:
            for auth, era in zip(list(y_authors), list(y_eras)):
                self.eras_map_[str(auth)] = str(era)

        # 1. Yazar merkez noktaları (centroids)
        centroids_df = df_merged.groupby("__Author__").mean()
        self.authors_ = list(centroids_df.index)
        self.centroids_ = centroids_df

        # 2. Öznitelikleri standartlaştır (Z-Score ölçekleme - Burrows standardı)
        scaler = StandardScaler()
        centroids_scaled = scaler.fit_transform(centroids_df.values)

        # 3. Çift yönlü mesafeleri hesapla
        if metric in ("burrows_delta", "cityblock", "manhattan"):
            # Burrows' Delta = ortalama mutlak Z-skor farkı
            condensed_dists = pdist(centroids_scaled, metric="cityblock") / centroids_scaled.shape[1]
        elif metric == "cosine":
            condensed_dists = pdist(centroids_scaled, metric="cosine")
        else:
            condensed_dists = pdist(centroids_scaled, metric=metric)

        # 4. Kare matrise dönüştür ve simetriyi garanti et
        dist_matrix = squareform(condensed_dists)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
        np.fill_diagonal(dist_matrix, 0.0)

        self.distance_df_ = pd.DataFrame(
            dist_matrix,
            index=self.authors_,
            columns=self.authors_,
        )
        return self.distance_df_

    def get_stylistic_soulmates(
        self,
        author_name: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Seçilen bir yazarın tüm edebiyat tarihindeki en yakın 'üslup ikizlerini'
        (hem kendi döneminden hem de farklı yüzyıllardan) mesafe skorlarıyla listeler.
        
        Args:
            author_name (str): İncelenecek yazar adı.
            top_k (int): Döndürülecek en yakın yazar sayısı.
            
        Returns:
            List[Dict[str, Any]]: En yakın yazarlar, mesafe, benzerlik yüzdesi ve edebi analiz.
        """
        if self.distance_df_ is None:
            raise ValueError("Mesafe matrisi henüz hesaplanmadı! Önce compute_author_distance_matrix çağrılmalıdır.")

        if author_name not in self.distance_df_.index:
            # Büyük/küçük harf duyarsız arama
            matched = [a for a in self.distance_df_.index if a.lower() == author_name.lower()]
            if matched:
                author_name = matched[0]
            else:
                raise ValueError(f"Yazar bulunamadı: '{author_name}'. Mevcut yazarlar: {len(self.authors_)} yazar.")

        row = self.distance_df_.loc[author_name].drop(index=author_name)
        sorted_soulmates = row.sort_values().head(top_k)

        author_era = self.eras_map_.get(author_name, "Bilinmeyen Dönem")
        author_entry = next((e for e in CANON_CATALOG.values() if e.name == author_name), None)

        soulmates_list: List[Dict[str, Any]] = []

        for soul_author, dist_val in sorted_soulmates.items():
            dist_float = float(dist_val)
            soul_era = self.eras_map_.get(soul_author, "Bilinmeyen Dönem")
            soul_entry = next((e for e in CANON_CATALOG.values() if e.name == soul_author), None)

            # Benzerlik yüzdesi
            if self.metric == "cosine":
                sim_pct = round(float(np.clip((1.0 - (dist_float / 2.0)) * 100, 15.0, 99.4)), 1)
            else:
                sim_pct = round(float(np.clip(100.0 / (1.0 + dist_float), 15.0, 99.4)), 1)

            is_same_era = (author_era == soul_era)
            rel_type = (
                "Dönemdaş Üslup Paralelliği (Contemporary Parallel)"
                if is_same_era
                else f"Tarihsel Köprü & Etki Ağı ({author_era} ➔ {soul_era})"
            )

            # Edebi stilistik içgörü
            insight = self._generate_stylistic_insight(author_name, str(soul_author), is_same_era, author_entry, soul_entry)

            soulmates_list.append({
                "author": str(soul_author),
                "distance": round(dist_float, 4),
                "similarity_pct": sim_pct,
                "era": soul_era,
                "canonical_work": soul_entry.canonical_work if soul_entry else "Kanonik Eser",
                "relationship_type": rel_type,
                "stylistic_insight": insight,
            })

        return soulmates_list

    def compute_linkage_matrix(
        self,
        distance_matrix: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        method: str = "average",
    ) -> np.ndarray:
        """
        Dendrogram çizimi için hiyerarşik kümeleme (Hierarchical Clustering) linkage matrisini hesaplar.
        
        Args:
            distance_matrix: Opsiyonel mesafe matrisi (None ise dahili matris kullanılır).
            method (str): Bağlantı metodu ('average', 'ward', 'complete', 'single').
            
        Returns:
            np.ndarray: (N-1) x 4 boyutlu linkage matrisi.
        """
        if distance_matrix is None:
            if self.distance_df_ is None:
                raise ValueError("Mesafe matrisi bulunamadı. Önce compute_author_distance_matrix çağrılmalıdır.")
            dm = self.distance_df_.values.copy()
        elif isinstance(distance_matrix, pd.DataFrame):
            dm = distance_matrix.values.copy()
        else:
            dm = np.array(distance_matrix, dtype=float).copy()

        # Simetri ve sıfır köşegen sağlama
        dm = (dm + dm.T) / 2.0
        np.fill_diagonal(dm, 0.0)

        # Ward metodu Öklid mesafesi gerektirir, cosine/delta için average veya complete idealdir
        if method == "ward" and self.metric == "cosine":
            method = "average"

        condensed = squareform(dm, force="tovector", checks=False)
        self.linkage_matrix_ = linkage(condensed, method=method)
        return self.linkage_matrix_

    def _generate_stylistic_insight(
        self,
        author_a: str,
        author_b: str,
        is_same_era: bool,
        entry_a: Optional[AuthorEntry],
        entry_b: Optional[AuthorEntry],
    ) -> str:
        """İki yazar arasındaki üslup benzerliğinin edebi ve morfolojik gerekçesini açıklar."""
        pair_key = tuple(sorted([author_a, author_b]))

        # Kanonik bilinen özel ilişkiler
        special_insights = {
            ("Christopher Marlowe", "William Shakespeare"): "Rönesans blank verse (ölçülü uyaksız şiir), trajik retorik ve zengin morfolojik fiil çekimi ortaklığı.",
            ("George Orwell", "Jonathan Swift"): "Yalın, doğrudan ve keskin alegorik hiciv (political allegory & satirical clarity) akrabalığı.",
            ("Charles Dickens", "George Eliot"): "19. yüzyıl Viktorya dönemi derin psikolojik karakter tasvirleri ve geniş cümle sözdizimi.",
            ("James Joyce", "Virginia Woolf"): "Modernist bilinç akışı (stream-of-consciousness), noktalama esnekliği ve yüksek sözcüksel çeşitlilik (TTR).",
            ("Edgar Allan Poe", "Herman Melville"): "Amerikan karanlık romantizmi, melankoli, felsefi derinlik ve gotik atmosfer.",
            ("Ernest Hemingway", "George Orwell"): "Yalın, dolaysız, sıfat ve süslemelerden arındırılmış ekonomik sözdizimi.",
            ("John Donne", "John Milton"): "17. yüzyıl metafizik imgelem, yoğun teolojik sözcük dağarcığı ve arkaik sentaks.",
            ("Jane Austen", "Mary Shelley"): "Romantizm dönemi anlatı dengesi, betimleyici zarf (-ly) sıklığı ve biçimsel zarafet.",
        }

        if pair_key in special_insights:
            return special_insights[pair_key]

        if is_same_era:
            return f"Aynı dönemin ({entry_a.era if entry_a else 'Ortak Dönem'}) sözdizimsel kalıpları, benzer cümle ritmi ve morfolojik ek dağılımı."
        else:
            era_a = entry_a.era if entry_a else "1. Dönem"
            era_b = entry_b.era if entry_b else "2. Dönem"
            return f"Yüzyıllar ötesi üslup akrabalığı: {era_a} ile {era_b} arasında leksikal yoğunluk ve sözcük çeşitliliği paralelliği."
