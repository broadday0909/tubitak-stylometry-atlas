"""
Çift Katmanlı Stilometri Sınıflandırıcısı ve Burrows' Delta Motoru (Stylometry Classifier)
========================================================================================
Proje: TÜBİTAK - Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası

Bu modül:
1. Katman 1 (Era Classifier): Edebi dönem tahmini (Renaissance, Victorian, Modernism vb.)
2. Katman 2 (Author Classifier): Global ve dönem kısıtlı yazar tahmini (Top-5 Adaylar)
3. Burrows' Delta Metrik Motoru: Hesaplamalı edebiyatta altın standart Z-skor mesafe analizi
4. StandardScaler + Linear SVM / Random Forest destekli Scikit-Learn Pipeline mimarisi
içerir.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.feature_extractor import StylometricFeatureExtractor

# Uyarı filtrelemeleri
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


class StylometryClassifier:
    """
    Çift Katmanlı (Dönem + Yazar) ve Burrows' Delta Destekli Stilometri Sınıflandırıcısı.
    """

    SUPPORTED_MODELS = ("svm", "rf", "linear_svm", "random_forest")

    def __init__(
        self,
        model_type: str = "svm",
        cv_folds: int = 5,
        random_state: int = 42,
        feature_extractor: Optional[StylometricFeatureExtractor] = None,
        **model_kwargs: Any,
    ) -> None:
        """Sınıflandırıcıyı ilklendirir."""
        self.model_type = model_type.lower()
        if self.model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Desteklenmeyen model türü: '{model_type}'.")

        self.cv_folds = cv_folds
        self.random_state = random_state
        self.model_kwargs = model_kwargs
        self.feature_extractor = feature_extractor or StylometricFeatureExtractor()

        # Katman 1: Dönem Sınıflandırıcısı Pipeline
        self.era_pipeline: Pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", self._init_base_estimator()),
            ]
        )

        # Katman 2: Yazar Sınıflandırıcısı Pipeline
        self.pipeline: Pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", self._init_base_estimator()),
            ]
        )

        self.is_fitted: bool = False
        self.is_era_fitted: bool = False
        self.classes_: np.ndarray = np.array([])
        self.era_classes_: np.ndarray = np.array([])
        self.feature_names_in_: List[str] = []
        self.cv_results_: Dict[str, Any] = {}
        self.era_cv_results_: Dict[str, Any] = {}

        # Burrows' Delta İçin Referans Profilleri
        self.author_profiles_: Dict[str, np.ndarray] = {}
        self.corpus_mean_: np.ndarray = np.array([])
        self.corpus_std_: np.ndarray = np.array([])
        self.author_to_era_map_: Dict[str, str] = {}

    def _init_base_estimator(self) -> BaseEstimator:
        """Algoritma motorunu başlatır."""
        if self.model_type in ("svm", "linear_svm"):
            params = {
                "kernel": "linear",
                "probability": True,
                "C": 1.0,
                "random_state": self.random_state,
            }
            params.update(self.model_kwargs)
            return SVC(**params)
        elif self.model_type in ("rf", "random_forest"):
            params = {
                "n_estimators": 100,
                "max_depth": None,
                "random_state": self.random_state,
            }
            params.update(self.model_kwargs)
            return RandomForestClassifier(**params)
        else:
            raise ValueError(f"Geçersiz model: {self.model_type}")

    # ==========================================================================
    # 1. Hiyerarşik Çift Katmanlı Eğitim (Hierarchical Training)
    # ==========================================================================
    def train_hierarchical(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_author: Union[pd.Series, np.ndarray, Sequence[str]],
        y_era: Union[pd.Series, np.ndarray, Sequence[str]],
    ) -> Dict[str, Any]:
        """
        Hem Dönem (Era) hem de Yazar (Author) modellerini aynı anda eğitir ve
        Burrows' Delta profil matrisini oluşturur.
        
        Args:
            X: Öznitelik matrisi.
            y_author: Yazar etiketleri.
            y_era: Dönem etiketleri.
            
        Returns:
            Dict[str, Any]: Her iki katmanın eğitim ve CV özet sonuçları.
        """
        # Yazar modelini eğit
        author_cv = self.train(X, y_author)

        # Dönem modelini eğit
        y_era_arr = np.array(y_era)
        self.era_classes_ = np.unique(y_era_arr)

        counts = np.bincount(pd.Categorical(y_era_arr).codes)
        min_samples = int(np.min(counts)) if len(counts) > 0 else 1
        effective_cv = max(2, min(self.cv_folds, min_samples))

        cv = StratifiedKFold(n_splits=effective_cv, shuffle=True, random_state=self.random_state)
        scoring = {
            "accuracy": "accuracy",
            "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
        }

        era_cv_scores = cross_validate(self.era_pipeline, X, y_era_arr, cv=cv, scoring=scoring)
        self.era_pipeline.fit(X, y_era_arr)
        self.is_era_fitted = True

        self.era_cv_results_ = {
            "cv_folds": effective_cv,
            "mean_accuracy": float(np.mean(era_cv_scores["test_accuracy"])),
            "std_accuracy": float(np.std(era_cv_scores["test_accuracy"])),
            "mean_f1_macro": float(np.mean(era_cv_scores["test_f1_macro"])),
        }

        # Yazar -> Dönem haritasını kur
        for auth, era in zip(y_author, y_era):
            self.author_to_era_map_[str(auth)] = str(era)

        # Burrows' Delta Z-skor referanslarını hazırla
        self._fit_burrows_delta_profiles(X, y_author)

        return {
            "author_layer": author_cv,
            "era_layer": self.era_cv_results_,
        }

    # ==========================================================================
    # 2. Standart Model Eğitimi (Katman 2)
    # ==========================================================================
    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, Sequence[str]],
    ) -> Dict[str, Any]:
        """Tekil yazar sınıflandırıcısını eğitir ve CV sonuçlarını döner."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            X_mat = X.values
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
            X_mat = np.array(X)

        y_array = np.array(y)
        unique_classes, counts = np.unique(y_array, return_counts=True)
        min_class_count = int(np.min(counts))

        effective_cv = max(2, min(self.cv_folds, min_class_count))
        cv = StratifiedKFold(n_splits=effective_cv, shuffle=True, random_state=self.random_state)

        scoring = {
            "accuracy": "accuracy",
            "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
            "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
            "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
        }

        cv_scores = cross_validate(self.pipeline, X, y_array, cv=cv, scoring=scoring)
        self.pipeline.fit(X, y_array)
        self.is_fitted = True
        self.classes_ = self.pipeline.classes_

        self.cv_results_ = {
            "cv_folds": effective_cv,
            "mean_accuracy": float(np.mean(cv_scores["test_accuracy"])),
            "std_accuracy": float(np.std(cv_scores["test_accuracy"])),
            "mean_f1_macro": float(np.mean(cv_scores["test_f1_macro"])),
            "std_f1_macro": float(np.std(cv_scores["test_f1_macro"])),
            "mean_precision_macro": float(np.mean(cv_scores["test_precision_macro"])),
            "mean_recall_macro": float(np.mean(cv_scores["test_recall_macro"])),
        }

        self._fit_burrows_delta_profiles(X, y)
        return self.cv_results_

    # ==========================================================================
    # 3. Katman 1: Dönem Tahmini (Era Prediction)
    # ==========================================================================
    def predict_era(
        self,
        X: Union[str, pd.DataFrame, pd.Series, np.ndarray, List[str]],
    ) -> Tuple[str, Dict[str, float]]:
        """
        Metnin edebi dönemini ve dönem olasılıklarını tahmin eder.
        
        Returns:
            Tuple[str, Dict[str, float]]: (En olası dönem, {dönem: olasılık})
        """
        if not self.is_era_fitted:
            # Era modeli ayrı eğitilmediyse en olası yazarın dönemini döndür
            author = self.predict(X)
            era = self.author_to_era_map_.get(str(author), "Unknown")
            return era, {era: 1.0}

        X_matrix, _ = self._prepare_input(X)
        pred_era = str(self.era_pipeline.predict(X_matrix)[0])
        probs = self.era_pipeline.predict_proba(X_matrix)[0]
        era_prob_dict = {cls: float(p) for cls, p in zip(self.era_pipeline.classes_, probs)}
        return pred_era, era_prob_dict

    # ==========================================================================
    # 4. Katman 2: Yazar Tahmini & Top-5 Adaylar
    # ==========================================================================
    def predict(
        self,
        X: Union[str, pd.DataFrame, pd.Series, np.ndarray, List[str]],
        use_hierarchical: bool = True,
    ) -> Union[str, np.ndarray]:
        """
        En olası yazar tahminini döner.
        
        Args:
            X: Metin, DataFrame veya özellik vektörü.
            use_hierarchical: Hiyerarşik dönem kısıtlı birleştirme kullanılsın mı.
        """
        self._check_is_fitted()
        X_matrix, is_single = self._prepare_input(X)
        if is_single and use_hierarchical and self.is_era_fitted and self.author_to_era_map_:
            top_candidates = self.predict_top_authors(X, top_k=1, use_hierarchical=True)
            if top_candidates:
                return top_candidates[0][0]
        preds = self.pipeline.predict(X_matrix)
        return str(preds[0]) if is_single else preds

    def predict_proba(
        self,
        X: Union[str, pd.DataFrame, pd.Series, np.ndarray, List[str]],
    ) -> Union[Dict[str, float], pd.DataFrame]:
        """Sınıf olasılıklarını döndürür."""
        self._check_is_fitted()
        X_matrix, is_single = self._prepare_input(X)
        probabilities = self.pipeline.predict_proba(X_matrix)

        if is_single:
            probs = probabilities[0]
            return {cls: float(prob) for cls, prob in zip(self.classes_, probs)}

        return pd.DataFrame(probabilities, columns=self.classes_)

    def predict_top_authors(
        self,
        X: Union[str, pd.DataFrame, pd.Series, np.ndarray],
        era_filter: Optional[str] = None,
        top_k: int = 5,
        use_hierarchical: bool = True,
        hierarchical_weight: float = 3.0,
    ) -> List[Tuple[str, float]]:
        """
        En yüksek olasılığa sahip ilk `top_k` yazar adayını döndürür.
        
        Hiyerarşik Koşullandırma (Hierarchical Conditioning):
        `use_hierarchical=True` olduğunda Katman 1 Dönem olasılıkları P(Era|X),
        ham yazar olasılıkları P(Author|X) ve Burrows' Delta Z-skor yakınlığı (exp(-Delta))
        birleştirilir; döneme ait yazarlar güçlendirilir, anakronik yazarlar sönümlendirilir.
        
        Args:
            X: Girdi metni veya öznitelik matrisi.
            era_filter: Belirli bir döneme zorlamak için dönem adı (None = otomatik).
            top_k: Sıralanacak aday sayısı.
            use_hierarchical: Hiyerarşik koşullu filtrelemenin etkinleştirilmesi.
            hierarchical_weight: Tahmin edilen döneme verilen ağırlık çarpanı.
            
        Returns:
            List[Tuple[str, float]]: [(yazar_adı, normalize_edilmis_olasilik), ...]
        """
        all_probs = self.predict_proba(X)
        if isinstance(all_probs, pd.DataFrame):
            all_probs = all_probs.iloc[0].to_dict()

        if era_filter and self.author_to_era_map_:
            # Yalnızca seçilen dönemdeki yazarları filtrele
            filtered = {
                auth: prob
                for auth, prob in all_probs.items()
                if self.author_to_era_map_.get(auth, "").lower() == era_filter.lower()
            }
            total_filtered = sum(filtered.values())
            if total_filtered > 0:
                all_probs = {k: v / total_filtered for k, v in filtered.items()}
            else:
                all_probs = filtered
        elif use_hierarchical and self.is_era_fitted and self.author_to_era_map_:
            pred_era, era_probs = self.predict_era(X)
            delta_list = self.compute_burrows_delta(X, top_k=len(self.author_profiles_))
            delta_dict = {auth: d for auth, d, _ in delta_list}

            # En olası ilk 2 edebi dönem
            sorted_eras = sorted(era_probs.items(), key=lambda item: item[1], reverse=True)
            top_eras = [e for e, _ in sorted_eras[:2]]

            hierarchical_scores: Dict[str, float] = {}
            for auth, p_raw in all_probs.items():
                auth_era = self.author_to_era_map_.get(auth, "")
                p_era = era_probs.get(auth_era, 1e-4)
                d = delta_dict.get(auth, 1.0)
                delta_sim = float(np.exp(-d))

                if auth_era == pred_era:
                    era_boost = hierarchical_weight
                elif auth_era in top_eras:
                    era_boost = max(1.0, hierarchical_weight * 0.4)
                else:
                    era_boost = 1.0 / (hierarchical_weight ** 1.5)

                score = (p_raw ** 1.0) * (p_era ** 1.2) * delta_sim * era_boost
                hierarchical_scores[auth] = score

            total_score = sum(hierarchical_scores.values())
            if total_score > 0:
                all_probs = {k: v / total_score for k, v in hierarchical_scores.items()}

        sorted_authors = sorted(all_probs.items(), key=lambda item: item[1], reverse=True)
        return sorted_authors[:top_k]

    # ==========================================================================
    # 5. Burrows' Delta Z-Skor Mesafesi Analizi
    # ==========================================================================
    def compute_burrows_delta(
        self,
        X: Union[str, pd.DataFrame, pd.Series, np.ndarray],
        top_k: int = 5,
    ) -> List[Tuple[str, float, float]]:
        """
        John Burrows'un klasik Delta (Z-Score Manhattan Distance) algoritmasını hesaplar.
        
        Formula:
            Delta(D, A) = (1/n) * sum( | (f_i(D) - mu_i)/sigma_i - (f_i(A) - mu_i)/sigma_i | )
            
        Returns:
            List[Tuple[str, float, float]]: [(yazar_adı, delta_mesafesi, benzerlik_yüzdesi), ...]
            (Daha küçük delta mesafesi = daha yüksek benzerlik)
        """
        self._check_is_fitted()
        X_matrix, _ = self._prepare_input(X)
        doc_features = X_matrix.values[0]

        # Dokümanın Z-skoru: (f - mu) / sigma
        eps = 1e-6
        doc_z = (doc_features - self.corpus_mean_) / (self.corpus_std_ + eps)

        delta_scores: List[Tuple[str, float, float]] = []
        for author_name, author_profile in self.author_profiles_.items():
            author_z = (author_profile - self.corpus_mean_) / (self.corpus_std_ + eps)
            # Ortalama mutlak Manhattan mesafesi (Burrows' Delta)
            delta = float(np.mean(np.abs(doc_z - author_z)))
            # Yüzdelik benzerlik skoru: 100 / (1 + delta)
            similarity_pct = float(100.0 / (1.0 + delta))
            delta_scores.append((author_name, round(delta, 4), round(similarity_pct, 2)))

        # En küçük delta mesafesine göre artan sırala
        delta_scores.sort(key=lambda item: item[1])
        return delta_scores[:top_k]

    def _fit_burrows_delta_profiles(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, Sequence[str]],
    ) -> None:
        """Külliyat ortalaması, standart sapması ve yazar profillerini kaydeder."""
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        y_arr = np.array(y)

        self.corpus_mean_ = np.mean(X_mat, axis=0)
        self.corpus_std_ = np.std(X_mat, axis=0)

        self.author_profiles_ = {}
        for author in np.unique(y_arr):
            mask = y_arr == author
            self.author_profiles_[str(author)] = np.mean(X_mat[mask], axis=0)

    # ==========================================================================
    # 6. Değerlendirme ve Öznitelik Önemi
    # ==========================================================================
    def evaluate(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray, Sequence[str]],
    ) -> Dict[str, Any]:
        """Test seti değerlendirme raporu üretir."""
        self._check_is_fitted()
        y_true = np.array(y_test)
        y_pred = self.pipeline.predict(X_test)

        acc = float(accuracy_score(y_true, y_pred))

        # Test setinde bazı sınıflar bulunmasa dahi ValueError fırlatılmasını önle
        unique_labels = self.classes_ if hasattr(self, "classes_") and self.classes_ is not None else np.unique(np.concatenate([y_true, y_pred]))
        labels_list = list(unique_labels)
        target_names_list = [str(x) for x in labels_list]

        report_str = classification_report(
            y_true,
            y_pred,
            labels=labels_list,
            target_names=target_names_list,
            zero_division=0,
        )
        report_dict = classification_report(
            y_true,
            y_pred,
            labels=labels_list,
            target_names=target_names_list,
            output_dict=True,
            zero_division=0,
        )
        conf_matrix = confusion_matrix(y_true, y_pred, labels=labels_list)

        return {
            "accuracy": acc,
            "classification_report_str": report_str,
            "classification_report_dict": report_dict,
            "confusion_matrix": conf_matrix,
            "classes": self.classes_,
        }

    def get_feature_importances(self, top_n: int = 15) -> pd.DataFrame:
        """En ayırt edici öznitelikleri sıralar."""
        self._check_is_fitted()
        classifier_step = self.pipeline.named_steps["classifier"]

        if hasattr(classifier_step, "coef_"):
            coef = classifier_step.coef_
            importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef[0])
            metric_label = "importance (mean_abs_coef)"
        elif hasattr(classifier_step, "feature_importances_"):
            importances = classifier_step.feature_importances_
            metric_label = "importance (gini)"
        else:
            return pd.DataFrame()

        df_importance = pd.DataFrame(
            {
                "feature": self.feature_names_in_,
                "importance": importances,
            }
        ).sort_values(by="importance", ascending=False).reset_index(drop=True)

        df_importance.rename(columns={"importance": metric_label}, inplace=True)
        return df_importance.head(top_n)

    def _prepare_input(
        self,
        X: Union[str, pd.DataFrame, pd.Series, np.ndarray, List[str]],
    ) -> Tuple[pd.DataFrame, bool]:
        """Girdiyi modelin beklediği DataFrame'e dönüştürür."""
        is_single = False
        if isinstance(X, str):
            row = self.feature_extractor.extract_features(X)
            df = pd.DataFrame([row])
            is_single = True
        elif isinstance(X, list) and all(isinstance(item, str) for item in X):
            rows = [self.feature_extractor.extract_features(t) for t in X]
            df = pd.DataFrame(rows)
            is_single = len(X) == 1
        elif isinstance(X, pd.Series):
            df = pd.DataFrame([X.to_dict()])
            is_single = True
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
            is_single = len(df) == 1
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                df = pd.DataFrame([X], columns=self.feature_names_in_ if self.feature_names_in_ else None)
                is_single = True
            else:
                df = pd.DataFrame(X, columns=self.feature_names_in_ if self.feature_names_in_ else None)
                is_single = len(df) == 1
        else:
            raise TypeError(f"Desteklenmeyen girdi türü: {type(X)}")

        if self.feature_names_in_ and set(self.feature_names_in_).issubset(set(df.columns)):
            df = df[self.feature_names_in_]

        return df, is_single

    def _check_is_fitted(self) -> None:
        """Eğitim durumunu denetler."""
        if not self.is_fitted:
            raise RuntimeError("Model henüz eğitilmedi. Lütfen önce `train()` veya `train_hierarchical()` çağırınız.")
