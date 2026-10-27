"""
TÜBİTAK Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası
Modül: Üslup Dönüşüm Reçetesi ve Yazar Koçu (Stylometric Prescription & Coach)
Developed by Bayram Akpamukçu
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.canon_catalog import CANON_CATALOG, AuthorEntry
from src.feature_extractor import StylometricFeatureExtractor


class StylometricCoach:
    """
    Kullanıcının metnini seçtiği hedef kanonik yazarın (Shakespeare, Orwell, Austen vb.)
    stilometrik profiliyle Z-skoru düzeyinde karşılaştıran ve hedef üsluba yaklaşmak için
    matematiksel, pedagojik ve somut reçeteler üreten yazar koçluğu motoru.

    Developed by Bayram Akpamukçu
    """

    FEATURE_FRIENDLY_NAMES: Dict[str, str] = {
        "avg_sentence_length_words": "Ortalama Cümle Uzunluğu (Kelime)",
        "avg_word_length_chars": "Ortalama Kelime Uzunluğu (Karakter)",
        "ttr": "Sözcük Çeşitliliği (Type-Token Ratio)",
        "root_ttr": "Kök Sözcük Zenginliği (Guiraud R)",
        "hapax_legomena_ratio": "Hapax Oranı (Tekil / Nadir Kelimeler)",
        "long_word_ratio": "Uzun Kelime Oranı (≥7 Harf)",
        "short_word_ratio": "Kısa Kelime Oranı (≤3 Harf)",
        "suffix_ly_ratio": "-ly Zarf Yoğunluğu",
        "suffix_ing_ratio": "-ing Süreklilik Oranı",
        "suffix_eth_ratio": "-eth Arkaik Fiil Çekimi",
        "suffix_est_ratio": "-est Arkaik Fiil Çekimi",
        "suffix_tion_sion_ratio": "-tion/-sion Soyut İsim Ekleri",
        "suffix_total_ratio": "Toplam Morfolojik Ek Oranı",
        "archaic_total_ratio": "Toplam Arkaik Sözcük Oranı",
        "punct_semicolon_ratio": "Noktalı Virgül (;) Sıklığı",
        "punct_comma_ratio": "Virgül (,) Sıklığı",
        "punct_question_ratio": "Soru İşareti (?) Sıklığı",
        "punct_exclamation_ratio": "Ünlem İşareti (!) Sıklığı",
        "punct_total_ratio": "Toplam Noktalama İşareti Oranı",
        "dialogue_quote_ratio": "Doğrudan Diyalog (Tırnak) Yoğunluğu",
        "stream_of_consciousness_index": "Bilinç Akışı İndeksi (Modernizm)",
        "sentence_length_variance_cv": "Cümle Boyutu Varyasyon Katsayısı (CV)",
        "fw_total_ratio": "Toplam İşlevsel Kelime Oranı",
    }

    def __init__(self, classifier: Any, extractor: Optional[StylometricFeatureExtractor] = None):
        self.classifier = classifier
        self.extractor = extractor or StylometricFeatureExtractor()

    def compare_with_target_author(
        self,
        user_text: str,
        target_author: str,
    ) -> Dict[str, Any]:
        """
        Kullanıcının metin vektörü ile hedef yazarın ortalama profil vektörü (centroid)
        arasındaki Z-skor sapmalarını hesaplar ve en büyük 5 stilometrik farkı çıkarır.
        
        Args:
            user_text (str): Kullanıcının yazdığı metin.
            target_author (str): Hedef yazar adı (Örn: 'William Shakespeare').
            
        Returns:
            Dict[str, Any]: Uyum yüzdesi, hedef yazar adı, en büyük 5 sapma listesi.
        """
        user_features = self.extractor.extract_features(user_text)

        # Hedef yazar profilini sınıflandırıcıdan al
        profiles = getattr(self.classifier, "author_profiles_", {})
        feature_names = getattr(
            self.classifier, "feature_names_in_", list(user_features.keys())
        )
        corpus_mean = getattr(
            self.classifier, "corpus_mean_", np.zeros(len(feature_names))
        )
        corpus_std = getattr(
            self.classifier, "corpus_std_", np.ones(len(feature_names))
        )

        matched_name = target_author
        if target_author not in profiles:
            # Büyük/küçük harf toleransı
            for k in profiles.keys():
                if k.lower() == target_author.lower():
                    matched_name = k
                    break

        if matched_name not in profiles:
            # Profil bulunamazsa kanon pasajlarından dinamik profil üret
            target_profile = self._fallback_author_profile(matched_name, feature_names)
        else:
            target_profile = profiles[matched_name]

        # Z-skorları hesapla
        eps = 1e-6
        user_vec = np.array([user_features.get(f, 0.0) for f in feature_names])
        user_z = (user_vec - corpus_mean) / (corpus_std + eps)
        target_z = (target_profile - corpus_mean) / (corpus_std + eps)

        z_diffs = user_z - target_z
        abs_diffs = np.abs(z_diffs)

        # En büyük sapmaların indeksleri
        top_indices = np.argsort(abs_diffs)[::-1][:6]

        deviations: List[Dict[str, Any]] = []
        for idx in top_indices:
            feat_key = feature_names[idx]
            u_val = float(user_vec[idx])
            t_val = float(target_profile[idx])
            z_gap = float(z_diffs[idx])

            direction = "Fazla (Azaltılmalı)" if z_gap > 0 else "Eksik (Artırılmalı)"
            friendly_name = self.FEATURE_FRIENDLY_NAMES.get(feat_key, feat_key.replace("_", " ").title())

            deviations.append({
                "feature": feat_key,
                "friendly_name": friendly_name,
                "user_value": round(u_val, 4),
                "target_value": round(t_val, 4),
                "z_diff": round(z_gap, 2),
                "abs_z_diff": round(abs(z_gap), 2),
                "direction": direction,
            })

        # Ortalama Burrows Delta mesafesi üzerinden uyum yüzdesi
        mean_delta = float(np.mean(abs_diffs))
        similarity_pct = round(float(np.clip(100.0 / (1.0 + mean_delta), 12.0, 97.5)), 1)

        return {
            "target_author": matched_name,
            "similarity_pct": similarity_pct,
            "mean_delta_distance": round(mean_delta, 3),
            "top_deviations": deviations[:5],
            "user_features": user_features,
        }

    def generate_prescription(
        self,
        user_text: str,
        target_author: str,
    ) -> List[Dict[str, str]]:
        """
        Kullanıcıya hedef yazarın üslubuna yaklaşabilmesi için somut,
        adım adım matematiksel ve pedagojik tavsiyeler üretir.
        
        Args:
            user_text (str): Kullanıcı metni.
            target_author (str): Hedef yazar adı.
            
        Returns:
            List[Dict[str, str]]: Başlık, eylem türü, öneri ve etki açıklaması içeren liste.
        """
        comparison = self.compare_with_target_author(user_text, target_author)
        deviations = comparison["top_deviations"]
        prescriptions: List[Dict[str, str]] = []

        for dev in deviations:
            feat = dev["feature"]
            u_val = dev["user_value"]
            t_val = dev["target_value"]
            z_gap = dev["z_diff"]
            auth = comparison["target_author"]

            rec_dict = self._build_feature_prescription(feat, u_val, t_val, z_gap, auth)
            if rec_dict:
                prescriptions.append(rec_dict)

        # Eğer reçete sayısı 3'ten azsa genel edebi üslup tavsiyesi ekle
        if len(prescriptions) < 3:
            prescriptions.append({
                "category": "Genel Sözdizimi & Ritim",
                "action": "Dengeleme",
                "recommendation": f"{auth} üslubundaki edebi ritmi yakalamak için düşüncelerinizi benzer paragraf blokları ve ritmik kelime tekrarlarıyla yapılandırın.",
                "impact": "Metnin genel edebi ahengini güçlendirir.",
            })

        return prescriptions

    def highlight_stylistic_markers(self, user_text: str) -> Dict[str, Any]:
        """
        Metin içindeki aşırı uzun cümleleri (30+ kelime), '-ly' zarflarını,
        arkaik kelimeleri ve noktalı virgülleri tespit edip istatistiklerini çıkarır.
        
        Args:
            user_text (str): Kullanıcı metni.
            
        Returns:
            Dict[str, Any]: Tespit edilen işaretçiler ve sayısal özetleri.
        """
        sentences = [
            s.strip() for s in re.split(r"[.!?]+", user_text) if s.strip()
        ]
        words = re.findall(r"[A-Za-z']+", user_text)
        lower_words = [w.lower().strip("'") for w in words]

        # 1. Uzun Cümleler (≥ 30 kelime)
        long_sentences = []
        for s in sentences:
            s_words = re.findall(r"[A-Za-z']+", s)
            if len(s_words) >= 30:
                long_sentences.append({
                    "word_count": len(s_words),
                    "preview": s[:85] + ("..." if len(s) > 85 else ""),
                })

        # 2. -ly Zarfları
        ly_words = sorted(list({w for w in lower_words if w.endswith("ly") and len(w) > 4 and not w.endswith(("fly", "reply", "apply"))}))

        # 3. Arkaik Kelimeler
        known_archaic = {
            "thou", "thee", "thy", "thine", "hath", "doth", "dost", "art",
            "'tis", "'twas", "wherefore", "whither", "hither", "ye", "nay",
            "aye", "behold", "quoth", "oft", "anon", "ere",
        }
        found_archaic = sorted(list({w for w in lower_words if w in known_archaic}))

        # 4. Noktalama İşaretçileri
        semicolons = user_text.count(";")
        colons = user_text.count(":")
        dashes = user_text.count("—") + user_text.count("--")

        return {
            "total_words": len(words),
            "total_sentences": max(1, len(sentences)),
            "long_sentences": long_sentences,
            "long_sentence_count": len(long_sentences),
            "adverbs_ly": ly_words,
            "adverb_count": len(ly_words),
            "archaic_words": found_archaic,
            "archaic_count": len(found_archaic),
            "semicolon_count": semicolons,
            "colon_count": colons,
            "dash_count": dashes,
        }

    def _build_feature_prescription(
        self,
        feat: str,
        user_val: float,
        target_val: float,
        z_gap: float,
        target_author: str,
    ) -> Optional[Dict[str, str]]:
        """Öznitelik sapmasına göre hedefe yönelik somut tavsiye metni oluşturur."""
        if feat == "avg_sentence_length_words":
            diff = abs(user_val - target_val)
            if z_gap < 0:
                return {
                    "category": "Cümle Uzunluğu & Sözdizimi",
                    "action": "Cümleleri Uzatın",
                    "recommendation": (
                        f"Ortalama cümle uzunluğunuz {user_val:.1f} kelime. {target_author} üslubuna yaklaşmak için "
                        f"cümlelerinizi ortalama +{diff:.1f} kelime uzatmalı; bağlaçlar, sıfat tamlamaları ve yan cümlecikler eklemelisiniz."
                    ),
                    "impact": "Metnin anlatı akışını klasik dönem realizmine yaklaştırır.",
                }
            else:
                return {
                    "category": "Cümle Uzunluğu & Sözdizimi",
                    "action": "Cümleleri Kısaltın",
                    "recommendation": (
                        f"Ortalama cümle uzunluğunuz {user_val:.1f} kelime. {target_author} üslubuna yaklaşmak için "
                        f"cümlelerinizi ortalama -{diff:.1f} kelime kısaltmalı; düşüncelerinizi daha vurucu ve net parçalara bölmelisiniz."
                    ),
                    "impact": "Cümlelerin doğrudanlığını ve berraklığını artırır.",
                }

        if feat == "suffix_ly_ratio":
            u_pct = user_val * 100
            t_pct = target_val * 100
            if z_gap > 0:
                return {
                    "category": "Zarf Yoğunluğu (-ly)",
                    "action": "Zarfları Azaltın",
                    "recommendation": (
                        f"Metninizdeki '-ly' zarf oranı (%{u_pct:.1f}), {target_author} standardının (%{t_pct:.1f}) üzerinde. "
                        f"Anlatımı zayıflatan fazla zarfları eleyerek güçlü eylemler (fiiller) kullanın (Orwell ilkesi)."
                    ),
                    "impact": "Gereksiz niteleyicileri temizleyerek anlatıyı keskinleştirir.",
                }
            else:
                return {
                    "category": "Zarf Yoğunluğu (-ly)",
                    "action": "Betimleyici Zarf Ekleyin",
                    "recommendation": (
                        f"{target_author} üslubundaki zengin tasvir derinliğini yakalamak için '-ly' zarf kullanımınızı "
                        f"%{u_pct:.1f}'den %{t_pct:.1f}'e çıkarın."
                    ),
                    "impact": "Karakter hareketlerine ve atmosfer betimlemelerine derinlik katar.",
                }

        if feat == "archaic_total_ratio" or feat.startswith("suffix_eth"):
            if z_gap < 0:
                return {
                    "category": "Arkaik Dil & Tarihsel Doku",
                    "action": "Arkaik İfadeler Ekleyin",
                    "recommendation": (
                        f"{target_author} döneminin ruhunu yansıtmak için 'thou', 'thee', 'hath', 'doth', ''tis' gibi "
                        f"arkaik zamir ve fiil çekimlerini metne serpiştirin."
                    ),
                    "impact": "Rönesans ve erken modern İngilizce havası kazandırır.",
                }
            else:
                return {
                    "category": "Arkaik Dil & Tarihsel Doku",
                    "action": "Dili Güncelleyin",
                    "recommendation": (
                        f"{target_author} daha yalın bir kelime dağarcığı tercih eder; arkaik sözcükleri çağdaş karşılıklarıyla değiştirin."
                    ),
                    "impact": "Metni anakronik yapaylıktan kurtarır.",
                }

        if feat == "punct_semicolon_ratio":
            if z_gap < 0:
                return {
                    "category": "Noktalama & Bağlantı",
                    "action": "Noktalı Virgül (;) Kullanın",
                    "recommendation": (
                        f"Birbiriyle doğrudan ilişkili bağımsız cümleleri noktayla ayırmak yerine noktalı virgül (;) ile "
                        f"bağlayarak {target_author} tarzı dengeli bir retorik kurun."
                    ),
                    "impact": "Cümleler arası mantıksal geçişleri pürüzsüzleştirir.",
                }
            else:
                return {
                    "category": "Noktalama & Bağlantı",
                    "action": "Noktalı Virgülleri Azaltın",
                    "recommendation": (
                        f"Noktalı virgül sıklığınız {target_author} ortalamasından yüksek. Uzun bileşikleri iki bağımsız cümleye ayırın."
                    ),
                    "impact": "Metnin okunabilirlik temposunu artırır.",
                }

        if feat in ("ttr", "hapax_legomena_ratio"):
            if z_gap < 0:
                return {
                    "category": "Kelime Çeşitliliği (TTR)",
                    "action": "Söz Dağarcığını Zenginleştirin",
                    "recommendation": (
                        f"Kelime çeşitliliğiniz (TTR: {user_val:.3f}), {target_author} düzeyinin ({target_val:.3f}) gerisinde. "
                        f"Aynı kelimeleri tekrarlamak yerine edebi eşanlamlılarını (synonyms) kullanın."
                    ),
                    "impact": "Tekdüzelikten kurtararak leksikal zenginlik katar.",
                }
            else:
                return {
                    "category": "Kelime Çeşitliliği (TTR)",
                    "action": "Kelime Dağarcığını Sadeleştirin",
                    "recommendation": (
                        f"{target_author} belirli tematik sözcükleri stratejik olarak tekrar eder; sözcük çeşitliliğini hafifçe törpüleyin."
                    ),
                    "impact": "Ana temaların daha akılda kalıcı olmasını sağlar.",
                }

        if feat == "stream_of_consciousness_index":
            if z_gap < 0:
                return {
                    "category": "Modernist Bilinç Akışı",
                    "action": "İç Monolog ve Tireler Ekleyin",
                    "recommendation": (
                        f"{target_author} üslubundaki psikolojik dalgalanmayı yakalamak için düşünce kırılmalarını tire (—), "
                        f"üç nokta (...) ve iç ses monologlarıyla aktarın."
                    ),
                    "impact": "Zihinsel akışı ve modernist derinliği güçlendirir.",
                }

        friendly = self.FEATURE_FRIENDLY_NAMES.get(feat, feat)
        action_verb = "Artırın" if z_gap < 0 else "Azaltın"
        return {
            "category": friendly,
            "action": f"{friendly} Değerini {action_verb}",
            "recommendation": (
                f"Mevcut {friendly} değeriniz ({user_val:.3f}), {target_author} ortalamasının ({target_val:.3f}) "
                f"{'altında' if z_gap < 0 else 'üzerinde'}. Hedef üsluba yaklaşmak için bu özniteliği dengeleyin."
            ),
            "impact": f"Hedef yazar ile aranızdaki stilometrik Z-sapmasını azaltır.",
        }

    def _fallback_author_profile(self, author_name: str, feature_names: List[str]) -> np.ndarray:
        """Kanon kataloğu örnek metinlerinden hızlı yedek yazar profili çıkarır."""
        entry = next((e for e in CANON_CATALOG.values() if e.name.lower() == author_name.lower()), None)
        if entry and entry.sample_passages:
            combined = " ".join(entry.sample_passages)
            feats = self.extractor.extract_features(combined)
            return np.array([feats.get(f, 0.0) for f in feature_names])
        return np.zeros(len(feature_names))
