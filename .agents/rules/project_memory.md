# TÜBİTAK Stilometri Atlası - Geliştirici Kural ve Hafıza Dosyası

Bu dosya Antigravity IDE her başlatıldığında otomatik olarak okunur ve şu kuralları uygular:

1. **Geliştirici Kimliği:** Proje Bayram Akpamukçu tarafından geliştirilen TÜBİTAK 2209 İngiliz Edebiyatı Stilometri Atlası'dır.
2. **Canlı Alan Adı:** `bayramakpamukcu.site` ve `www.bayramakpamukcu.site` canlıda çalışmaktadır.
3. **Masaüstü Paketleme:** `launcher.py` pywebview ile masaüstü penceresi olarak çalışır. `build_exe.py` çalıştırıldığında masaüstü .exe ve zip oluşturulur.
4. **Model ve Veri Güvenliği:** Her yazar için en az 4 blok oluşturulur (`min_chunks=4`), `evaluate()` metodunda `labels` parametresi daima açık verilir.
5. **Streamlit Ayarları:** `.streamlit/config.toml` içindeki `enableCORS = false` ve `enableXsrfProtection = false` ayarları iframe entegrasyonu için zorunludur.
