# SPK UMKM Ayam Geprek (SAW + Naive Bayes & C.4.5)

Aplikasi web berbasis **Python + Streamlit** untuk klasifikasi perputaran menu (`Fast` / `Slow`) menggunakan **Naive Bayes & C.4.5** serta perangkingan prioritas restock menggunakan metode **SAW (*Simple Additive Weighting*)** pada UMKM Ayam Geprek.

## Struktur File
- `app.py` — Antarmuka web Streamlit (Dashboard, Klasifikasi, SAW, Ringkasan Rumus).
- `algorithms.py` — Implementasi algoritma SAW, Categorical Naive Bayes, dan C.4.5 (Gain Ratio).
- `umkm_ayam_geprek.csv` — Dataset 12 menu/produk UMKM Ayam Geprek (`6 Fast`, `6 Slow`).
- `requirements.txt` — Daftar library Python yang dibutuhkan.

## Cara Menjalankan Software
1. Pastikan Python 3.10+ sudah terinstal.
2. Instal library yang dibutuhkan:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```
4. Buka browser di alamat yang muncul di terminal (biasanya `http://localhost:8501`).
# umkm-algoritma
