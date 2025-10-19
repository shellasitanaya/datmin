# Laporan Analisis Pelatihan Model Otomatis

### Proyek: Analisis Data (Jenis Masalah: regression)

---

## 1. Analisis Perbandingan Model

Proses pelatihan otomatis telah selesai dijalankan untuk menemukan model terbaik.

**Wawasan Utama:**
* **Model Pemenang 🏆:** **CatBoost Regressor** menunjukkan performa terbaik secara keseluruhan, unggul dalam metrik evaluasi utama.
* Model-model berbasis *ensemble* (seperti CatBoost, Random Forest, Extra Trees) umumnya menunjukkan performa yang solid, menandakan dataset ini memiliki pola yang cukup kompleks.


**Tabel Ringkasan Performa (3 Model Teratas):**

| Model                   |     MAE |         MSE |    RMSE |     R2 |   RMSLE |   MAPE |   TT (Sec) |
|:------------------------|--------:|------------:|--------:|-------:|--------:|-------:|-----------:|
| CatBoost Regressor      | 584.662 | 1.41162e+06 | 1162.17 | 0.9867 |  0.0641 | 0.0471 |      0.347 |
| Extra Trees Regressor   | 719.687 | 2.03387e+06 | 1390.89 | 0.9809 |  0.0786 | 0.0585 |      0.114 |
| Random Forest Regressor | 725.23  | 2.33639e+06 | 1491.56 | 0.9781 |  0.0785 | 0.0577 |      0.099 |


## 2. Rincian Pipeline Pra-Pemrosesan

Sebelum model dilatih, data mentah telah melalui serangkaian proses persiapan otomatis untuk memastikan kualitasnya, seperti:
- **Imputasi Data:** Mengisi nilai yang hilang secara otomatis.
- **Encoding Kategorikal:** Mengubah data teks menjadi format numerik yang dapat diproses oleh model.



## 3. Saran dan Rekomendasi Perbaikan

Model **CatBoost Regressor** menunjukkan performa yang sangat baik dan dapat menjadi kandidat kuat untuk implementasi.

**Langkah selanjutnya yang direkomendasikan:**
- **Lakukan *Hyperparameter Tuning*:** Jalankan optimisasi pada model `CatBoost Regressor` untuk potensi peningkatan akurasi lebih lanjut.
- **Analisis Kepentingan Fitur:** Gunakan model yang sudah dilatih untuk memahami fitur mana yang paling berpengaruh terhadap prediksi.
- **Finalisasi Model:** Latih ulang model terbaik pada keseluruhan dataset, lalu simpan untuk digunakan di masa depan.
