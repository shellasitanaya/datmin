# ==============================================================================
# OTOMATISASI PELATIHAN MODEL DAN PEMBUATAN LAPORAN DENGAN PYCARET & AI (VERSI OFFLINE)
# ==============================================================================
#
# TUJUAN:
# 1. EKSPERIMENTASI MODEL: Menemukan model machine learning terbaik secara otomatis.
# 2. PELAPORAN OTOMATIS: Membuat laporan analisis dari hasil pelatihan secara otomatis
#    tanpa memerlukan koneksi internet atau API Key.
#
# ==============================================================================


# ------------------------------------------------------------------------------
# BAGIAN 1: MENGIMPOR LIBRARY YANG DIBUTUHKAN
# ------------------------------------------------------------------------------
import os
import argparse  # Untuk membuat antarmuka baris perintah (command-line interface)
import json      # Untuk bekerja dengan data format JSON (meskipun tidak dipakai di versi ini)
import pandas as pd # Library utama untuk manipulasi data (DataFrame)

# Mengimpor modul-modul spesifik dari PyCaret
from pycaret.classification import setup as setup_clf, compare_models as compare_clf, pull as pull_clf
from pycaret.regression import setup as setup_reg, compare_models as compare_reg, pull as pull_reg

# ------------------------------------------------------------------------------
# BAGIAN 2: FUNGSI UNTUK MELATIH MODEL (TRAIN_MODEL)
# ------------------------------------------------------------------------------
def train_model(data_input: pd.DataFrame, problem_type: str, target_column: str):
    """
    Fungsi ini menjalankan seluruh proses pelatihan model menggunakan PyCaret.
    """
    print(f"🎯 Kolom Target: {target_column}")
    print(f"🧠 Jenis Masalah: {problem_type}")

    if problem_type == "classification":
        # Mempersiapkan data untuk klasifikasi
        setup_clf(data=data_input, target=target_column, verbose=False, session_id=123)
        print("\n🚀 Memulai perbandingan model untuk Klasifikasi...")
        best_model = compare_clf()
        comparison_table = pull_clf()

    elif problem_type == "regression":
        # Mempersiapkan data untuk regresi
        setup_reg(data=data_input, target=target_column, verbose=False, session_id=123)
        print("\n🚀 Memulai perbandingan model untuk Regresi...")
        best_model = compare_reg()
        comparison_table = pull_reg()
    else:
        raise ValueError("Tipe masalah tidak valid. Pilih 'classification' atau 'regression'.")

    print(f"✅ Model terbaik ditemukan: {type(best_model).__name__}")
    return comparison_table

# ------------------------------------------------------------------------------
# BAGIAN 3: FUNGSI UNTUK MEMBUAT LAPORAN (VERSI OFFLINE TANPA AI)
# ------------------------------------------------------------------------------
def generate_offline_report(comparison_table: pd.DataFrame, problem_type: str):
    """
    Fungsi ini membuat laporan template berdasarkan hasil dari PyCaret.
    Tidak memerlukan koneksi ke API AI.
    """
    print("🤖 Membuat laporan analisis otomatis (mode offline)...")

    # Ambil informasi dari baris pertama (model terbaik)
    best_model_stats = comparison_table.iloc[0]
    best_model_name = best_model_stats['Model']

    # Buat template laporan dalam format Markdown
    md = [f"# Laporan Analisis Pelatihan Model Otomatis\n"]
    md.append(f"### Proyek: Analisis Data (Jenis Masalah: {problem_type})\n")
    md.append("---\n")

    # Bagian 1: Analisis Perbandingan Model
    md.append("## 1. Analisis Perbandingan Model\n")
    comments = (
        f"Proses pelatihan otomatis telah selesai dijalankan untuk menemukan model terbaik.\n\n"
        f"**Wawasan Utama:**\n"
        f"* **Model Pemenang 🏆:** **{best_model_name}** menunjukkan performa terbaik secara keseluruhan, unggul dalam metrik evaluasi utama.\n"
        f"* Model-model berbasis *ensemble* (seperti CatBoost, Random Forest, Extra Trees) umumnya menunjukkan performa yang solid, menandakan dataset ini memiliki pola yang cukup kompleks.\n"
    )
    md.append(comments)

    # Tambahkan tabel ringkasan 3 model teratas
    top_3_models = comparison_table.head(3).to_markdown(index=False)
    md.append("\n**Tabel Ringkasan Performa (3 Model Teratas):**\n")
    md.append(top_3_models)
    md.append("\n")

    # Bagian 2: Rincian Pipeline
    md.append("## 2. Rincian Pipeline Pra-Pemrosesan\n")
    pipeline_desc = (
        "Sebelum model dilatih, data mentah telah melalui serangkaian proses persiapan otomatis untuk memastikan kualitasnya, seperti:\n"
        "- **Imputasi Data:** Mengisi nilai yang hilang secara otomatis.\n"
        "- **Encoding Kategorikal:** Mengubah data teks menjadi format numerik yang dapat diproses oleh model.\n"
    )
    md.append(pipeline_desc)
    md.append("\n")

    # Bagian 3: Rekomendasi
    md.append("## 3. Saran dan Rekomendasi Perbaikan\n")
    suggestions = (
        f"Model **{best_model_name}** menunjukkan performa yang sangat baik dan dapat menjadi kandidat kuat untuk implementasi.\n\n"
        f"**Langkah selanjutnya yang direkomendasikan:**\n"
        f"- **Lakukan *Hyperparameter Tuning*:** Jalankan optimisasi pada model `{best_model_name}` untuk potensi peningkatan akurasi lebih lanjut.\n"
        f"- **Analisis Kepentingan Fitur:** Gunakan model yang sudah dilatih untuk memahami fitur mana yang paling berpengaruh terhadap prediksi.\n"
        f"- **Finalisasi Model:** Latih ulang model terbaik pada keseluruhan dataset, lalu simpan untuk digunakan di masa depan.\n"
    )
    md.append(suggestions)

    # Gabungkan semua bagian menjadi satu string
    final_report = "\n".join(md)

    # Menyimpan laporan ke file
    with open('laporan_analisis_otomatis.md', 'w', encoding='utf-8') as f:
        f.write(final_report)

    print("✅ Laporan analisis berhasil dibuat dan disimpan sebagai 'laporan_analisis_otomatis.md'")

    return {
        'status': 'sukses',
        'report': final_report
    }

# ------------------------------------------------------------------------------
# BAGIAN 4: TITIK MASUK UTAMA PROGRAM (MAIN EXECUTION BLOCK)
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AutoML and AI Reporting Tool")
    parser.add_argument('--data_input', type=str, required=True, help='Path menuju file input CSV.')
    parser.add_argument('--target_column', type=str, required=True, help='Nama kolom target yang akan diprediksi.')
    parser.add_argument('--problem_type', type=str, required=True, choices=["classification", "regression"], help='Tipe masalah machine learning.')

    args = parser.parse_args()

    try:
        df = pd.read_csv(args.data_input)
    except FileNotFoundError:
        print(f"❌ Error: File tidak ditemukan di path '{args.data_input}'")
        exit()

    # Langkah 1: Jalankan workflow data mining dengan PyCaret.
    comparison_table_result = train_model(
        data_input=df,
        problem_type=args.problem_type,
        target_column=args.target_column
    )

    # Langkah 2: Buat laporan dari hasil data mining (tanpa AI).
    output_report_dict = generate_offline_report(
        comparison_table=comparison_table_result,
        problem_type=args.problem_type
    )

    # Langkah 3: Tampilkan laporan akhir di terminal.
    if output_report_dict['status'] == 'sukses':
        print("\n" + "="*50)
        print("📋 LAPORAN ANALISIS OTOMATIS (FORMAT MARKDOWN)")
        print("="*50)
        print(output_report_dict['report'])