
# OTOMATISASI PELATIHAN MODEL DAN PEMBUATAN LAPORAN DENGAN PYCARET & AI

# DATA  YANG DIOLAH:
# - Menggunakan dataset sampel bernama 'diamond', yang berisi informasi tentang ribuan berlian.
#
# PROSES DATA MINING YANG DILAKUKAN#
# 1. Data Preparation:
#    - Dilakukan secara otomatis oleh fungsi `setup()` dari PyCaret.
#    - Proses yang dicakup: imputasi (mengisi data hilang), encoding (mengubah
#      teks jadi angka), scaling (menyamakan skala fitur), dan pembagian data
#      menjadi set latih & uji.
#
# 2. Modeling:
#    - Dilakukan oleh fungsi `compare_models()` dari PyCaret.
#    - Proses ini adalah inti dari penemuan pola, di mana puluhan algoritma
#      dicoba pada data untuk menemukan mana yang memiliki daya prediksi terbaik.
#
# 3. Evaluation:
#    - Dilakukan dalam dua tahap:
#      a. EVALUASI KUANTITATIF: PyCaret menghitung metrik performa (seperti
#         Akurasi, R-squared, F1-score) untuk setiap model.
#      b. EVALUASI KUALITATIF (Otomatis): AI Anthropic (Claude) mengambil
#         hasil metrik tersebut dan memberikan interpretasi dalam bahasa
#         manusia, menjelaskan kekuatan/kelemahan model, dan memberikan konteks.
#
# 4. Deployment Preparation:
#    - Hasil akhirnya adalahsebuah model terlatih (objek PyCaret) dan laporan analisis. 


# 1. IMPORT LIBRARY
import os
import argparse  # Untuk membuat antarmuka baris perintah (command-line interface)
import json      # Untuk bekerja dengan data format JSON
import pandas as pd # Library utama untuk manipulasi data (DataFrame)
import anthropic # Library resmi untuk berinteraksi dengan API Anthropic (Claude)

# Mengimpor modul-modul spesifik dari PyCaret
# PyCaret adalah library utama yang menjalankan proses data mining (AutoML)
from pycaret.classification import setup as setup_clf, compare_models as compare_clf, pull as pull_clf
from pycaret.regression import setup as setup_reg, compare_models as compare_reg, pull as pull_reg


# 2. TRAIN MODEL
def train_model(data_input: pd.DataFrame, problem_type: str, target_column: str):
    """
    Fungsi ini menjalankan seluruh proses pelatihan model menggunakan PyCaret.
    Ini adalah inti dari workflow data mining dalam skrip ini.

    Args:
        data_input (pd.DataFrame): DataFrame yang berisi data untuk dilatih.
        problem_type (str): Jenis masalah ML ('classification' atau 'regression').
        target_column (str): Nama kolom yang akan diprediksi (kolom target).

    Returns:
        pd.DataFrame: Tabel (DataFrame) yang berisi perbandingan performa model.
    """
    print(f"🎯 Kolom Target: {target_column}")
    print(f"🧠 Jenis Masalah: {problem_type}")

    # Memilih alur kerja berdasarkan jenis masalah yang ditentukan pengguna
    if problem_type == "classification":
        # DATA MINING: TAHAP PERSIAPAN DATA untuk Klasifikasi
        # `setup_clf` mempersiapkan data: membagi data, mengisi nilai kosong, encoding, dll.
        setup_clf(data=data_input, target=target_column, verbose=False, session_id=123)
        print("\n🚀 Memulai perbandingan model untuk Klasifikasi...")
        
        # DATA MINING: TAHAP PEMODELAN & EVALUASI
        # `compare_clf` melatih dan mengevaluasi semua model klasifikasi yang tersedia.
        best_model = compare_clf()
        
        # DATA MINING: TAHAP EVALUASI
        # `pull_clf` mengambil tabel hasil perbandingan untuk dianalisis lebih lanjut.
        comparison_table = pull_clf()

    elif problem_type == "regression":
        # DATA MINING: TAHAP PERSIAPAN DATA untuk Regresi
        setup_reg(data=data_input, target=target_column, verbose=False, session_id=123)
        print("\n🚀 Memulai perbandingan model untuk Regresi...")
        
        # DATA MINING: TAHAP PEMODELAN & EVALUASI
        # `compare_reg` melatih dan mengevaluasi semua model regresi.
        best_model = compare_reg()
        
        # DATA MINING: TAHAP EVALUASI
        # `pull_reg` mengambil tabel hasil perbandingan.
        comparison_table = pull_reg()
    else:
        # Jika tipe masalah tidak valid, hentikan program
        raise ValueError("Tipe masalah tidak valid. Pilih 'classification' atau 'regression'.")
    
    print(f"✅ Model terbaik ditemukan: {type(best_model).__name__}")
    
    # Mengembalikan tabel perbandingan untuk dianalisis oleh AI
    return comparison_table


# 3. DESCRIBE TRAINING JOB (FUNGSI UNTUK MENGANALISIS HASIL DENGAN AI )
def describe_training_job(comparison_table: pd.DataFrame):
    """
    Fungsi ini mengirimkan hasil pelatihan ke AI Anthropic untuk dianalisis dan
    menghasilkan laporan terstruktur.

    Args:
        comparison_table (pd.DataFrame): Tabel perbandingan model dari PyCaret.

    Returns:
        dict: Sebuah dictionary yang berisi status dan laporan dalam format Markdown.
    """
    print("🤖 Menghubungi AI (Claude 3.5 Sonnet) untuk analisis...")

    # saving the API key as an environment variable
    os.environ['ANTHROPIC_API_KEY'] = "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"

    # connect to the API
    client = anthropic.Anthropic()

    # Mendefinisikan 'tool' atau skema output. Ini adalah "cetak biru" yang kita berikan
    # kepada AI untuk memastikan jawabannya terstruktur dan konsisten dalam format JSON.
    tools = [{
        "name": "pycaret_training_explainer",
        "description": "Struktur untuk penjelasan hasil training PyCaret...",
        "input_schema": {
            "type": "object",
            "properties": {
                "comparison_breakdown": {
                    "type": "object",
                    "properties": {"comments": {"type": "string", "description": "Wawasan dari hasil perbandingan model."}}
                },
                "pipeline_breakdown": {
                    "type": "object",
                    "description": "Detail setiap langkah dalam pipeline PyCaret.",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "Nama langkah (misal, 'imputer', 'scaler')."},
                                    "type": {"type": "string", "description": "Tipe transformer atau model yang digunakan."},
                                    "description": {"type": "string", "description": "Penjelasan tujuan dan fungsi langkah tersebut."}
                                },
                                "required": ["name", "type", "description"]
                            }
                        }
                    }
                },
                "model_suggestions": {
                    "type": "object",
                    "description": "Saran untuk meningkatkan pipeline pemodelan dan akurasi.",
                    "properties": {
                        "improvements": {
                            "type": "array",
                            "items": {"type": "string", "description": "Satu saran untuk perbaikan model."}
                        }
                    }
                }
            },
            "required": ["comparison_breakdown", "pipeline_breakdown", "model_suggestions"]
        }
    }]
    
    # Mengubah tabel perbandingan (DataFrame) dari PyCaret menjadi format JSON agar bisa dibaca oleh AI.
    result_json = json.dumps(comparison_table.to_dict(orient='records'), indent=4)

    # Membuat prompt (instruksi) yang akan dikirimkan ke AI.
    # Di sini kita memberikan peran, tugas, dan data yang dibutuhkan dalam satu instruksi.
    prompt = f"""Anda adalah seorang Data Scientist ahli...
    Tugas Anda adalah menganalisis hasil ini...
    Berikut adalah hasil training PyCaret dalam format JSON:
    {result_json}"""
    
    try:
        # Mengirimkan permintaan ke model AI Claude melalui API.
        # AI akan memproses prompt dan data JSON, lalu menghasilkan jawaban sesuai skema 'tools'.
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=2048,
            tools=tools,
            messages=[{"role": "user", "content": prompt}]
        )

        # Jawaban dari AI diekstrak dari respons API.
        analysis_result = {}
        for content in response.content:
            if content.type == 'tool_use':
                analysis_result = content.input
                break
        
        if not analysis_result:
            raise ValueError("AI tidak menghasilkan output yang sesuai dengan format yang diharapkan.")

        # Mengubah output JSON dari AI menjadi format Markdown yang lebih mudah dibaca.
        markdown_output = json_to_markdown(analysis_result)

        # Menyimpan laporan Markdown ke file lokal agar bisa dibuka dan dibaca nanti.
        with open('laporan_analisis_otomatis.md', 'w', encoding='utf-8') as f:
            f.write(markdown_output)
        
        print("✅ Laporan analisis berhasil dibuat dan disimpan sebagai 'laporan_analisis_otomatis.md'")
        
        # Mengembalikan status dan isi laporan untuk ditampilkan.
        return {
            'status': 'sukses',
            'report': markdown_output
        }
    
    except Exception as e:
        # Menangani jika terjadi error saat berkomunikasi dengan API.
        print(f"❌ Terjadi kesalahan saat berkomunikasi dengan AI: {e}")
        return {
            'status': 'gagal',
            'report': f"Error: {e}"
        }

# 4. JSON TO MARKDOWN (FUNGSI BANTU UNTUK MEMFORMAT LAPORAN)
def json_to_markdown(data: dict):
    """
    Fungsi ini mengubah output JSON terstruktur dari AI menjadi string Markdown
    agar mudah dibaca oleh manusia.
    """
    md = ["# Laporan Analisis Pelatihan Model Otomatis\n"]

    # Format bagian "Comparison Breakdown"
    md.append("## 1. Analisis Perbandingan Model\n")
    md.append(data.get("comparison_breakdown", {}).get("comments", "Tidak ada komentar."))
    md.append("\n")

    # Format bagian "Pipeline Breakdown"
    md.append("## 2. Rincian Pipeline Pra-Pemrosesan\n")
    steps = data.get("pipeline_breakdown", {}).get("steps", [])
    if steps:
        for i, step in enumerate(steps):
            md.append(f"### Langkah {i+1}: {step.get('name', 'Tanpa Nama')}")
            md.append(f"- **Tipe:** {step.get('type', 'Tidak diketahui')}")
            md.append(f"- **Deskripsi:** {step.get('description', 'Tidak ada deskripsi.')}\n")
    else:
        md.append("Tidak ada rincian pipeline.\n")

    # Format bagian "Model Suggestions"
    md.append("## 3. Saran dan Rekomendasi Perbaikan\n")
    improvements = data.get("model_suggestions", {}).get("improvements", [])
    if improvements:
        for suggestion in improvements:
            md.append(f"- {suggestion}")
    else:
        md.append("Tidak ada saran perbaikan.")
    
    return "\n".join(md)


# 5. MAIN EXECUTION BLOCK
if __name__ == '__main__':
    # Membuat parser untuk menerima argumen dari baris perintah saat skrip dijalankan.
    parser = argparse.ArgumentParser(description="AutoML and AI Reporting Tool")
    parser.add_argument('--data_input', type=str, required=True, help='Path menuju file input CSV.')
    parser.add_argument('--target_column', type=str, required=True, help='Nama kolom target yang akan diprediksi.')
    parser.add_argument('--problem_type', type=str, required=True, choices=["classification", "regression"], help='Tipe masalah machine learning.')
    
    # Membaca argumen yang diberikan oleh pengguna dari terminal.
    args = parser.parse_args()

    # Memuat data dari file CSV yang path-nya diberikan oleh pengguna.
    try:
        df = pd.read_csv(args.data_input)
    except FileNotFoundError:
        print(f"❌ Error: File tidak ditemukan di path '{args.data_input}'")
        exit()

    # Langkah 1: Memanggil fungsi untuk menjalankan workflow data mining dengan PyCaret.
    comparison_table_result = train_model(
        data_input=df,
        problem_type=args.problem_type,
        target_column=args.target_column
    )

    # Langkah 2: Memanggil fungsi untuk mengirim hasil data mining ke AI untuk dianalisis.
    output_report_dict = describe_training_job(comparison_table_result)
    
    # Langkah 3: Menampilkan laporan akhir di terminal jika proses berhasil.
    if output_report_dict['status'] == 'sukses':
        print("\n" + "="*50)
        print("📋 LAPORAN ANALISIS OTOMATIS (FORMAT MARKDOWN)")
        print("="*50)
        print(output_report_dict['report'])