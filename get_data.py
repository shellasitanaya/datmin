from pycaret.datasets import get_data
import pandas as pd

# Memuat dataset 'diamond' dari library PyCaret
diamond_dataset = get_data('diamond')

# Menyimpan dataset tersebut menjadi file bernama 'data.csv'
# index=False agar nomor baris tidak ikut disimpan
diamond_dataset.to_csv('data.csv', index=False)

print("✅ File 'data.csv' berisi data diamond berhasil dibuat!")