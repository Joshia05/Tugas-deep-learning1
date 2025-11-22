import os
import shutil
import pandas as pd

# Path ke folder train dan test
train_folder = 'data/train'
test_folder = 'data/test'

# Path ke file CSV
train_csv = 'data/train.csv'
test_csv = 'data/test.csv'

# Fungsi untuk memindahkan gambar berdasarkan CSV
def move_images(csv_file, folder_path):
    # Baca CSV
    df = pd.read_csv(csv_file)
    
    # Iterasi melalui setiap baris di CSV
    for index, row in df.iterrows():
        filename = row['filename']
        label = row['label']
        
        # Tentukan path sumber gambar dan folder label tujuan
        src_path = os.path.join(folder_path, filename)
        label_folder = os.path.join(folder_path, label)
        
        # Jika folder label belum ada, buat folder tersebut
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)
        
        # Tentukan path tujuan untuk gambar
        dest_path = os.path.join(label_folder, filename)
        
        # Cek apakah file ada di folder sumber
        if os.path.exists(src_path):
            # Pindahkan gambar ke folder label yang sesuai
            shutil.move(src_path, dest_path)
            print(f"Pindah {filename} ke {dest_path}")
        else:
            print(f"File {filename} tidak ditemukan di {folder_path}")

# Pindahkan gambar di folder train dan test
move_images(train_csv, train_folder)
move_images(test_csv, test_folder)
