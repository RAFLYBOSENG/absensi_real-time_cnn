import os
import json
import random
from db import get_db_connection
from utils import normalize_name, get_dataset_folders

DATASET_DIR = 'dataset'
conn = get_db_connection()
folders = set(get_dataset_folders(DATASET_DIR))
db_users = set([row['nama'] for row in conn.execute('SELECT nama FROM users').fetchall()])

orphan_folders = folders - db_users

for folder in orphan_folders:
    norm_name = normalize_name(folder)
    # Cek lagi, siapa tahu sudah ada user dengan nama normalisasi
    user = conn.execute('SELECT * FROM users WHERE nama = ?', (norm_name,)).fetchone()
    if not user:
        # Tambah user dummy
        nim = str(random.randint(100000000, 999999999))
        prodi = "Teknik Informatika"
        semester = "1"
        universitas = "Universitas Bale Bandung"
        conn.execute(
            'INSERT INTO users (nama, nim, prodi, semester, universitas) VALUES (?, ?, ?, ?, ?)',
            (norm_name, nim, prodi, semester, universitas)
        )
        conn.commit()
        user = conn.execute('SELECT * FROM users WHERE nama = ?', (norm_name,)).fetchone()
        print(f"User {norm_name} ditambahkan ke database.")
    # Buat meta.json
    meta = {
        "user_id": user['id'],
        "nama": user['nama'],
        "nim": user['nim'],
        "prodi": user['prodi'],
        "semester": user['semester'],
        "universitas": user['universitas']
    }
    meta_path = os.path.join(DATASET_DIR, folder, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f)
    print(f"meta.json dibuat/diperbarui untuk {folder}")

conn.close()
print("Sinkronisasi selesai. Silakan jalankan ulang training.")
