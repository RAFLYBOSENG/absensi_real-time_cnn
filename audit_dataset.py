#!/usr/bin/env python3
"""
Script Audit Dataset, Mapping, dan Database
===========================================
Script ini mengecek konsistensi antara:
1. Folder dataset dan user di database
2. Meta.json di setiap folder dataset
3. class_indices.json dan class_id_map.json
4. User_id di mapping dengan database

Usage: python audit_dataset.py
"""

import os
import json
from db import get_db_connection
from utils import normalize_name, get_dataset_folders

def audit_dataset_mapping():
    """Audit lengkap dataset, mapping, dan database"""
    print("🔍 AUDIT DATASET, MAPPING, DAN DATABASE")
    print("=" * 50)
    
    # 1. Cek folder dataset vs user database
    print("\n📁 1. AUDIT FOLDER DATASET vs DATABASE")
    print("-" * 30)
    folders = set(get_dataset_folders('dataset'))
    conn = get_db_connection()
    db_users = set([row['nama'] for row in conn.execute('SELECT nama FROM users').fetchall()])
    conn.close()
    
    orphan_folders = folders - db_users
    orphan_users = db_users - folders
    
    print(f"📊 Total folder dataset: {len(folders)}")
    print(f"📊 Total user database: {len(db_users)}")
    
    if orphan_folders:
        print(f"❌ Folder tanpa user di database: {orphan_folders}")
    else:
        print("✅ Semua folder dataset ada user di database")
    
    if orphan_users:
        print(f"❌ User database tanpa folder dataset: {orphan_users}")
    else:
        print("✅ Semua user database ada folder dataset")
    
    # 2. Cek meta.json di setiap folder
    print("\n📄 2. AUDIT META.JSON")
    print("-" * 30)
    meta_errors = []
    meta_success = 0
    
    for folder in folders:
        meta_path = os.path.join('dataset', folder, 'meta.json')
        if not os.path.exists(meta_path):
            meta_errors.append(f"❌ {folder}: meta.json tidak ada")
        else:
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                # Validasi struktur meta.json
                required_keys = ['user_id', 'nama', 'nim', 'prodi', 'semester', 'universitas']
                missing_keys = [key for key in required_keys if key not in meta]
                if missing_keys:
                    meta_errors.append(f"❌ {folder}: meta.json tidak lengkap (missing: {missing_keys})")
                else:
                    meta_success += 1
            except Exception as e:
                meta_errors.append(f"❌ {folder}: meta.json error - {e}")
    
    print(f"✅ Meta.json valid: {meta_success}")
    for error in meta_errors:
        print(error)
    
    # 3. Cek mapping files
    print("\n🗂️ 3. AUDIT MAPPING FILES")
    print("-" * 30)
    mapping_errors = []
    
    # Cek class_indices.json
    indices_path = 'dataset/class_indices.json'
    if not os.path.exists(indices_path):
        mapping_errors.append("❌ class_indices.json tidak ada")
    else:
        try:
            with open(indices_path) as f:
                class_indices = json.load(f)
            print(f"✅ class_indices.json: {len(class_indices)} kelas")
        except Exception as e:
            mapping_errors.append(f"❌ class_indices.json error: {e}")
    
    # Cek class_id_map.json
    id_map_path = 'dataset/class_id_map.json'
    if not os.path.exists(id_map_path):
        mapping_errors.append("❌ class_id_map.json tidak ada")
    else:
        try:
            with open(id_map_path) as f:
                class_id_map = json.load(f)
            print(f"✅ class_id_map.json: {len(class_id_map)} mapping")
        except Exception as e:
            mapping_errors.append(f"❌ class_id_map.json error: {e}")
    
    # 4. Validasi konsistensi mapping
    print("\n🔗 4. VALIDASI KONSISTENSI MAPPING")
    print("-" * 30)
    
    if 'class_indices' in locals() and 'class_id_map' in locals():
        # Cek index consistency
        indices_set = set(str(v) for v in class_indices.values())
        id_map_set = set(class_id_map.keys())
        
        if indices_set != id_map_set:
            print(f"❌ Index tidak sinkron!")
            print(f"   class_indices: {indices_set}")
            print(f"   class_id_map: {id_map_set}")
            mapping_errors.append("Index mapping tidak sinkron")
        else:
            print("✅ Index mapping sinkron")
        
        # Cek user_id validity
        conn = get_db_connection()
        user_ids_db = set(row['id'] for row in conn.execute('SELECT id FROM users').fetchall())
        user_ids_map = set(int(v) for v in class_id_map.values())
        conn.close()
        
        invalid_user_ids = user_ids_map - user_ids_db
        if invalid_user_ids:
            print(f"❌ User_id tidak valid di database: {invalid_user_ids}")
            mapping_errors.append("Ada user_id tidak valid di database")
        else:
            print("✅ Semua user_id valid di database")
    
    # 5. Cek model file
    print("\n🤖 5. AUDIT MODEL FILE")
    print("-" * 30)
    model_path = 'dataset/face_model.h5'
    if not os.path.exists(model_path):
        print("❌ face_model.h5 tidak ada")
        mapping_errors.append("Model file tidak ada")
    else:
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"✅ face_model.h5: {file_size:.1f} MB")
    
    # 6. Summary
    print("\n📋 6. RINGKASAN AUDIT")
    print("-" * 30)
    total_errors = len(orphan_folders) + len(orphan_users) + len(meta_errors) + len(mapping_errors)
    
    if total_errors == 0:
        print("🎉 SEMUA VALID! Dataset, mapping, dan database sudah sinkron.")
        return True
    else:
        print(f"⚠️ Ditemukan {total_errors} masalah yang perlu diperbaiki:")
        if orphan_folders:
            print(f"   - {len(orphan_folders)} folder tanpa user di database")
        if orphan_users:
            print(f"   - {len(orphan_users)} user tanpa folder dataset")
        if meta_errors:
            print(f"   - {len(meta_errors)} masalah meta.json")
        if mapping_errors:
            print(f"   - {len(mapping_errors)} masalah mapping")
        return False

def fix_common_issues():
    """Perbaiki masalah umum yang ditemukan"""
    print("\n🔧 PERBAIKAN OTOMATIS")
    print("-" * 30)
    
    # Generate meta.json yang hilang
    print("📄 Memperbaiki meta.json...")
    conn = get_db_connection()
    fixed_count = 0
    
    for folder in get_dataset_folders('dataset'):
        meta_path = os.path.join('dataset', folder, 'meta.json')
        if not os.path.exists(meta_path):
            user = conn.execute('SELECT * FROM users WHERE nama = ?', (folder,)).fetchone()
            if user:
                meta = {
                    "user_id": user['id'],
                    "nama": user['nama'],
                    "nim": user['nim'],
                    "prodi": user['prodi'],
                    "semester": user['semester'],
                    "universitas": user['universitas']
                }
                with open(meta_path, 'w') as f:
                    json.dump(meta, f)
                print(f"✅ meta.json dibuat untuk {folder}")
                fixed_count += 1
    
    conn.close()
    print(f"📊 Meta.json diperbaiki: {fixed_count} file")

if __name__ == "__main__":
    # Jalankan audit
    is_valid = audit_dataset_mapping()
    
    if not is_valid:
        print("\n❓ Apakah ingin menjalankan perbaikan otomatis? (y/n)")
        response = input().lower().strip()
        if response == 'y':
            fix_common_issues()
            print("\n🔄 Menjalankan audit ulang...")
            audit_dataset_mapping()
        else:
            print("💡 Untuk memperbaiki masalah, jalankan: python train_faces.py")
    else:
        print("\n🎯 Sistem siap untuk training dan absensi!") 