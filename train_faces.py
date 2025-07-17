import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
import os
import json
import shutil
from PIL import Image
import sqlite3
from db import get_db_connection
from utils import normalize_name, get_dataset_folders

# Parameter
IMG_SIZE = (160, 160)
BATCH_SIZE = 20
EPOCHS = 50  # Naikkan epoch agar training lebih optimal
FINE_TUNE_EPOCHS = 15  # Fine-tuning juga dinaikkan
TRAIN_DIR = 'dataset'

# Augmentasi dan rescaling
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True
)

# Generator training
train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Generator validasi
val_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# Load base MobileNetV2 (tanpa fully-connected layer)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Freeze semua layer base
for layer in base_model.layers:
    layer.trainable = False

# Tambahkan layer classifier di atasnya
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# Unfreeze sebagian layer untuk fine-tuning
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Re-compile
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tuning
history_finetune = model.fit(
    train_generator,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=val_generator
)

# Buat folder model jika belum ada
os.makedirs("model", exist_ok=True)

# Simpan model
model.save('model/face_model.h5')

# Simpan mapping kelas
with open('model/class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)

# Copy file model dan mapping ke folder dataset
os.makedirs('dataset', exist_ok=True)
shutil.copy('model/face_model.h5', 'dataset/face_model.h5')
shutil.copy('model/class_indices.json', 'dataset/class_indices.json')

# Buat mapping index kelas ke user_id dari database
conn = sqlite3.connect('database.db')
cur = conn.cursor()
class_indices = train_generator.class_indices  # dict: nama_folder -> index
index_to_userid = {}
for folder_name, idx in class_indices.items():
    meta_path = os.path.join(TRAIN_DIR, folder_name, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        index_to_userid[str(idx)] = meta['user_id']
    else:
        print(f"WARNING: meta.json tidak ditemukan di {folder_name}, mapping gagal!")
conn.close()
with open('dataset/class_id_map.json', 'w') as f:
    json.dump(index_to_userid, f)
print("Mapping index ke user_id:", index_to_userid)

print("✅ Training selesai. Model disimpan di model/face_model.h5 dan dataset/")

def check_dataset_db_sync(dataset_dir='dataset'):
    # Ambil semua folder di dataset
    folders = set(get_dataset_folders(dataset_dir))
    # Ambil semua nama user di database
    conn = get_db_connection()
    db_users = set([row['nama'] for row in conn.execute('SELECT nama FROM users').fetchall()])
    conn.close()
    # Folder tanpa user di DB
    orphan_folders = folders - db_users
    # User DB tanpa folder
    orphan_users = db_users - folders
    if orphan_folders:
        print(f"WARNING: Folder tanpa user di database: {orphan_folders}")
    if orphan_users:
        print(f"WARNING: User di database tanpa folder dataset: {orphan_users}")
    if not orphan_folders and not orphan_users:
        print("Dataset dan database sudah sinkron.")
    return orphan_folders, orphan_users

# Panggil fungsi ini sebelum training
if __name__ == "__main__":
    orphan_folders, orphan_users = check_dataset_db_sync()
    # (Opsional) Hentikan training jika ada masalah sinkronisasi
    if orphan_folders or orphan_users:
        print("Perbaiki sinkronisasi dataset ↔ database sebelum training!")
        exit(1)
    # ... lanjutkan training ...
