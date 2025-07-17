import base64
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, Response, session
from werkzeug.utils import secure_filename
import os
import datetime
import numpy as np
import cv2
import tensorflow as tf
from db import get_db_connection, init_db
import shutil
import json
import subprocess
from PIL import Image
from werkzeug.security import check_password_hash, generate_password_hash
from functools import wraps
import glob
from mtcnn import MTCNN
import calendar

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Route untuk landing page
@app.route('/landing')
def landing():
    return render_template('landing.html')

# Jadikan landing page sebagai tampilan awal
@app.route('/')
def root():
    return redirect(url_for('landing'))

# Decorator untuk proteksi route
def login_required(role=None):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session or (role and session.get('role') != role):
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = 'dataset/face_model.h5'
MAPPING_PATH = 'dataset/class_indices.json'

def reload_model():
    global model, class_indices, inv_class_indices
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(MAPPING_PATH) as f:
        class_indices = json.load(f)
    inv_class_indices = {v: k for k, v in class_indices.items()}

# Inisialisasi database dan model saat startup
init_db()
reload_model()

# Tambahan untuk video streaming
camera = cv2.VideoCapture(0)

def normalize_name(name):
    # Hilangkan spasi depan belakang dan karakter aneh, ganti spasi dengan underscore
    return '_'.join(name.strip().split())

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Yield frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/index')
@login_required()
def index():
    if session.get('role') == 'mahasiswa':
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
        if user:
            total_kehadiran = conn.execute('SELECT COUNT(*) FROM attendance WHERE user_id = ? AND status = "hadir"', (user['id'],)).fetchone()[0]
            total_hari = 30
            persentase_kehadiran = round((total_kehadiran / total_hari) * 100, 2) if total_hari else 0
            total_terlambat = 0
            now = datetime.datetime.now()
            year, month = now.year, now.month
            attendance_per_day_hadir = [0]*30
            attendance_per_day_sakit = [0]*30
            attendance_per_day_alfa = [0]*30
            rows = conn.execute('SELECT timestamp, status FROM attendance WHERE user_id = ?', (user['id'],)).fetchall()
            for row in rows:
                try:
                    tgl = datetime.datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
                    if tgl.year == year and tgl.month == month and 1 <= tgl.day <= 30:
                        if row['status'] == 'hadir':
                            attendance_per_day_hadir[tgl.day-1] += 1
                        elif row['status'] == 'sakit':
                            attendance_per_day_sakit[tgl.day-1] += 1
                        elif row['status'] in ('alfa', 'bolos'):
                            attendance_per_day_alfa[tgl.day-1] += 1
                except Exception:
                    pass
            conn.close()
            return render_template('index.html',
                nama=user['nama'],
                prodi=user['prodi'],
                semester=user['semester'],
                universitas=user['universitas'],
                total_kehadiran=total_kehadiran,
                persentase_kehadiran=persentase_kehadiran,
                total_terlambat=total_terlambat,
                attendance_per_day_hadir=attendance_per_day_hadir,
                attendance_per_day_sakit=attendance_per_day_sakit,
                attendance_per_day_alfa=attendance_per_day_alfa
            )
        else:
            conn.close()
            return render_template('index.html',
                nama='-', prodi='-', semester='-', universitas='-',
                total_kehadiran=0, persentase_kehadiran=0, total_terlambat=0,
                attendance_per_day_hadir=[0]*30,
                attendance_per_day_sakit=[0]*30,
                attendance_per_day_alfa=[0]*30
            )
    else:
        # Admin: tampilkan dashboard admin atau data semua mahasiswa (bisa dikembangkan)
        return render_template('dashboard.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        nama = request.form['nama']
        nim = request.form['nim']
        prodi = request.form['prodi']
        semester = request.form['semester']
        universitas = "Teknik Informatika Universitas Bale Bandung"
        files = request.files.getlist('foto')

        # Normalisasi nama untuk folder dan database
        nama_folder = normalize_name(nama)
        dir_path = os.path.join(app.config['UPLOAD_FOLDER'], nama_folder)
        os.makedirs(dir_path, exist_ok=True)  # Buat folder sekali saja sebelum loop
        dataset_dir = os.path.join('dataset', nama_folder)
        os.makedirs(dataset_dir, exist_ok=True)

        detector = MTCNN()
        wajah_terdeteksi = 0
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                img = Image.open(file.stream).convert('RGB')
                img_np = np.array(img)
                faces = detector.detect_faces(img_np)
                if faces:
                    x, y, w, h = faces[0]['box']
                    # Pastikan koordinat positif
                    x, y = max(0, x), max(0, y)
                    face_img = img.crop((x, y, x+w, y+h)).resize((160, 160))
                    face_img.save(os.path.join(dir_path, filename))
                    face_img.save(os.path.join(dataset_dir, filename))
                    wajah_terdeteksi += 1
                else:
                    flash(f'Wajah tidak terdeteksi pada file: {filename}, file dilewati.')

        if wajah_terdeteksi == 0:
            flash('Tidak ada wajah yang terdeteksi pada semua file yang diupload. Registrasi dibatalkan.')
            return redirect(url_for('register'))

        conn = get_db_connection()
        conn.execute(
            'INSERT INTO users (nama, nim, prodi, semester, universitas) VALUES (?, ?, ?, ?, ?)',
            (nama_folder, nim, prodi, semester, universitas)
        )
        conn.commit()
        user_id = conn.execute('SELECT id FROM users WHERE nama = ?', (nama_folder,)).fetchone()['id']
        meta = {
            "user_id": user_id,
            "nama": nama_folder,
            "nim": nim,
            "prodi": prodi,
            "semester": semester,
            "universitas": universitas
        }
        with open(os.path.join(dataset_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f)
        conn.close()

        # Otomatisasi retrain model setelah pendaftaran user baru
        try:
            subprocess.run(['python', 'train_faces.py'], check=True)
            reload_model()  # Reload model dan mapping setelah training
            flash('Registrasi berhasil! Model telah di-train ulang. Silakan lakukan absensi.')
        except Exception as e:
            flash(f'Registrasi berhasil, tapi training model gagal: {e}')
        return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/attendance')
def attendance():
    return render_template('attendance.html')

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    data = request.json
    img_data = data['image'].split(",")[1]
    decoded = base64.b64decode(img_data)
    nparr = np.frombuffer(decoded, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_resized = cv2.resize(img, (160,160))
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)

    predictions = model.predict(img_expanded)
    predicted_class = int(np.argmax(predictions, axis=1)[0])
    confidence = float(np.max(predictions))

    # Logging prediksi
    print("Predictions:", predictions)
    print("Predicted class:", predicted_class)
    print("Confidence:", confidence)
    print("Mapping index ke nama:", inv_class_indices)

    user_id = None
    # Ambil user_id dari class_id_map.json
    with open('dataset/class_id_map.json') as f:
        class_id_map = json.load(f)
    user_id = class_id_map.get(str(predicted_class), None)
    print("Predicted user_id:", user_id)
    user = None
    if confidence < 0.6:
        return jsonify({'status': 'error', 'message': f'Wajah tidak dikenali (confidence={confidence:.2f})'})
    if user_id:
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        if user:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status = data.get('status', 'hadir')
            conn.execute(
                'INSERT INTO attendance (user_id, timestamp, status) VALUES (?, ?, ?)',
                (user['id'], timestamp, status)
            )
            conn.commit()
            conn.close()
            return jsonify({
                'status': 'success',
                'nama': user['nama'],
                'nim': user['nim'],
                'prodi': user['prodi'],
                'semester': user['semester'],
                'universitas': user['universitas'],
                'confidence': round(confidence,2)
            })
        else:
            return jsonify({'status': 'error', 'message': 'Wajah tidak dikenali (user id tidak ditemukan).'})
    else:
        print('Mapping user id gagal untuk predicted_class:', predicted_class)
        return jsonify({'status': 'error', 'message': 'Wajah tidak dikenali (mapping user id gagal).'})

@app.route('/rekap')
def rekap():
    conn = get_db_connection()
    records = conn.execute('''
        SELECT a.timestamp, u.nama, u.nim, u.prodi, u.semester, u.universitas
        FROM attendance a
        JOIN users u ON a.user_id = u.id
        ORDER BY a.timestamp DESC
    ''').fetchall()
    conn.close()
    return render_template('rekap.html', records=records)

@app.route('/profile')
@login_required('mahasiswa')
def profile():
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    conn.close()
    return render_template('profile.html',
        nama=user['nama'], nim=user['nim'], prodi=user['prodi'],
        semester=user['semester'], universitas=user['universitas'])

@app.route('/quick_register', methods=['POST'])
def quick_register():
    data = request.json
    nama = data['nama']
    nim = data['nim']
    prodi = data['prodi']
    semester = data['semester']
    universitas = "Teknik Informatika Universitas Bale Bandung"
    images = data['images']  # List of 3 base64 images

    if len(images) < 3:
        return jsonify({'status': 'error', 'message': 'Harus upload 3 foto (depan, kanan, kiri) untuk pendaftaran.'})

    nama_folder = normalize_name(nama)
    dir_path = os.path.join(app.config['UPLOAD_FOLDER'], nama_folder)
    os.makedirs(dir_path, exist_ok=True)
    dataset_dir = os.path.join('dataset', nama_folder)
    os.makedirs(dataset_dir, exist_ok=True)

    # Cek duplikasi user berdasarkan nama_folder atau NIM
    conn = get_db_connection()
    existing = conn.execute('SELECT * FROM users WHERE nama = ? OR nim = ?', (nama_folder, nim)).fetchone()
    if existing:
        conn.close()
        return jsonify({'status': 'error', 'message': 'User dengan nama atau NIM ini sudah terdaftar.'})

    # Simpan dan resize 3 foto
    for idx, img_b64 in enumerate(images):
        img_data = img_b64.split(",")[1]
        decoded = base64.b64decode(img_data)
        nparr = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img).convert('RGB').resize((160, 160))
        filename = f"{nama_folder}_{idx+1}.jpg"
        pil_img.save(os.path.join(dir_path, filename))
        pil_img.save(os.path.join(dataset_dir, filename))

    # Simpan ke database
    conn.execute(
        'INSERT INTO users (nama, nim, prodi, semester, universitas) VALUES (?, ?, ?, ?, ?)',
        (nama_folder, nim, prodi, semester, universitas)
    )
    conn.commit()
    user_id = conn.execute('SELECT id FROM users WHERE nama = ?', (nama_folder,)).fetchone()['id']
    meta = {
        "user_id": user_id,
        "nama": nama_folder,
        "nim": nim,
        "prodi": prodi,
        "semester": semester,
        "universitas": universitas
    }
    with open(os.path.join(dataset_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)
    conn.close()

    # Retrain model dan reload
    try:
        subprocess.run(['python', 'train_faces.py'], check=True)
        reload_model()
        return jsonify({'status': 'success', 'message': 'Pendaftaran berhasil, model sudah diupdate. Silakan absensi ulang.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Pendaftaran gagal: {e}'})

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        role = request.form.get('role')
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        if role == 'admin':
            admin = conn.execute('SELECT * FROM admin WHERE username = ?', (username,)).fetchone()
            conn.close()
            if admin and check_password_hash(admin['password'], password):
                session['user_id'] = admin['id']
                session['role'] = 'admin'
                flash('Login admin berhasil!')
                return redirect(url_for('index'))
            else:
                flash('Username atau password admin salah!')
        else:  # mahasiswa
            user = conn.execute('SELECT * FROM users WHERE nim = ?', (username,)).fetchone()
            conn.close()
            if user and password == user['nim']:
                session['user_id'] = user['id']
                session['role'] = 'mahasiswa'
                flash('Login mahasiswa berhasil!')
                return redirect(url_for('index'))
            else:
                flash('NIM tidak ditemukan atau salah!')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Berhasil logout.')
    return redirect(url_for('login'))

@app.route('/student_register', methods=['GET', 'POST'])
def student_register():
    if request.method == 'POST':
        nama = request.form['nama']
        nim = request.form['nim']
        prodi = request.form['prodi']
        semester = request.form['semester']
        universitas = "Teknik Informatika Universitas Bale Bandung"
        # Cek duplikasi NIM
        conn = get_db_connection()
        existing = conn.execute('SELECT * FROM users WHERE nim = ?', (nim,)).fetchone()
        if existing:
            conn.close()
            flash('NIM sudah terdaftar!')
            return redirect(url_for('student_register'))
        # Simpan ke database
        nama_folder = normalize_name(nama)
        conn.execute(
            'INSERT INTO users (nama, nim, prodi, semester, universitas) VALUES (?, ?, ?, ?, ?)',
            (nama_folder, nim, prodi, semester, universitas)
        )
        conn.commit()
        conn.close()
        flash('Akun mahasiswa berhasil dibuat! Silakan login.')
        return redirect(url_for('login'))
    return render_template('student_register.html')

@app.route('/admin')
@login_required('admin')
def admin_dashboard():
    conn = get_db_connection()
    total_mahasiswa = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
    total_absensi = conn.execute('SELECT COUNT(*) FROM attendance').fetchone()[0]
    # Ambil 10 hari terakhir untuk grafik
    chart_labels = []
    chart_data = []
    rows = conn.execute('''
        SELECT substr(timestamp, 1, 10) as tgl, COUNT(*) as jumlah
        FROM attendance
        GROUP BY tgl
        ORDER BY tgl DESC
        LIMIT 10
    ''').fetchall()
    rows = list(reversed(rows))
    for r in rows:
        chart_labels.append(r['tgl'])
        chart_data.append(r['jumlah'])
    conn.close()
    # Hitung total foto wajah
    total_foto = 0
    for folder in os.listdir('dataset'):
        folder_path = os.path.join('dataset', folder)
        if os.path.isdir(folder_path):
            total_foto += len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    # Info model
    try:
        with open('dataset/class_indices.json') as f:
            class_indices = json.load(f)
        num_classes = len(class_indices)
    except Exception:
        num_classes = 0
    try:
        last_retrain = datetime.datetime.fromtimestamp(os.path.getmtime('dataset/face_model.h5')).strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        last_retrain = '-'

    # Hitung statistik per hari untuk grafik
    now = datetime.datetime.now()
    days_in_month = calendar.monthrange(now.year, now.month)[1]
    chart_data_hadir = [0]*days_in_month
    chart_data_sakit = [0]*days_in_month
    chart_data_alfa = [0]*days_in_month
    total_sakit = 0
    total_alfa = 0

    for folder in os.listdir('dataset'):
        folder_path = os.path.join('dataset', folder)
        if os.path.isdir(folder_path):
            meta_file = os.path.join(folder_path, 'meta.json')
            if os.path.exists(meta_file):
                with open(meta_file) as f:
                    meta = json.load(f)
                    user_id = meta.get('user_id')
                    if user_id:
                        conn = get_db_connection()
                        user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
                        if user:
                            # Ambil data absensi dari database
                            attendance_rows = conn.execute('''
                                SELECT timestamp, status
                                FROM attendance
                                WHERE user_id = ? AND timestamp >= ? AND timestamp < ?
                            ''', (
                                user_id,
                                datetime.datetime(now.year, now.month, 1),
                                datetime.datetime(now.year, now.month, days_in_month) + datetime.timedelta(days=1)
                            )).fetchall()
                            for row in attendance_rows:
                                try:
                                    tgl = datetime.datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
                                    if 1 <= tgl.day <= days_in_month:
                                        if row['status'] == 'hadir':
                                            chart_data_hadir[tgl.day-1] += 1
                                        elif row['status'] == 'sakit':
                                            chart_data_sakit[tgl.day-1] += 1
                                            total_sakit += 1
                                        elif row['status'] in ('alfa', 'bolos'):
                                            chart_data_alfa[tgl.day-1] += 1
                                            total_alfa += 1
                                except Exception:
                                    pass
                        conn.close()

    print('chart_data_hadir:', chart_data_hadir)
    print('chart_data_alfa:', chart_data_alfa)

    return render_template(
        'admin_dashboard.html',
        menu='dashboard',
        total_mahasiswa=total_mahasiswa,
        total_absensi=total_absensi,
        total_foto=total_foto,
        num_classes=num_classes,
        last_retrain=last_retrain,
        chart_labels=[f'Hari {i+1}' for i in range(days_in_month)],
        chart_data_hadir=chart_data_hadir,
        chart_data_sakit=chart_data_sakit,
        chart_data_alfa=chart_data_alfa,
        total_sakit=total_sakit,
        total_alfa=total_alfa
    )

@app.route('/admin/users')
@login_required('admin')
def admin_users():
    conn = get_db_connection()
    users = conn.execute('SELECT * FROM users').fetchall()
    conn.close()
    return render_template('admin_users.html', menu='users', users=users)

@app.route('/admin/users/<int:user_id>')
@login_required('admin')
def admin_user_detail(user_id):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    if not user:
        flash('Mahasiswa tidak ditemukan!')
        return redirect(url_for('admin_users'))
    return render_template('admin_user_detail.html', menu='user_detail', user=user)

@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
@login_required('admin')
def admin_user_delete(user_id):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    if user:
        conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        flash('Mahasiswa berhasil dihapus!')
    else:
        flash('Mahasiswa tidak ditemukan!')
    conn.close()
    return redirect(url_for('admin_users'))

@app.route('/admin/attendance')
@login_required('admin')
def admin_attendance():
    conn = get_db_connection()
    records = conn.execute('''
        SELECT a.id, u.nama, u.nim, u.prodi, a.timestamp
        FROM attendance a
        JOIN users u ON a.user_id = u.id
        ORDER BY a.timestamp DESC
    ''').fetchall()
    conn.close()
    return render_template('admin_attendance.html', menu='attendance', records=records)

@app.route('/admin/faces', methods=['GET', 'POST'])
@login_required('admin')
def admin_faces():
    conn = get_db_connection()
    users = conn.execute('SELECT * FROM users').fetchall()
    conn.close()
    message = None
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        photo = request.files.get('photo')
        is_profile = request.form.get('is_profile') == 'on'
        if user_id and photo:
            user = next((u for u in users if str(u['id']) == str(user_id)), None)
            if user:
                folder = os.path.join('dataset', user['nama'])
                os.makedirs(folder, exist_ok=True)
                filename = secure_filename(photo.filename)
                if is_profile:
                    filename = 'profile.jpg'
                filepath = os.path.join(folder, filename)
                img = Image.open(photo).convert('RGB').resize((160, 160))
                img.save(filepath)
                # Simpan juga ke static/uploads agar bisa diakses dari web
                upload_folder = os.path.join('static', 'uploads', user['nama'])
                os.makedirs(upload_folder, exist_ok=True)
                img.save(os.path.join(upload_folder, filename))
                message = f'Foto wajah berhasil diupload untuk {user["nama"]}!'
            else:
                message = 'Mahasiswa tidak ditemukan.'
        else:
            message = 'Form tidak lengkap.'
        flash(message)
        return redirect(url_for('admin_faces'))
    faces_data = []
    for u in users:
        # Cek profile.jpg di static/uploads
        upload_folder = os.path.join('static', 'uploads', u['nama'])
        profile_path = os.path.join(upload_folder, 'profile.jpg')
        if os.path.exists(profile_path):
            images = ['profile.jpg']
        else:
            # Jika tidak ada profile.jpg, ambil foto pertama (jika ada)
            all_imgs = sorted([os.path.basename(f) for f in glob.glob(os.path.join(upload_folder, '*.jpg'))])
            images = all_imgs[:1] if all_imgs else []
        faces_data.append({'user': u, 'images': images})
    return render_template('admin_faces.html', menu='faces', faces_data=faces_data, users=users)

@app.route('/admin/model')
@login_required('admin')
def admin_model():
    try:
        with open('dataset/class_indices.json') as f:
            class_indices = json.load(f)
        num_classes = len(class_indices)
    except Exception:
        num_classes = 0
    total_images = 0
    for folder in os.listdir('dataset'):
        folder_path = os.path.join('dataset', folder)
        if os.path.isdir(folder_path):
            total_images += len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    try:
        last_retrain = datetime.datetime.fromtimestamp(os.path.getmtime('dataset/face_model.h5')).strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        last_retrain = '-'
    return render_template('admin_model.html', menu='model', num_classes=num_classes, total_images=total_images, last_retrain=last_retrain)

@app.route('/admin/export')
@login_required('admin')
def admin_export():
    return render_template('admin_dashboard.html', menu='export')

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    data = request.json
    img_data = data['image'].split(",")[1]
    decoded = base64.b64decode(img_data)
    nparr = np.frombuffer(decoded, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_resized = cv2.resize(img, (160,160))
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)

    predictions = model.predict(img_expanded)
    predicted_class = int(np.argmax(predictions, axis=1)[0])
    confidence = float(np.max(predictions))

    # Ambil user_id dari class_id_map.json
    with open('dataset/class_id_map.json') as f:
        class_id_map = json.load(f)
    user_id = class_id_map.get(str(predicted_class), None)

    if confidence < 0.6 or not user_id:
        return jsonify({'status': 'error', 'message': f'Wajah tidak dikenali (confidence={confidence:.2f})'})

    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    if user:
        return jsonify({
            'status': 'success',
            'nama': user['nama'],
            'nim': user['nim'],
            'prodi': user['prodi'],
            'semester': user['semester'],
            'universitas': user['universitas'],
            'confidence': round(confidence,2)
        })
    else:
        return jsonify({'status': 'error', 'message': 'User tidak ditemukan di database.'})

if __name__ == '__main__':
    app.run(debug=True)
