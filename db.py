import sqlite3
from werkzeug.security import generate_password_hash

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    with open('instance/schema.sql') as f:
        conn.executescript(f.read())
    # Tambah kolom status jika belum ada
    try:
        conn.execute("ALTER TABLE attendance ADD COLUMN status TEXT DEFAULT 'hadir'")
    except Exception:
        pass  # Kolom sudah ada
    # Tambah kolom time_status jika belum ada
    try:
        conn.execute("ALTER TABLE attendance ADD COLUMN time_status TEXT DEFAULT 'hadir'")
    except Exception:
        pass  # Kolom sudah ada
    conn.commit()
    conn.close()

# Tambah admin default jika belum ada, atau update password jika sudah ada
try:
    username = 'admin'
    password = 'admin123'  # Ganti sesuai keinginan
    hashed = generate_password_hash(password)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT id FROM admin WHERE username = ?', (username,))
    if cur.fetchone():
        cur.execute('UPDATE admin SET password = ? WHERE username = ?', (hashed, username))
        print(f'Password admin "{username}" diupdate.')
    else:
        cur.execute('INSERT INTO admin (username, password) VALUES (?, ?)', (username, hashed))
        print(f'Admin "{username}" ditambahkan.')
    conn.commit()
    conn.close()
except Exception as e:
    print('Gagal menambah/memperbarui admin:', e)