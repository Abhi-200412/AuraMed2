# =====================================================
# app.py - FINAL WORKING VERSION (March 2026)
# =====================================================

import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib
# MUST use 'Agg' backend to avoid thread GUI errors in Flask
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash

# Import the corrected model
from model import SBCAE

app = Flask(__name__)
app.secret_key = 'aura-med-secret-key-2026-change-in-production'

# Folders
UPLOAD_FOLDER = 'uploads'
STATIC_IMGS = 'static/imgs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_IMGS, exist_ok=True)

# Device & Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ANOMALY_THRESHOLD = 0.0408   # Change this if needed after testing

# ====================== LOAD MODEL ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

model = SBCAE(latent_dim=256).to(DEVICE)

try:
    state_dict = torch.load("final_model.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ Model loaded successfully on {DEVICE}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Make sure final_model.pth and model.py are in the same folder!")
    exit()

# ====================== DETECTION FUNCTION ======================
def detect_anomaly(image_path, threshold):
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        recon, _ = model(tensor)

    error = F.l1_loss(recon, tensor).item()

    # ---------------------------
    # BINARY DECISION SYSTEM
    # ---------------------------

    if error <= threshold:
        status = "normal"
        message = "✅ NORMAL IMAGE"
        alert_class = "success"
        is_anomaly = False
    else:
        status = "anomaly"
        message = "🚨 ANOMALY DETECTED!"
        alert_class = "danger"
        is_anomaly = True

    percentage = min((error / threshold) * 100, 100)

    # ---------------------------
    # PLOT
    # ---------------------------

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor='#1a1a2e')

    orig = np.clip((tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 0.5) + 0.5, 0, 1)
    rec  = np.clip((recon.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 0.5) + 0.5, 0, 1)

    axes[0].imshow(orig)
    axes[0].set_title('Original Image', color='white', fontsize=16)
    axes[0].axis('off')

    axes[1].imshow(rec)
    axes[1].set_title(f'Reconstructed\n(Error: {error:.4f})', color='white', fontsize=16)
    axes[1].axis('off')

    plt.tight_layout()
    plot_path = os.path.join(STATIC_IMGS, 'result.png')
    plt.savefig(plot_path, dpi=300, facecolor='#1a1a2e')
    plt.close()

    return error, percentage, status, message, alert_class, is_anomaly

import sqlite3
import csv
import io
from flask import Response

# ====================== DATABASE SETUP ======================
DB_FILE = 'users.db'

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin BOOLEAN DEFAULT 0
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            error_score REAL NOT NULL,
            percentage REAL NOT NULL,
            is_anomaly BOOLEAN NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    ''')
    c.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('ANOMALY_THRESHOLD', '0.0408')")
    
    # Check for default admin
    c.execute('SELECT id FROM users WHERE username = ?', ('admin',))
    if not c.fetchone():
        from werkzeug.security import generate_password_hash
        hashed_pwd = generate_password_hash('admin123')
        c.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)', ('admin', hashed_pwd, 1))

    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()

# ====================== ROUTES ======================
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        
        hashed_password = generate_password_hash(password)
        
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists!', 'danger')
        finally:
            conn.close()
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('SELECT id, password, is_admin FROM users WHERE username = ?', (username,))
        result = c.fetchone()
        conn.close()
        
        if result and check_password_hash(result[1], password):
            session['user'] = username
            session['user_id'] = result[0]
            session['is_admin'] = bool(result[2])
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('dashboard'))
            
        flash('Invalid username or password!', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('user_id', None)
    session.pop('is_admin', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
        
    # Ensure user_id is in session (for users logged in before the update)
    if 'user_id' not in session:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('SELECT id FROM users WHERE username = ?', (session['user'],))
        row = c.fetchone()
        conn.close()
        if row:
            session['user_id'] = row[0]
        else:
            return redirect(url_for('logout'))

    results = []

    if request.method == 'POST':

        files = request.files.getlist('files')

        if not files or files[0].filename == '':
            flash('Please select at least one image!', 'warning')
            return render_template('dashboard.html', results=results)
            
        if len(files) > 10:
            flash('You can only upload a maximum of 10 images at once.', 'danger')
            return render_template('dashboard.html', results=results)

        # Fetch dynamic threshold
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT value FROM settings WHERE key='ANOMALY_THRESHOLD'")
        row = c.fetchone()
        conn.close()
        current_threshold = float(row[0]) if row else ANOMALY_THRESHOLD

        for file in files:

            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            try:
                error, percentage, status, message, alert_class, is_anomaly = detect_anomaly(filepath, current_threshold)

                # Save to history
                try:
                    conn = sqlite3.connect(DB_FILE)
                    c = conn.cursor()
                    c.execute('''
                        INSERT INTO history (user_id, filename, error_score, percentage, is_anomaly)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (session.get('user_id'), file.filename, error, percentage, is_anomaly))
                    conn.commit()
                    conn.close()
                except Exception as db_e:
                    print("DB Error:", db_e)

                results.append({
                    'filename': file.filename,
                    'error': round(error, 4),
                    'percentage': round(percentage, 1),
                    'message': message,
                    'alert_class': alert_class
                })

            except Exception as e:
                flash(f'Error processing {file.filename}: {str(e)}', 'danger')

            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)

        flash('Analysis completed successfully!', 'success')

    return render_template('dashboard.html', results=results)

@app.route('/history')
def history():
    if 'user' not in session:
        flash('Please log in to view your history.', 'danger')
        return redirect(url_for('login'))
        
    # Ensure user_id is in session
    user_id = session.get('user_id')
    if not user_id:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('SELECT id FROM users WHERE username = ?', (session['user'],))
        row = c.fetchone()
        conn.close()
        if row:
            user_id = row[0]
            session['user_id'] = user_id
        else:
            return redirect(url_for('logout'))
            
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT filename, error_score, percentage, is_anomaly, timestamp FROM history WHERE user_id = ? ORDER BY timestamp DESC', (user_id,))
    rows = c.fetchall()
    conn.close()
    
    history_records = []
    for r in rows:
        history_records.append({
            'filename': r[0],
            'error_score': round(r[1], 4),
            'percentage': round(r[2], 1),
            'is_anomaly': bool(r[3]),
            'timestamp': r[4]
        })
        
    return render_template('history.html', history=history_records)

@app.route('/admin')
def admin_panel():
    if 'user' not in session or not session.get('is_admin'):
        flash('Access denied. Administrators only.', 'danger')
        return redirect(url_for('dashboard'))
        
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Get stats
    c.execute('SELECT COUNT(*) FROM users')
    total_users = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM history')
    total_scans = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM history WHERE is_anomaly = 1')
    total_anomalies = c.fetchone()[0]
    
    c.execute("SELECT value FROM settings WHERE key='ANOMALY_THRESHOLD'")
    threshold_row = c.fetchone()
    current_threshold = float(threshold_row[0]) if threshold_row else ANOMALY_THRESHOLD

    # Get all users for the user management table & filter dropdown
    c.execute('SELECT id, username, is_admin FROM users ORDER BY id ASC')
    all_users = [{'id': r[0], 'username': r[1], 'is_admin': bool(r[2])} for r in c.fetchall()]
    
    # Filtering Logic
    filter_user_id = request.args.get('user_id', '')
    filter_anomaly = request.args.get('anomaly_only', '')
    
    query = '''
        SELECT u.username, h.filename, h.error_score, h.percentage, h.is_anomaly, h.timestamp 
        FROM history h 
        JOIN users u ON h.user_id = u.id 
        WHERE 1=1
    '''
    params = []
    
    if filter_user_id.isdigit():
        query += ' AND u.id = ?'
        params.append(int(filter_user_id))
    if filter_anomaly == '1':
        query += ' AND h.is_anomaly = 1'
        
    query += ' ORDER BY h.timestamp DESC'
    
    c.execute(query, params)
    rows = c.fetchall()
    # 1. Anomalies Over Time
    c.execute('''
        SELECT date(timestamp), COUNT(*), SUM(is_anomaly)
        FROM history
        GROUP BY date(timestamp)
        ORDER BY date(timestamp) DESC
        LIMIT 10
    ''')
    time_data = c.fetchall()
    dates = [row[0] for row in reversed(time_data)]
    total_scans_time = [row[1] for row in reversed(time_data)]
    anomalies_time = [row[2] for row in reversed(time_data)]
    
    # 2. User Activity (Top 5 active users)
    c.execute('''
        SELECT u.username, COUNT(h.id) as scan_count
        FROM users u
        LEFT JOIN history h ON u.id = h.user_id
        GROUP BY u.username
        ORDER BY scan_count DESC
        LIMIT 5
    ''')
    activity_data = c.fetchall()
    active_users = [row[0] for row in activity_data]
    user_scan_counts = [row[1] for row in activity_data]
    
    # 3. Error Distribution
    c.execute('''
        SELECT 
            SUM(CASE WHEN error_score < 0.02 THEN 1 ELSE 0 END),
            SUM(CASE WHEN error_score >= 0.02 AND error_score < 0.04 THEN 1 ELSE 0 END),
            SUM(CASE WHEN error_score >= 0.04 AND error_score < 0.06 THEN 1 ELSE 0 END),
            SUM(CASE WHEN error_score >= 0.06 AND error_score < 0.08 THEN 1 ELSE 0 END),
            SUM(CASE WHEN error_score >= 0.08 THEN 1 ELSE 0 END)
        FROM history
    ''')
    dist_row = c.fetchone()
    error_dist = [int(d) if d else 0 for d in dist_row] if dist_row else [0,0,0,0,0]

    conn.close()
    
    chart_data = {
        'dates': dates,
        'total_scans_time': total_scans_time,
        'anomalies_time': anomalies_time,
        'active_users': active_users,
        'user_scan_counts': user_scan_counts,
        'error_dist': error_dist
    }
    
    all_history = []
    for r in rows:
        all_history.append({
            'username': r[0],
            'filename': r[1],
            'error_score': round(r[2], 4),
            'percentage': round(r[3], 1),
            'is_anomaly': bool(r[4]),
            'timestamp': r[5]
        })
        
    stats = {
        'total_users': total_users,
        'total_scans': total_scans,
        'total_anomalies': total_anomalies
    }
        
    return render_template('admin.html', stats=stats, all_history=all_history, all_users=all_users, filters={'user_id': filter_user_id, 'anomaly_only': filter_anomaly}, chart_data=chart_data)



@app.route('/admin/export/csv')
def export_csv():
    if 'user' not in session or not session.get('is_admin'):
        return redirect(url_for('dashboard'))
        
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        SELECT u.username, h.filename, h.error_score, h.percentage, h.is_anomaly, h.timestamp 
        FROM history h 
        JOIN users u ON h.user_id = u.id 
        ORDER BY h.timestamp DESC
    ''')
    rows = c.fetchall()
    conn.close()
    
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Username', 'Filename', 'Error Score', 'Confidence %', 'Is Anomaly', 'Timestamp'])
    for r in rows:
        cw.writerow([r[0], r[1], round(r[2], 4), round(r[3], 1), 'Yes' if r[4] else 'No', r[5]])
        
    output = si.getvalue()
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=aura_med_history.csv"}
    )

@app.route('/admin/user/<int:uid>/promote', methods=['POST'])
def promote_user(uid):
    if 'user' not in session or not session.get('is_admin'):
        return redirect(url_for('dashboard'))
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE users SET is_admin = 1 WHERE id = ?", (uid,))
    conn.commit()
    conn.close()
    flash('User promoted to Administrator.', 'success')
    return redirect(url_for('admin_panel'))

@app.route('/admin/user/<int:uid>/delete', methods=['POST'])
def delete_user(uid):
    if 'user' not in session or not session.get('is_admin'):
        return redirect(url_for('dashboard'))
    if uid == session.get('user_id'):
        flash('You cannot delete your own account.', 'danger')
        return redirect(url_for('admin_panel'))
        
    conn = sqlite3.connect(DB_FILE)
    # Enable foreign keys for cascade delete
    conn.execute("PRAGMA foreign_keys = ON")
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE id = ?", (uid,))
    conn.commit()
    conn.close()
    flash('User and their associated history deleted.', 'success')
    return redirect(url_for('admin_panel'))

# ====================== START APP ======================
if __name__ == '__main__':
    print("\n🚀 AuraMed Anomaly Detection App Started!")
    print("   Go to: http://127.0.0.1:5000/register")
    app.run(debug=True)