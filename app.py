# =====================================================
# app.py - FINAL WORKING VERSION (March 2026)
# =====================================================

import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
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
ANOMALY_THRESHOLD = 0.0766   # Change this if needed after testing

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
def detect_anomaly(image_path):
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        recon, _ = model(tensor)
    
    error = F.l1_loss(recon, tensor).item()
    is_anomaly = error > ANOMALY_THRESHOLD
    percentage = min((error / ANOMALY_THRESHOLD) * 100, 100)
    
    # Plot Original vs Reconstructed
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
    
    return error, is_anomaly, percentage

# ====================== USERS (in-memory) ======================
users = {}

# ====================== ROUTES ======================
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        if username in users:
            flash('Username already exists!', 'danger')
            return render_template('register.html')
        users[username] = generate_password_hash(password)
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        if username in users and check_password_hash(users[username], password):
            session['user'] = username
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid username or password!', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    results = []

    if request.method == 'POST':

        files = request.files.getlist('files')

        if not files or files[0].filename == '':
            flash('Please select at least one image!', 'warning')
            return render_template('dashboard.html', results=results)

        for file in files:

            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            try:
                error, is_anomaly, percentage = detect_anomaly(filepath)

                results.append({
                    'filename': file.filename,
                    'error': round(error, 4),
                    'percentage': round(percentage, 1),
                    'message': "🚨 ANOMALY DETECTED!" if is_anomaly else "✅ NORMAL IMAGE",
                    'alert_class': "danger" if is_anomaly else "success"
                })

            except Exception as e:
                flash(f'Error processing {file.filename}: {str(e)}', 'danger')

            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)

        flash('Analysis completed successfully!', 'success')

    return render_template('dashboard.html', results=results)

# ====================== START APP ======================
if __name__ == '__main__':
    print("\n🚀 AuraMed Anomaly Detection App Started!")
    print("   Go to: http://127.0.0.1:5000/register")
    app.run(debug=True)