# AuraMed - Medical Anomaly Detection System

AuraMed is an advanced, AI-powered medical anomaly detection platform designed to assist radiologists and clinicians in identifying abnormalities in medical imagery (CT, MRI, and X-Rays). Utilizing a custom **Spatial-Channel Attention Autoencoder (SBCAE)**, the system performs zero-shot anomaly detection by learning a robust baseline of healthy anatomy and flagging deviations without requiring meticulously labeled pathological data.

## 🌟 Key Features

*   **Zero-Shot Anomaly Detection**: Built on unsupervised learning principles. The model is trained exclusively on healthy scans, allowing it to detect unseen or rare anomalies (tumors, lesions, tuberculosis) via reconstruction error (L1 Loss).
*   **Decoupled Architecture**: A lightweight, glassmorphism-styled Flask frontend securely interfaces with a powerful PyTorch inference backend.
*   **Clinical Dashboard**: Secure user authentication (Standard Users vs. Administrators), personal scan history tracking, and visual diagnostic reports.
*   **Admin Analytics**: Global scan history tracking, user management, and CSV data exporting.
*   **Dynamic Thresholding**: The evaluation pipeline automatically calculates the optimal classification threshold using the Youden Index and Balanced Accuracy metrics.

## 📊 Datasets

AuraMed was trained and evaluated using high-quality open-source medical imaging datasets:
*   **Brain Tumor Multimodal Image (CT and MRI):** [Kaggle Dataset Link](https://www.kaggle.com/datasets/murtozalikhon/brain-tumor-multimodal-image-ct-and-mri)
*   **Tuberculosis (TB) Chest X-Rays:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)

## ⚙️ Data Preprocessing

Before images are passed into the SBCAE model for training or inference, a standardized preprocessing pipeline is automatically applied to ensure computational stability across all modalities:
1.  **Format Conversion:** All scans are strictly converted to RGB space (preventing structural tensor crashes when handling native grayscale X-Rays).
2.  **Resolution Resizing:** Images are bilinearly interpolated to exactly `224x224` pixels to match the input layer of the autoencoder.
3.  **Tensor Normalization:** Pixel intensities are converted to PyTorch Tensors and normalized to a mean and standard deviation of `0.5` (mapping values to a `[-1, 1]` range) to optimize gradient convergence.

## 🚀 Installation & Setup

1. **Clone the repository** (or download the project folder).
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Application**:
   ```bash
   python app.py
   ```
4. **Access the Web Interface**: Open your browser and navigate to `http://127.0.0.1:5000`.

*(Note: On the first run, the SQLite database `users.db` will be automatically initialized with a default `admin` user).*

## 🧠 Model Training & Evaluation

### Bulk Preprocessing (`preprocessing.py`)
Before training or evaluating on a completely new raw dataset, you must format the data. The bulk preprocessing script recursively scans your raw dataset directory, applies the RGB conversions, interpolation, and tensor normalizations, and saves the clean data to the project's data folder.
Ensure you configure the `INPUT_ROOT` and `OUTPUT_ROOT` variables inside the script to point to your local directories, then run:
```bash
python preprocessing.py
```

### Training (`train.py`)
To retrain the SBCAE model on your own dataset, ensure your data is structured in `data/train/normal/` and run:
```bash
python train.py
```
This will generate `final_model.pth`.

### Evaluation (`eval.py`)
To evaluate the model against a test set (containing both `normal` and `anomaly` folders), run:
```bash
python eval.py
```
This script computes reconstruction scores, optimizes the classification threshold, and generates highly detailed visualizations (Confusion Matrix, ROC Curve, Score Distribution).

## 📈 Performance Metrics

Based on our recent extensive testing across mixed modalities, AuraMed achieved the following state-of-the-art results:

*   **AUROC:** 0.9827
*   **Accuracy:** 95.57%
*   **Precision:** 95.27%
*   **Recall (Sensitivity):** 98.39%
*   **Specificity:** 89.50%
*   **Optimal Threshold:** 0.0408

### Visualizations generated during evaluation:
*   `confusion_matrix.png`: Demonstrates a highly sensitive detection rate (98.4% true positive rate for anomalies), minimizing dangerous false negatives.
*   `roc_curve.png`: Shows the trade-off between TPR and FPR with the optimal operating point highlighted.
*   `threshold_analysis.png`: Tracks F1 Score and Balanced Accuracy across all possible thresholds to mathematically prove the optimal cutoff.

## 🛠️ Technology Stack
*   **Backend & Routing:** Python, Flask, Werkzeug
*   **Deep Learning:** PyTorch, Torchvision
*   **Data Science:** NumPy, Pandas, Scikit-Learn, Matplotlib, Seaborn
*   **Database:** SQLite3
*   **Frontend:** HTML5, Vanilla CSS (Glassmorphism UI), JavaScript, Chart.js
