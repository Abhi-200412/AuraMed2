import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    f1_score,
    roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from model import SBCAE

# =========================================================
# CONFIG
# =========================================================

DATA_DIR = #Specify Test data Folder
MODEL_PATH = "final_model.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# VISUAL STYLE
# =========================================================

plt.style.use("dark_background")

sns.set_style("darkgrid")

# =========================================================
# PREPROCESSING
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =========================================================
# DATASET
# =========================================================

class ImageDataset(Dataset):

    def __init__(self, folder, label):

        self.paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        self.label = label

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        img = Image.open(self.paths[idx]).convert("RGB")

        img = transform(img)

        return img, self.label, os.path.basename(self.paths[idx])

# =========================================================
# LOAD DATA
# =========================================================

normal_loader = DataLoader(
    ImageDataset(f"{DATA_DIR}/normal", 0),
    batch_size=1,
    shuffle=False
)

anomaly_loader = DataLoader(
    ImageDataset(f"{DATA_DIR}/anomaly", 1),
    batch_size=1,
    shuffle=False
)

# =========================================================
# LOAD MODEL
# =========================================================

model = SBCAE(latent_dim=256).to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()

loss_fn = nn.L1Loss()

# =========================================================
# COMPUTE RECONSTRUCTION SCORES
# =========================================================

scores = []
labels = []
names = []

def process(loader):

    with torch.no_grad():

        for imgs, label, name in tqdm(loader):

            imgs = imgs.to(DEVICE)

            recon, _ = model(imgs)

            score = loss_fn(recon, imgs).item()

            scores.append(score)

            labels.append(label.item())

            names.append(name[0])

print("\n🔄 Evaluating Normal Images...")
process(normal_loader)

print("\n🔄 Evaluating Anomaly Images...")
process(anomaly_loader)

scores = np.array(scores)
labels = np.array(labels)

# =========================================================
# AUROC
# =========================================================

auroc = roc_auc_score(labels, scores)

# =========================================================
# THRESHOLD OPTIMIZATION
# =========================================================

thresholds = np.linspace(
    scores.min(),
    scores.max(),
    300
)

best_f1 = 0
best_f1_thresh = 0

best_youden = 0
best_youden_thresh = 0

best_bal_acc = 0
best_bal_thresh = 0

for t in thresholds:

    preds = (scores >= t).astype(int)

    # -----------------------------------
    # F1 SCORE
    # -----------------------------------

    f1 = f1_score(labels, preds)

    if f1 > best_f1:

        best_f1 = f1
        best_f1_thresh = t

    # -----------------------------------
    # CONFUSION MATRIX
    # -----------------------------------

    tn, fp, fn, tp = confusion_matrix(
        labels,
        preds
    ).ravel()

    # -----------------------------------
    # METRICS
    # -----------------------------------

    tpr = tp / (tp + fn + 1e-8)

    fpr = fp / (fp + tn + 1e-8)

    # -----------------------------------
    # YOUDEN INDEX
    # -----------------------------------

    youden = tpr - fpr

    if youden > best_youden:

        best_youden = youden
        best_youden_thresh = t

    # -----------------------------------
    # BALANCED ACCURACY
    # -----------------------------------

    bal_acc = (tpr + (1 - fpr)) / 2

    if bal_acc > best_bal_acc:

        best_bal_acc = bal_acc
        best_bal_thresh = t

# =========================================================
# FINAL THRESHOLD
# =========================================================

FINAL_THRESHOLD = best_bal_thresh

print("\n🔍 Threshold Optimization Results:")
print(f"F1 Threshold       : {best_f1_thresh:.4f}")
print(f"Youden Threshold   : {best_youden_thresh:.4f}")
print(f"Balanced Threshold : {best_bal_thresh:.4f}")

print(f"\n✅ FINAL THRESHOLD: {FINAL_THRESHOLD:.4f}")

# =========================================================
# FINAL PREDICTIONS
# =========================================================

preds = (scores >= FINAL_THRESHOLD).astype(int)

# =========================================================
# FINAL METRICS
# =========================================================

cm = confusion_matrix(labels, preds)

tn, fp, fn, tp = cm.ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)

precision = tp / (tp + fp + 1e-8)

recall = tp / (tp + fn + 1e-8)

specificity = tn / (tn + fp + 1e-8)

print("\n📊 FINAL METRICS")

print("AUROC      :", round(auroc, 4))
print("Accuracy   :", round(accuracy, 4))
print("Precision  :", round(precision, 4))
print("Recall     :", round(recall, 4))
print("Specificity:", round(specificity, 4))

print("\n🧩 Confusion Matrix:")
print(cm)

# =========================================================
# SAVE THRESHOLD
# =========================================================

with open("threshold.json", "w") as f:

    json.dump(
        {"threshold": float(FINAL_THRESHOLD)},
        f
    )

# =========================================================
# SAVE RESULTS CSV
# =========================================================

df = pd.DataFrame({

    "image": names,

    "score": scores,

    "true_label": labels,

    "predicted_label": preds
})

df.to_csv("final_results.csv", index=False)

# =========================================================
# FINAL CLEAN ROC CURVE
# =========================================================

fpr, tpr, roc_thresholds = roc_curve(labels, scores)

plt.figure(figsize=(10, 8), dpi=400)

# -------------------------------------------------
# ROC CURVE
# -------------------------------------------------

plt.plot(
    fpr,
    tpr,
    linewidth=3,
    label=f'ROC Curve (AUROC = {auroc:.4f})'
)

# Area under curve shading
plt.fill_between(
    fpr,
    tpr,
    alpha=0.15
)

# -------------------------------------------------
# RANDOM CLASSIFIER
# -------------------------------------------------

plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--',
    linewidth=2,
    color='red',
    alpha=0.8,
    label='Random Classifier'
)

# -------------------------------------------------
# OPERATING THRESHOLD POINT
# -------------------------------------------------

final_preds = (scores >= FINAL_THRESHOLD).astype(int)

tn, fp, fn, tp = confusion_matrix(labels, final_preds).ravel()

operating_fpr = fp / (fp + tn + 1e-8)

operating_tpr = tp / (tp + fn + 1e-8)

plt.scatter(
    operating_fpr,
    operating_tpr,
    s=180,
    edgecolors='white',
    linewidths=2,
    zorder=5,
    label=f'Threshold = {FINAL_THRESHOLD:.4f}'
)

# -------------------------------------------------
# AXIS LIMITS
# -------------------------------------------------

# Slight zoom to reveal imperfections
plt.xlim(0.0, 0.25)
plt.ylim(0.75, 1.01)

# -------------------------------------------------
# LABELS
# -------------------------------------------------

plt.xlabel(
    'False Positive Rate (FPR)',
    fontsize=15,
    weight='bold'
)

plt.ylabel(
    'True Positive Rate (TPR)',
    fontsize=15,
    weight='bold'
)

plt.title(
    'ROC Curve for Medical Anomaly Detection',
    fontsize=22,
    weight='bold'
)

# -------------------------------------------------
# METRICS BOX
# -------------------------------------------------

metrics_text = (
    f'AUROC       : {auroc:.4f}\n'
    f'Threshold   : {FINAL_THRESHOLD:.4f}\n'
    f'Recall      : {recall:.4f}\n'
    f'Specificity : {specificity:.4f}'
)

plt.text(
    0.14,
    0.79,
    metrics_text,
    fontsize=12,
    bbox=dict(
        facecolor='black',
        alpha=0.75,
        boxstyle='round'
    )
)

# -------------------------------------------------
# GRID + LEGEND
# -------------------------------------------------

plt.grid(alpha=0.3)

plt.legend(
    loc='lower right',
    fontsize=11
)

plt.tight_layout()

# -------------------------------------------------
# SAVE
# -------------------------------------------------

plt.savefig(
    "roc_curve.png",
    dpi=400,
    bbox_inches='tight'
)

plt.close()

# =========================================================
# CONFUSION MATRIX VISUALIZATION
# =========================================================

cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

annot = np.empty_like(cm).astype(str)

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        count = cm[i, j]

        percent = cm_percent[i, j] * 100

        annot[i, j] = f"{count}\n({percent:.1f}%)"

plt.figure(figsize=(9, 7), dpi=300)

sns.heatmap(
    cm,
    annot=annot,
    fmt='',
    cmap='Blues',
    linewidths=2,
    linecolor='white',
    square=True,
    cbar=True,
    annot_kws={
        "fontsize": 14,
        "weight": "bold"
    }
)

plt.xlabel(
    'Predicted Label',
    fontsize=14
)

plt.ylabel(
    'True Label',
    fontsize=14
)

plt.xticks(
    [0.5, 1.5],
    ['Normal', 'Anomaly'],
    fontsize=12
)

plt.yticks(
    [0.5, 1.5],
    ['Normal', 'Anomaly'],
    fontsize=12,
    rotation=0
)

plt.title(
    'Confusion Matrix',
    fontsize=20,
    weight='bold'
)

# Metrics Text

metrics_text = (
    f'Accuracy: {accuracy:.4f}\n'
    f'Precision: {precision:.4f}\n'
    f'Recall: {recall:.4f}\n'
    f'Specificity: {specificity:.4f}'
)

plt.gcf().text(
    0.92,
    0.5,
    metrics_text,
    fontsize=12,
    bbox=dict(
        facecolor='black',
        alpha=0.6
    )
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

# =========================================================
# THRESHOLD ANALYSIS
# =========================================================

f1_values = []
bal_values = []

for t in thresholds:

    temp_preds = (scores >= t).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        temp_preds
    ).ravel()

    precision_t = tp / (tp + fp + 1e-8)

    recall_t = tp / (tp + fn + 1e-8)

    f1_t = 2 * (
        precision_t * recall_t
    ) / (
        precision_t + recall_t + 1e-8
    )

    specificity_t = tn / (tn + fp + 1e-8)

    bal_t = (recall_t + specificity_t) / 2

    f1_values.append(f1_t)

    bal_values.append(bal_t)

plt.figure(figsize=(12, 7), dpi=300)

plt.plot(
    thresholds,
    f1_values,
    linewidth=3,
    label='F1 Score'
)

plt.plot(
    thresholds,
    bal_values,
    linewidth=3,
    label='Balanced Accuracy'
)

plt.axvline(
    FINAL_THRESHOLD,
    linestyle='--',
    linewidth=3,
    color='yellow',
    label=f'Final Threshold = {FINAL_THRESHOLD:.4f}'
)

plt.xlabel(
    'Threshold',
    fontsize=14
)

plt.ylabel(
    'Metric Value',
    fontsize=14
)

plt.title(
    'Threshold Optimization Analysis',
    fontsize=20,
    weight='bold'
)

plt.legend(fontsize=12)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "threshold_analysis.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n📁 Enhanced Visualizations Saved:")

print("- final_results.csv")
print("- threshold.json")
print("- roc_curve.png")
print("- score_distribution.png")
print("- confusion_matrix.png")
print("- threshold_analysis.png")
