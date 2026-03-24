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
    roc_auc_score, confusion_matrix,
    f1_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from model import SBCAE

# ---------------------------
# CONFIG
# ---------------------------

DATA_DIR = "data/test"
MODEL_PATH = "final_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# PREPROCESSING
# ---------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ---------------------------
# DATASET
# ---------------------------

class ImageDataset(Dataset):
    def __init__(self, folder, label):
        self.paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.label = label

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = transform(img)
        return img, self.label, os.path.basename(self.paths[idx])

# ---------------------------
# LOAD DATA
# ---------------------------

normal_loader = DataLoader(
    ImageDataset(f"{DATA_DIR}/normal", 0),
    batch_size=1, shuffle=False
)

anomaly_loader = DataLoader(
    ImageDataset(f"{DATA_DIR}/anomaly", 1),
    batch_size=1, shuffle=False
)

# ---------------------------
# LOAD MODEL
# ---------------------------

model = SBCAE(latent_dim=256).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

loss_fn = nn.L1Loss()

# ---------------------------
# COMPUTE SCORES
# ---------------------------

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

process(normal_loader)
process(anomaly_loader)

scores = np.array(scores)
labels = np.array(labels)

# ---------------------------
# AUROC
# ---------------------------

auroc = roc_auc_score(labels, scores)

# ---------------------------
# THRESHOLD OPTIMIZATION
# ---------------------------

thresholds = np.linspace(scores.min(), scores.max(), 300)

best_f1 = 0
best_f1_thresh = 0

best_youden = 0
best_youden_thresh = 0

best_bal_acc = 0
best_bal_thresh = 0

for t in thresholds:

    preds = (scores >= t).astype(int)

    # F1
    f1 = f1_score(labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_f1_thresh = t

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    tpr = tp / (tp + fn + 1e-8)
    fpr = fp / (fp + tn + 1e-8)

    # Youden Index
    youden = tpr - fpr
    if youden > best_youden:
        best_youden = youden
        best_youden_thresh = t

    # Balanced Accuracy
    bal_acc = (tpr + (1 - fpr)) / 2
    if bal_acc > best_bal_acc:
        best_bal_acc = bal_acc
        best_bal_thresh = t

# FINAL SELECTION
FINAL_THRESHOLD = best_bal_thresh

print("\n🔍 Threshold Optimization Results:")
print(f"F1 Threshold       : {best_f1_thresh:.4f}")
print(f"Youden Threshold   : {best_youden_thresh:.4f}")
print(f"Balanced Threshold : {best_bal_thresh:.4f}")
print(f"\n✅ FINAL THRESHOLD: {FINAL_THRESHOLD:.4f}")

# ---------------------------
# FINAL PREDICTIONS
# ---------------------------

preds = (scores >= FINAL_THRESHOLD).astype(int)

# ---------------------------
# METRICS
# ---------------------------

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

print("\nConfusion Matrix:")
print(cm)

# ---------------------------
# SAVE THRESHOLD
# ---------------------------

with open("threshold.json", "w") as f:
    json.dump({"threshold": float(FINAL_THRESHOLD)}, f)

# ---------------------------
# SAVE CSV
# ---------------------------

df = pd.DataFrame({
    "image": names,
    "score": scores,
    "true_label": labels,
    "predicted_label": preds
})

df.to_csv("final_results.csv", index=False)

# ---------------------------
# PLOTS
# ---------------------------

# ROC Curve
fpr, tpr, _ = roc_curve(labels, scores)
plt.figure()
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.savefig("roc_curve.png")
plt.close()

# Score Distribution
plt.figure()
plt.hist(scores[labels == 0], bins=50, alpha=0.5, label="Normal")
plt.hist(scores[labels == 1], bins=50, alpha=0.5, label="Anomaly")
plt.axvline(FINAL_THRESHOLD, linestyle="--", label="Threshold")
plt.legend()
plt.title("Score Distribution")
plt.savefig("score_distribution.png")
plt.close()

# Confusion Matrix
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

print("\n📁 Files Saved:")
print("- final_results.csv")
print("- threshold.json")
print("- roc_curve.png")
print("- score_distribution.png")
print("- confusion_matrix.png")