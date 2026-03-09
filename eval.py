import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    accuracy_score
)
from model import SBCAE
from tqdm import tqdm

# ---------------------------
# CONFIG
# ---------------------------
DATA_DIR = r"D:\Project\aura-med\data\test"
MODEL_PATH = "sbcae_final.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# TRANSFORM
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ---------------------------
# DATASET
# ---------------------------
class SimpleDataset(Dataset):
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
        return transform(img), self.label, os.path.basename(self.paths[idx])

# ---------------------------
# LOAD DATA
# ---------------------------
normal_ds = SimpleDataset(f"{DATA_DIR}/normal", label=0)
anomaly_ds = SimpleDataset(f"{DATA_DIR}/anomaly", label=1)

normal_loader = DataLoader(normal_ds, batch_size=1, shuffle=False)
anomaly_loader = DataLoader(anomaly_ds, batch_size=1, shuffle=False)

# ---------------------------
# MODEL
# ---------------------------
model = SBCAE(latent_dim=256).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

l1 = nn.L1Loss(reduction="mean")

# ---------------------------
# SCORING
# ---------------------------
all_scores = []
all_labels = []
all_names = []

def process_loader(loader):
    with torch.no_grad():
        for imgs, labels, names in tqdm(loader):
            imgs = imgs.to(DEVICE)
            recon, _ = model(imgs)
            score = l1(recon, imgs).item()

            all_scores.append(score)
            all_labels.append(labels.item())
            all_names.append(names[0])

process_loader(normal_loader)
process_loader(anomaly_loader)

y_scores = np.array(all_scores)
y_true = np.array(all_labels)

# ---------------------------
# AUROC & THRESHOLD
# ---------------------------
auroc = roc_auc_score(y_true, y_scores)

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]

y_pred = (y_scores >= best_threshold).astype(int)

# ---------------------------
# METRICS
# ---------------------------
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

accuracy = accuracy_score(y_true, y_pred)
precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
specificity = tn / (tn + fp + 1e-8)

# ---------------------------
# SAVE SINGLE CSV
# ---------------------------
final_df = pd.DataFrame({
    "image_id": all_names,
    "anomaly_score": y_scores,
    "true_label": y_true,
    "predicted_label": y_pred,
    "AUROC": auroc,
    "Optimal_Threshold": best_threshold,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "Specificity": specificity
})

final_df.to_csv("final_results.csv", index=False)

# ---------------------------
# PRINT SUMMARY
# ---------------------------
print("\n✅ Evaluation Complete")
print(f"🔥 AUROC: {auroc:.4f}")
print(f"🎯 Optimal Threshold: {best_threshold:.6f}")
print("📁 Results saved to final_results.csv")