import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from model import SBCAE
from tqdm import tqdm

# ---------------------------
# CONFIG
# ---------------------------

DATA_DIR = r"D:\Project\TEST\TEST-01\data\test"
MODEL_PATH = "final_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# PREPROCESSING
# ---------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

# ---------------------------
# DATASET
# ---------------------------

class SimpleDataset(Dataset):

    def __init__(self, folder, label):

        self.paths = [
            os.path.join(folder,f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png",".jpg",".jpeg"))
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
    SimpleDataset(f"{DATA_DIR}/normal",0),
    batch_size=1,
    shuffle=False
)

anomaly_loader = DataLoader(
    SimpleDataset(f"{DATA_DIR}/anomaly",1),
    batch_size=1,
    shuffle=False
)

# ---------------------------
# LOAD MODEL
# ---------------------------

model = SBCAE(latent_dim=256).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE))
model.eval()

loss_fn = nn.L1Loss(reduction="mean")

# ---------------------------
# COMPUTE SCORES
# ---------------------------

normal_scores = []
anomaly_scores = []

names = []
labels = []

def compute_scores(loader, score_list):

    with torch.no_grad():

        for imgs, label, name in tqdm(loader):

            imgs = imgs.to(DEVICE)

            recon,_ = model(imgs)

            score = loss_fn(recon,imgs).item()

            score_list.append(score)

            names.append(name[0])
            labels.append(label.item())

compute_scores(normal_loader, normal_scores)
compute_scores(anomaly_loader, anomaly_scores)

normal_scores = np.array(normal_scores)
anomaly_scores = np.array(anomaly_scores)

scores = np.concatenate([normal_scores, anomaly_scores])
labels = np.array(labels)

# ---------------------------
# AUROC
# ---------------------------

auroc = roc_auc_score(labels, scores)

# ---------------------------
# CALIBRATED THRESHOLD
# ---------------------------

mean_normal = normal_scores.mean()
std_normal = normal_scores.std()

threshold = mean_normal + 3 * std_normal

print("\nCalibration Statistics")
print("Normal mean:", mean_normal)
print("Normal std :", std_normal)
print("Calibrated Threshold:", threshold)

# ---------------------------
# PREDICTIONS
# ---------------------------

predictions = (scores >= threshold).astype(int)

# ---------------------------
# METRICS
# ---------------------------

cm = confusion_matrix(labels,predictions)

tn,fp,fn,tp = cm.ravel()

accuracy = accuracy_score(labels,predictions)

precision = tp/(tp+fp+1e-8)
recall = tp/(tp+fn+1e-8)
specificity = tn/(tn+fp+1e-8)

# ---------------------------
# SAVE RESULTS
# ---------------------------

results = pd.DataFrame({
    "image_id": names,
    "anomaly_score": scores,
    "true_label": labels,
    "predicted_label": predictions
})

results.to_csv("final_results.csv",index=False)

# ---------------------------
# PRINT RESULTS
# ---------------------------

print("\nEvaluation Complete")
print("AUROC:",round(auroc,4))
print("Threshold:",threshold)
print("Accuracy:",round(accuracy,4))
print("Precision:",round(precision,4))
print("Recall:",round(recall,4))
print("Specificity:",round(specificity,4))
print("Results saved to final_results.csv")