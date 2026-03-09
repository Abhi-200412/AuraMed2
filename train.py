import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import StepLR
from model import SBCAE
from tqdm import tqdm

# ---------------------------
# CONFIG
# ---------------------------
DATA_DIR =  r"D:\Project\aura-med\data\train"
BATCH_SIZE = 8
EPOCHS = 80
LR = 1e-4
LATENT_LAMBDA = 1e-3
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
dataset = ImageFolder(DATA_DIR, transform=transform)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

# ---------------------------
# MODEL
# ---------------------------
model = SBCAE(latent_dim=256).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=1e-5
)

scheduler = StepLR(
    optimizer,
    step_size=25,
    gamma=0.5
)

loss_fn = nn.L1Loss()

# ---------------------------
# TRAIN LOOP
# ---------------------------
for epoch in range(EPOCHS):

    model.train()
    running_loss = 0

    for imgs, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

        imgs = imgs.to(DEVICE)

        recon, z = model(imgs)

        recon_loss = loss_fn(recon, imgs)
        latent_loss = torch.mean(z ** 2)

        loss = recon_loss + LATENT_LAMBDA * latent_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    avg_loss = running_loss / len(loader)

    print(f"Epoch {epoch+1} Loss: {avg_loss:.5f}")

# ---------------------------
# SAVE MODEL
# ---------------------------
torch.save(model.state_dict(), "final_model.pth")

print("\n✅ Training Complete")
print("Model saved as final_model.pth")