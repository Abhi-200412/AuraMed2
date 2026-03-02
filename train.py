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
DATA_DIR = "data/train"
BATCH_SIZE = 8          # smaller batch = stronger bottleneck
EPOCHS = 80             # fewer epochs needed
LR = 1e-4
LATENT_LAMBDA = 1e-3    # 🔥 key parameter
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# DATA
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = ImageFolder(DATA_DIR, transform=transform)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,   # Windows safe
    pin_memory=False
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

recon_loss_fn = nn.L1Loss()

# ---------------------------
# TRAINING LOOP
# ---------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for imgs, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs = imgs.to(DEVICE)

        recon, z = model(imgs)

        # Reconstruction loss
        recon_loss = recon_loss_fn(recon, imgs)

        # 🔒 Bottleneck regularization
        latent_loss = torch.mean(z ** 2)

        loss = recon_loss + LATENT_LAMBDA * latent_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {avg_loss:.5f}")

# ---------------------------
# SAVE FINAL MODEL
# ---------------------------
torch.save(model.state_dict(), "sbcae_final.pth")

print("\n✅ Training complete")
print("📦 Model saved as sbcae_final.pth")