import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# BASIC BLOCK
# ---------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# ---------------------------
# STRONG BOTTLENECK AE
# ---------------------------
class SBCAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()

        # -------- Encoder --------
        self.enc1 = ConvBlock(3, 64)    # 224 → 112
        self.enc2 = ConvBlock(64, 128)  # 112 → 56
        self.enc3 = ConvBlock(128, 256) # 56 → 28
        self.enc4 = ConvBlock(256, 512) # 28 → 14

        self.flatten = nn.Flatten()
        self.fc_latent = nn.Linear(512 * 14 * 14, latent_dim)

        # -------- Decoder --------
        self.fc_decode = nn.Linear(latent_dim, 512 * 14 * 14)

        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)

        x = self.flatten(x)
        z = self.fc_latent(x)              # 🔒 bottleneck

        x = self.fc_decode(z)
        x = x.view(-1, 512, 14, 14)

        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)

        return x, z