import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)  # 64x64
        self.enc2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 32x32
        self.enc3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 16x16

        # Decoder
        self.dec1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 32x32
        self.dec2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 64x64
        self.dec3 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)     # 128x128

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))

        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = torch.sigmoid(self.dec3(x))  # Normalize output
        return x