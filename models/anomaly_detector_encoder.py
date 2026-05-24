import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from models.anomaly_processor import load_and_preprocess_image


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

def load_model(category):
    model_path = f"models/autoencoder_{category}.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_path} not found")
    model = AutoEncoder()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def compute_anomaly_score(model, image_tensor):
    """
    image_tensor: shape (1, 3, 128, 128)
    """
    with torch.no_grad():
        output = model(image_tensor)
        loss = F.mse_loss(output, image_tensor).item()
    return loss, output

def run_anomaly_inference(filepath, category):
    """GUI 및 외부 영역에서 호출할 이상치 탐지 엔드포인트"""
    threshold = 0.0001

    model = load_model(category)
    tensor_original, tensor_preprocessed = load_and_preprocess_image(filepath, category)
    loss, output = compute_anomaly_score(model, tensor_original)

    results_anomaly = "Anomaly" if loss > threshold else "Normal"

    print(f"[{category}] 이미지: {filepath}")
    print(f"Anomaly Score: {loss:.6f}")
    print("Status:", results_anomaly)  # Threshold 조절 가능

    return loss, results_anomaly, output

