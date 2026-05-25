import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
import matplotlib
matplotlib.use('Agg') # tkinter 백엔드 충돌 방지
import matplotlib.cm as cm
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

def compute_error_heatmap(input_tensor, output_tensor):          
    """입력/복원 텐서 차이로 에러맵 PIL Image 생성"""
    error = torch.abs(input_tensor - output_tensor)              
    error_map = error.squeeze(0).mean(dim=0)                     # (H, W) 채널 평균
    error_np = error_map.cpu().detach().numpy()                  

    min_val, max_val = error_np.min(), error_np.max()            # 0~1 정규화
    if max_val > min_val:                                        
        error_np = (error_np - min_val) / (max_val - min_val)   

    colored = cm.get_cmap('hot')(error_np)                       # hot 컬러맵 적용 (오류 클수록 흰/노랑, 작을수록 검정)
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)    # RGBA → RGB

    heatmap = Image.fromarray(colored_rgb)                       
    heatmap = heatmap.filter(ImageFilter.GaussianBlur(radius=2)) # 노이즈 스무딩
    return heatmap                                               # PIL Image 반환

def load_threshold(category):                                   
    """thresholds.json에서 카테고리별 threshold 로드"""
    threshold_path = os.path.join(os.path.dirname(__file__), "thresholds.json")
    if os.path.exists(threshold_path):                            
        with open(threshold_path, "r") as f:                     
            thresholds = json.load(f)                            
        if category in thresholds:                               
            return thresholds[category]["threshold"]             
    return 0.005                                                 # calibrate 전 임시 fallback

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
    model = load_model(category)
    tensor_original, _ = load_and_preprocess_image(filepath, category)
    loss, output = compute_anomaly_score(model, tensor_original)

    threshold = load_threshold(category)                        # 카테고리별 threshold
    results_anomaly = "Anomaly" if loss > threshold else "Normal"

    heatmap = compute_error_heatmap(tensor_original, output)    # 에러맵 생성

    print(f"이미지: {filepath}")
    print(f"category: {category}")
    print(f"Anomaly Score: {loss:.6f}  (Threshold: {threshold:.6f})")
    print("Status:", results_anomaly)  

    return loss, results_anomaly, heatmap                       # tensor → PIL Image