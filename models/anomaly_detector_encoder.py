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
from config import THRESHOLD_PATH


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


_thresholds_cache = None

def _load_thresholds():
    global _thresholds_cache
    if _thresholds_cache is not None:
        return _thresholds_cache
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, "r") as f:
            _thresholds_cache = json.load(f)
        print("threshold 데이터를 불러옵니다 : {THRESHOLD_PATH}")
    else:
        _thresholds_cache = {}
    return _thresholds_cache

def load_threshold(category):                                   
    """config.json에 지정된 경로에서 카테고리별 threshold 로드"""
    thresholds = _load_thresholds()
    return thresholds.get(category, {}).get("threshold", 0.005)

def compute_anomaly_score(model, image_tensor):
    """image_tensor: shape (1, 3, 128, 128)"""
    with torch.no_grad():
        output = model(image_tensor)
        loss = F.mse_loss(output, image_tensor).item()
    return loss, output

def compute_patch_anomaly_score(model, image_tensor, patch_size=16):
    with torch.no_grad():
        output = model(image_tensor)
    error_map = ((image_tensor - output) ** 2).mean(dim=1, keepdim=True)  # (1,1,128,128)
    patches = error_map.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size) # patches.shape = (1, 1, n_h, n_w, patch_size, patch_size) = (1, 1, 8, 8, 16, 16) = 16 x 16 크기의 patch 64개 (8x8 grid)
    patch_scores = patches.contiguous().view(1, -1, patch_size * patch_size).mean(dim=-1) # (1, 64, 256) (mean)-> (1, 64)
    score = patch_scores.max().item()  # 가장 이상한 패치 점수
    return score, output

def run_anomaly_inference(filepath, category):
    """GUI 및 외부 영역에서 호출할 이상치 탐지 엔드포인트"""
    model = load_model(category)
    tensor_original, _ = load_and_preprocess_image(filepath, category)
    # loss, output = compute_anomaly_score(model, tensor_original)
    loss, output = compute_patch_anomaly_score(model, tensor_original)

    threshold = load_threshold(category)                        # 카테고리별 threshold
    results_anomaly = "Anomaly" if loss > threshold else "Normal"

    heatmap = compute_error_heatmap(tensor_original, output)    # 에러맵 생성

    print(f"이미지: {filepath}")
    print(f"category: {category}")
    print(f"Anomaly Score: {loss:.6f}  (Threshold: {threshold:.6f})")
    print("Status:", results_anomaly)  

    return loss, results_anomaly, heatmap                       # tensor → PIL Image