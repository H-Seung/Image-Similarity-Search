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
    threshold = 0.004

    model = load_model(category)
    tensor_original, tensor_preprocessed = load_and_preprocess_image(filepath, category)
    loss, output = compute_anomaly_score(model, tensor_original)

    results_anomaly = "Anomaly" if loss > threshold else "Normal"

    print(f"[{category}] 이미지: {filepath}")
    print(f"Anomaly Score: {loss:.6f}")
    print("Status:", results_anomaly)  # Threshold 조절 가능

    return loss, results_anomaly


# 🚀 개발자가 이 모델 파일만 단독으로 실행해서 성능 검증하고 싶을 때 쓰는 블록
if __name__ == '__main__':
    def show_images(original_tensor, preprocessed_tensor, reconstructed_tensor, category, img_path, loss, threshold):
        """
        Args:
            original_tensor: (1, 3, H, W) - 전처리 전 이미지
            preprocessed_tensor: (1, 3, H, W) - 전처리 후 이미지
            reconstructed_tensor: (1, 3, H, W) - AE 복원 결과
        """
        to_pil = ToPILImage()
        original_img = to_pil(original_tensor.squeeze(0).cpu())
        preprocessed_img = to_pil(preprocessed_tensor.squeeze(0).cpu())
        reconstructed_img = to_pil(reconstructed_tensor.squeeze(0).cpu())

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(original_img); axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(preprocessed_img); axes[1].set_title("Preprocessed"); axes[1].axis("off")
        axes[2].imshow(reconstructed_img); axes[2].set_title("Reconstructed"); axes[2].axis("off")

        # 상단에 카테고리, 경로, 스코어, 판별 결과 표시
        status = "Anomaly" if loss > threshold else "Normal"
        fig.suptitle(
            f"[{category}] {img_path}\nAnomaly Score: {loss:.6f} | Status: {status}",
            fontsize=12,
            y=1.0
        )
        plt.tight_layout()
        plt.show()

    # 단독 테스트 수행
    test_img = "data/cable/test/good/008.png"
    test_cat = "cable"
    test_threshold = 0.001499

    print("--- 모델 단독 추론 테스트 시작 ---")
    loss_val, status_val = run_anomaly_inference(test_img, test_cat)
    print(f"결과 스코어: {loss_val:.6f} | 판정: {status_val}")
    
    # 디버깅용 시각화 창 띄우기
    m = load_model(test_cat)
    t_orig, t_prep = load_and_preprocess_image(test_img, test_cat)
    _, out_tensor = compute_anomaly_score(m, t_orig)
    show_images(t_orig, t_prep, out_tensor, test_cat, test_img, loss_val, test_threshold)