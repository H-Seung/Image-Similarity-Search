from PIL import Image
import torchvision.transforms as transforms
from models.anomaly_detector_encoder import load_model, compute_anomaly_score
from load_image import load_image
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

# visualization
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

    axes[0].imshow(original_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(preprocessed_img)
    axes[1].set_title("Preprocessed")
    axes[1].axis("off")

    axes[2].imshow(reconstructed_img)
    axes[2].set_title("Reconstructed")
    axes[2].axis("off")

    # 상단에 카테고리, 경로, 스코어, 판별 결과 표시
    status = "Anomaly" if loss > threshold else "Normal"
    fig.suptitle(
        f"[{category}] {img_path}\nAnomaly Score: {loss:.6f} | Status: {status}",
        fontsize=12,
        y=1.0
    )

    plt.tight_layout()
    plt.show()

def run_anomaly_inference(filepath, category):
    img_path = filepath
    category = category
    threshold = 0.004

    model = load_model(category)
    tensor_original, tensor_preprocessed = load_image(img_path, category)
    loss, output = compute_anomaly_score(model, tensor_original)
    results_anomaly = None

    if loss > threshold :
        results_anomaly = "Anomaly"
    else:
        results_anomaly = "Normal"

    print(f"[{category}] 이미지: {img_path}")
    print(f"Anomaly Score: {loss:.6f}")
    print("Status:", "Anomaly" if loss > threshold else "Normal")  # Threshold 조절 가능

    return loss, results_anomaly

if __name__ == '__main__':
    # 사용자 설정
    # img_path = "data/cable/test/poke_insulation/005.png"
    img_path = "data/cable/test/good/008.png"
    category = "cable"
    threshold = 0.001499

    model = load_model(category)
    tensor_original, tensor_preprocessed = load_image(img_path, category)
    loss, output = compute_anomaly_score(model, tensor_original)

    print(f"[{category}] 이미지: {img_path}")
    print(f"Anomaly Score: {loss:.6f}")
    print("Status:", "Anomaly" if loss > threshold else "Normal")  # Threshold 조절 가능

    # 시각화 (원본, 전처리, 복원)
    show_images(tensor_original, tensor_preprocessed, output, category, img_path, loss, threshold)