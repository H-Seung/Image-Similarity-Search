from PIL import Image
import torchvision.transforms as transforms
from models.anomaly_detector_encoder import load_model, compute_anomaly_score
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

# visualization
def show_original_vs_reconstructed(original_tensor, reconstructed_tensor):
    """
    Args:
        original_tensor: shape (1, 3, H, W)
        reconstructed_tensor: shape (1, 3, H, W)
    """
    to_pil = ToPILImage()

    original_img = to_pil(original_tensor.squeeze(0).cpu())
    reconstructed_img = to_pil(reconstructed_tensor.squeeze(0).cpu())

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_img)
    plt.title("Reconstructed")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Load image
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


if __name__ == '__main__':
    # 사용자 설정
    img_path = "data/carpet/test/good/003.png"
    category = "carpet"
    threshold = 0.0011

    model = load_model(category)
    img = load_image(img_path)
    loss, output = compute_anomaly_score(model, img)

    print(f"[{category}] 이미지: {img_path}")
    print(f"Anomaly Score: {loss:.6f}")
    print("Status:", "Anomaly" if loss > threshold else "Normal")  # Threshold 조절 가능

    # 시각화 추가
    show_original_vs_reconstructed(img, output)