import torch
from torchvision import models, transforms
from PIL import Image
import clip
import sys
import os

class Embedder:
    """
    CLIP/ResNet 임베딩 추출기
    """
    def __init__(self, model_name="clip", device="cpu"):
        self.device = device
        self.model_name = model_name.lower()

        if self.model_name == "clip":
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        elif self.model_name == "resnet":
            model = models.resnet50(pretrained=True)
            self.model = torch.nn.Sequential(*list(model.children())[:-1]).to(self.device).eval()
            self.preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def get_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.model_name == "clip":
                vec = self.model.encode_image(img_tensor)
            else:
                vec = self.model(img_tensor).squeeze()

        vec = vec / vec.norm()  # 정규화
        return vec.cpu()


# 테스트 실행 코드 (직접 실행 시)
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python embedder.py path/to/image.jpg [clip|resnet]")
        sys.exit(1)

    image_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "clip"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    embedder = Embedder(model_name=model_name, device=device)
    embedding = embedder.get_embedding(image_path)

    print(f"\n✅ Embedding for '{image_path}' using {model_name.upper()}:\n")
    print(embedding)
    print(f"\n▶ Shape: {embedding.shape}")