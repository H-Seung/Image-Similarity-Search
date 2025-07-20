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
            model = models.resnet50(weights='ResNet50_Weights.DEFAULT')  # pretrained=True 대신 사용
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
        """이미지 임베딩 추출 (오류 처리 강화)"""
        try:
            # 이미지 로드 및 변환
            image = Image.open(image_path).convert("RGB")
            img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                if self.model_name == "clip":
                    vec = self.model.encode_image(img_tensor)
                else:
                    vec = self.model(img_tensor).squeeze()

            # 차원 정리 - 1D 벡터로 만들기
            if vec.dim() > 1:
                vec = vec.squeeze()

            # 정규화 (zero vector 방지)
            vec_norm = vec.norm()
            if vec_norm > 0:
                vec = vec / vec_norm
            else:
                print(f"⚠️ Zero vector detected for {image_path}")
                # 작은 랜덤 벡터로 대체
                vec = torch.randn_like(vec) * 0.01
                vec = vec / vec.norm()

            # 최종적으로 1D 벡터 보장
            vec = vec.flatten()
            return vec.cpu()

        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            raise


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

    try:
        embedder = Embedder(model_name=model_name, device=device)
        embedding = embedder.get_embedding(image_path)

        print(f"\n✅ Embedding for '{image_path}' using {model_name.upper()}:")
        print(f"▶ Shape: {embedding.shape}")
        print(f"▶ Norm: {embedding.norm():.6f}")

    except Exception as e:
        print(f"❌ Failed to process image: {e}")
        sys.exit(1)