import torch
import clip
from PIL import Image
import os


def test_clip_setup():
    """CLIP 모델 로딩 및 기본 기능 테스트"""
    print("🔧 Testing CLIP setup...")

    # CUDA 사용 가능 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    try:
        # CLIP 모델 로딩
        model, preprocess = clip.load("ViT-B/32", device=device)
        print("✅ CLIP model loaded successfully")

        # 더미 이미지로 테스트
        dummy_image = Image.new('RGB', (224, 224), color='red')
        image_tensor = preprocess(dummy_image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model.encode_image(image_tensor)
            embedding = embedding / embedding.norm()

        print(f"✅ Embedding shape: {embedding.shape}")
        print(f"✅ Embedding sample: {embedding[0][:5]}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_image_similarity():
    """두 이미지 간 유사도 테스트"""
    print("\n🔍 Testing image similarity...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 테스트용 이미지 2개 생성
    img1 = Image.new('RGB', (224, 224), color='red')
    img2 = Image.new('RGB', (224, 224), color='blue')
    img3 = Image.new('RGB', (224, 224), color='red')  # img1과 같은 색

    images = [img1, img2, img3]
    embeddings = []

    for i, img in enumerate(images):
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(img_tensor)
            embedding = embedding / embedding.norm()
            embeddings.append(embedding)

    # 유사도 계산
    sim_12 = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1])
    sim_13 = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[2])

    print(f"Red vs Blue similarity: {sim_12.item():.4f}")
    print(f"Red vs Red similarity: {sim_13.item():.4f}")
    print("✅ Similarity test completed")


if __name__ == "__main__":
    print("🚀 Starting CLIP test...\n")

    if test_clip_setup():
        test_image_similarity()
        print("\n🎉 All tests passed! Your setup is working.")
    else:
        print("\n❌ Setup failed. Please check your installation.")