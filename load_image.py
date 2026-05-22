from PIL import Image
import torchvision.transforms as transforms
from preprocess_img import get_transform

# Load image with preprocessing
def load_image(image_path, category):
    # 1) PIL 이미지 로드
    pil_img = Image.open(image_path).convert('RGB')

    # 2) 기본 및 카테고리별 전처리 정의
    base_resize = [transforms.Resize((128, 128))]
    to_tensor = [transforms.ToTensor()]

    preprocess = [
        *get_transform(category).transforms
    ]

    # 3) tensor 및 변환 적용
    tensor_original = transforms.Compose(base_resize + to_tensor)(pil_img).unsqueeze(0) # (1, C, H, W) 형태
    tensor_preprocessed = transforms.Compose(base_resize + preprocess )(pil_img).unsqueeze(0)

    return tensor_original, tensor_preprocessed

