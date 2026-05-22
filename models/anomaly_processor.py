import os
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image

def apply_clahe(pil_img: Image.Image) -> Image.Image:
    """이미지에 CLAHE(기반 대비 향상) 필터 적용"""
    if not isinstance(pil_img, Image.Image):
        # Tensor가 들어와도 안전하게 변환
        pil_img = transforms.ToPILImage()(pil_img)
    img = np.array(pil_img)
    if img.ndim == 2:  # grayscale safety
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb)

def get_category_transform(category: str):
    """카테고리별 특수 전처리 정의 (ToTensor는 여기서 제외하여 PIL 상태 유지)"""
    if category == 'cable':
        """cable : Clahe, 조명·대비, 정규화 """
        steps = [
            transforms.Lambda(apply_clahe),
            transforms.ColorJitter(brightness=0.1, contrast=0.15),
            # transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3), # 정규화 : 모델 학습이 잘 되게 하려고 사용(0 ~ 255)
        ]
    else:
        steps = [
            transforms.Resize((128, 128)),
            # transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    return transforms.Compose(steps)

def load_and_preprocess_image(image_path: str, category: str):
    """GUI 및 추론에서 호출할 메인 함수: 원본 및 전처리 텐서 반환"""
    pil_img = Image.open(image_path).convert('RGB')

    base_resize = [transforms.Resize((128, 128))]
    to_tensor = [transforms.ToTensor()]

    # 카테고리별 특수 전처리 단계 가져오기
    category_steps = get_category_transform(category).transforms

    # 3) 텐서 변환 및 배치 차원 추가 (1, C, H, W)
    tensor_original = transforms.Compose(base_resize + to_tensor)(pil_img).unsqueeze(0) # (1, C, H, W) 형태
    tensor_preprocessed = transforms.Compose(base_resize+ category_steps + to_tensor)(pil_img).unsqueeze(0)

    return tensor_original, tensor_preprocessed
