from torchvision import transforms
import cv2
import numpy as np
from PIL import Image

def apply_clahe(pil_img: Image.Image) -> Image.Image:
    if not isinstance(pil_img, Image.Image):
        # 혹시 Tensor가 들어와도 안전하게 변환
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

def get_transform(category: str):

    if category == 'cable':
        """cable : Clahe, 조명·대비, 정규화 """
        steps = [
            transforms.Resize((128, 128)),
            transforms.Lambda(apply_clahe),
            transforms.ColorJitter(brightness=0.1, contrast=0.15),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3), # 정규화 : 모델 학습이 잘 되게 하려고 사용(0 ~ 255)
        ]
    else:
        steps = [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    return transforms.Compose(steps)

