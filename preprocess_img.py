import os
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


# get_transform을 이용해 전처리 이미지를 만들고 저장하는 코드 (위치 정리 필요)
def save_tensor_as_image(tensor, save_path):
    """Tensor(C,H,W) → PIL 이미지로 변환 후 저장"""
    to_pil = transforms.ToPILImage()
    img = to_pil(tensor.cpu())
    img.save(save_path)

def process_images(input_dir, output_dir, category):
    os.makedirs(output_dir, exist_ok=True)
    transform = get_transform(category)

    count = 0
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            continue

        img_path = os.path.join(input_dir, fname)
        pil_img = Image.open(img_path).convert('RGB')

        # get_transform 적용
        tensor_preprocessed = transform(pil_img)

        # 전처리 저장
        save_tensor_as_image(
            tensor_preprocessed,
            os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_prep.png"))

        count += 1

    print(f"✅ 총 {count}개 이미지 처리 완료 → {output_dir}")


if __name__ == '__main__':
    category = "cable"
    process_images(
        f"data/{category}/train/good",
        f"data_processed/{category}/train",
        category=category
    )
