import os
from PIL import Image
from torchvision import transforms
from models.autoencoder.anomaly_processor import get_category_transform

def save_tensor_as_image(tensor, save_path):
    """Tensor(C,H,W) → PIL 이미지로 변환 후 저장"""
    to_pil = transforms.ToPILImage()
    img = to_pil(tensor.cpu())
    img.save(save_path)

def batch_process_and_save(input_dir, output_dir, category):
    """학습 목적으로 특정 폴더의 이미지들을 전처리하여 결과 폴더에 저장하는 유틸리티"""
    os.makedirs(output_dir, exist_ok=True)

    # 통합된 processor에서 전처리 룰셋만 그대로 가져와 활용
    category_transform = get_category_transform(category)
    full_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        category_transform,
        transforms.ToTensor()
    ])

    count = 0
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            continue

        img_path = os.path.join(input_dir, fname)
        pil_img = Image.open(img_path).convert('RGB')

        # 전처리 적용 후 텐서 변환
        processed_tensor = full_transform(pil_img)

        # 저장
        save_path = os.path.join(output_dir, fname)
        save_tensor_as_image(processed_tensor, save_path)

        count += 1

    print(f"✅ category : [{category}], 총 {count}개 이미지 처리 완료 → {output_dir}")


if __name__ == '__main__':
    target_category = "cable"
    batch_process_and_save(
        f"data/{target_category}/train/good",
        f"data_processed/{target_category}/train",
        category=target_category
    )
