"""
모든 카테고리의 train/good 이미지 전체/패치에 대해 MSE 분포를 계산하고
카테고리별 threshold를 models/thresholds.json에 저장합니다.

사용법: python utils/calibrate_thresholds.py
"""
import os
import sys                                                             
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 프로젝트 루트
sys.path.insert(0, ROOT_DIR)                                           
import json
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from models.anomaly_detector_encoder import AutoEncoder, compute_anomaly_score, compute_patch_anomaly_score 

DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
THRESHOLD_JSON = os.path.join(MODELS_DIR, "thresholds_patch.json")
PERCENTILE = 99


def get_available_categories():
    categories = []
    for fname in os.listdir(MODELS_DIR):
        if fname.startswith("autoencoder_") and fname.endswith(".pth") and "_prep" not in fname:
            cat = fname[len("autoencoder_"):-len(".pth")]
            categories.append(cat)
    return sorted(categories)


def compute_train_scores(category):
    model_path = os.path.join(MODELS_DIR, f"autoencoder_{category}.pth")
    model = AutoEncoder()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    train_dir = os.path.join(DATA_DIR, category, "train", "good")
    if not os.path.exists(train_dir):
        print(f"  ⚠️  {train_dir} 없음, 스킵")
        return None

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    image_files = [
        f for f in os.listdir(train_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ]

    scores = []
    for fname in tqdm(image_files, desc=f"  [{category}]", leave=False):
        path = os.path.join(train_dir, fname)
        try:
            img = Image.open(path).convert('RGB')
            tensor = transform(img).unsqueeze(0)
            # score, _ = compute_anomaly_score(model, tensor)
            score, _ = compute_patch_anomaly_score(model, tensor)
            scores.append(score)
        except Exception as e:
            print(f"  ⚠️  {fname}: {e}")

    return scores


def main():
    categories = get_available_categories()
    print(f"발견된 카테고리: {categories}\n")

    thresholds = {}

    for cat in categories:
        print(f"[{cat}] 점수 계산 중...")
        scores = compute_train_scores(cat)
        if not scores:
            continue

        arr = np.array(scores)
        thresholds[cat] = {
            "threshold": float(np.percentile(arr, PERCENTILE)),
            "mean":      float(arr.mean()),
            "std":       float(arr.std()),
            "p95":       float(np.percentile(arr, 95)),
            "p99":       float(np.percentile(arr, 99)),
            "p999":      float(np.percentile(arr, 99.9)),
            "n_samples": len(scores),
        }
        print(f"  n={len(scores)}  mean={arr.mean():.6f}  "
              f"std={arr.std():.6f}  p99={np.percentile(arr, 99):.6f}\n")

    with open(THRESHOLD_JSON, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"✅ 저장 완료: {THRESHOLD_JSON}")


if __name__ == "__main__":
    main()
