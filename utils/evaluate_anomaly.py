"""
카테고리별 AUROC 정량 평가 스크립트
사용법: python utils/evaluate_anomaly.py
"""
import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from models.anomaly_detector_encoder import AutoEncoder, compute_anomaly_score, compute_patch_anomaly_score

DATA_DIR   = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def get_available_categories():
    cats = []
    for fname in os.listdir(MODELS_DIR):
        if fname.startswith("autoencoder_") and fname.endswith(".pth") and "_prep" not in fname:
            cats.append(fname[len("autoencoder_"):-len(".pth")])
    return sorted(cats)


def evaluate_category(category):
    model_path = os.path.join(MODELS_DIR, f"autoencoder_{category}.pth")
    model = AutoEncoder()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    test_dir = os.path.join(DATA_DIR, category, "test")
    if not os.path.exists(test_dir):
        print(f"  ⚠️  {test_dir} 없음, 스킵")
        return None, None

    labels, scores = [], []

    for defect_type in sorted(os.listdir(test_dir)):
        defect_dir = os.path.join(test_dir, defect_type)
        if not os.path.isdir(defect_dir):
            continue
        label = 0 if defect_type == "good" else 1  # good=정상, 나머지=이상

        for fname in tqdm(os.listdir(defect_dir), desc=f"  [{category}/{defect_type}]", leave=False):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            path = os.path.join(defect_dir, fname)
            try:
                img = Image.open(path).convert('RGB')
                tensor = transform(img).unsqueeze(0)
                # score, _ = compute_anomaly_score(model, tensor)
                score, _ = compute_patch_anomaly_score(model, tensor)
                labels.append(label)
                scores.append(score)
            except Exception as e:
                print(f"  ⚠️  {fname}: {e}")

    return np.array(labels), np.array(scores)


def main():
    categories = get_available_categories()
    print(f"발견된 카테고리: {categories}\n")

    results = {}
    for cat in categories:
        print(f"[{cat}] 평가 중...")
        labels, scores = evaluate_category(cat)
        if labels is None or len(np.unique(labels)) < 2:
            print(f"  ⚠️  정상/이상 샘플 모두 필요, 스킵\n")
            continue

        auroc = roc_auc_score(labels, scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        precision = tpr / (tpr + fpr + 1e-8)
        f1 = 2 * precision * tpr / (precision + tpr + 1e-8)
        best_idx = np.argmax(f1)
        best_thresh = thresholds[best_idx]

        results[cat] = {"auroc": auroc, "best_threshold": best_thresh, "best_f1": f1[best_idx]}
        print(f"  AUROC={auroc:.4f}  best_F1={f1[best_idx]:.4f}  best_thresh={best_thresh:.6f}\n")

    print("=" * 55)
    print(f"{'Category':<15} {'AUROC':>8} {'Best F1':>9} {'Best Thresh':>13}")
    print("-" * 55)
    for cat, r in results.items():
        print(f"{cat:<15} {r['auroc']:>8.4f} {r['best_f1']:>9.4f} {r['best_threshold']:>13.6f}")
    print("=" * 55)

    mean_auroc = np.mean([r['auroc'] for r in results.values()])
    print(f"\n평균 AUROC: {mean_auroc:.4f}")


if __name__ == "__main__":
    main()
