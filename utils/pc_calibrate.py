"""
PatchCore Memory Bank 생성 — K-Center Greedy Coreset Sampling
사용법: python utils/patchcore_calibrate.py
결과: models/patchcore/memory_bank/patchcore_{category}.pkl
"""
import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import pickle
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from models.patchcore.model import PatchCoreFeatureExtractor, IMAGE_TRANSFORM
from utils.common import get_available_categories

DATA_DIR      = os.path.join(ROOT_DIR, "data")
MODELS_DIR    = os.path.join(ROOT_DIR, "models")
CATEGORIES_PATH = os.path.join(ROOT_DIR, "models", "used_categories.json")
MB_DIR        = os.path.join(MODELS_DIR, "patchcore", "memory_bank")
IMAGE_EXTS    = ('.png', '.jpg', '.jpeg', '.bmp')
CORESET_RATIO = 0.10    # 전체 patch의 10% 선택

os.makedirs(MB_DIR, exist_ok=True)


def k_center_greedy(features: np.ndarray, n_samples: int) -> np.ndarray:
    """
    K-Center Greedy Coreset Sampling (논문: Towards Total Recall, Roth et al. 2022)
    sklearn + numpy BLAS 연산만 사용.
    features : (N, D)  float32
    n_samples: 선택할 coreset 크기
    """
    np.random.seed(42)
    
    n = len(features)
    n_samples = min(n_samples, n)

    selected = [int(np.random.randint(0, n))]
    min_dists_sq = np.full(n, np.inf, dtype=np.float32)

    # ||a||² 사전 계산 → 반복 내 distance 계산을 행렬곱으로 대체
    norms_sq = np.einsum('ij,ij->i', features, features)    # (N,)

    for _ in tqdm(range(n_samples - 1), desc="  K-Center Greedy", leave=False):
        c_idx = selected[-1]
        c_norm_sq = norms_sq[c_idx]
        # ||a - c||² = ||a||² + ||c||² - 2*(a·c)
        dists_sq = norms_sq + c_norm_sq - 2.0 * (features @ features[c_idx])
        np.maximum(dists_sq, 0, out=dists_sq)               # 수치 안정성
        np.minimum(min_dists_sq, dists_sq, out=min_dists_sq)
        selected.append(int(np.argmax(min_dists_sq)))

    return np.array(selected, dtype=np.int64)


def collect_features(category, extractor):
    train_dir = os.path.join(DATA_DIR, category, "train", "good")
    if not os.path.exists(train_dir):
        print(f"  ⚠️  {train_dir} 없음, 스킵")
        return None, None

    all_features, feature_shape = [], None
    image_files = [f for f in sorted(os.listdir(train_dir))
                   if f.lower().endswith(IMAGE_EXTS)]

    for fname in tqdm(image_files, desc=f"  feature 추출", leave=False):
        path = os.path.join(train_dir, fname)
        try:
            img = Image.open(path).convert('RGB')
            tensor = IMAGE_TRANSFORM(img).unsqueeze(0)
            feats, fshape = extractor.extract(tensor)
            all_features.append(feats)
            if feature_shape is None:
                feature_shape = fshape
        except Exception as e:
            print(f"  ⚠️  {fname}: {e}")

    if not all_features:
        return None, None
    return np.vstack(all_features).astype(np.float32), feature_shape


def main():
    categories = get_available_categories()
    print(f"Category: {categories}\n")

    extractor = PatchCoreFeatureExtractor(device='cpu')

    thresholds_dict = {}
    for cat in categories:
        out_path = os.path.join(MB_DIR, f"patchcore_{cat}.pkl")
        print(f"[{cat}] 처리 중...")
        all_features, feature_shape = collect_features(cat, extractor)
        if all_features is None:
            continue

        if os.path.exists(out_path):
            print(f"  memory bank 이미 존재, 로드...")
            with open(out_path, 'rb') as f:
                memory_bank = pickle.load(f)['memory_bank']
        else:
            n_total   = len(all_features)
            n_coreset = max(1, int(n_total * CORESET_RATIO))
            print(f"  전체 patch: {n_total:,}  →  coreset: {n_coreset:,} ({CORESET_RATIO*100:.0f}%)")
            indices     = k_center_greedy(all_features, n_coreset)
            memory_bank = all_features[indices]
            with open(out_path, 'wb') as f:
                pickle.dump({
                    'memory_bank':   memory_bank,
                    'feature_shape': feature_shape,
                    'n_total':       n_total,
                    'n_coreset':     len(memory_bank),
                }, f)
            print(f"  저장: {out_path}  memory_bank.shape={memory_bank.shape}\n")

        # 거리 계산 -> threshold 계산
        print(f"  knn distance 및 threshold 계산 중...")
        knn = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1)
        knn.fit(memory_bank)
        dists, _ = knn.kneighbors(all_features)
        threshold = float(np.percentile(dists[:, 0], 99))
        thresholds_dict[cat] = {"threshold": threshold}
        print(f"  threshold (p99): {threshold:.4f}")        


    thresh_path = os.path.join(MB_DIR, "thresholds.json")
    with open(thresh_path, "w") as f:
        json.dump(thresholds_dict, f, indent=2)
    print(f"✅ threshold 저장: {thresh_path}")

    print("✅ 완료")


if __name__ == "__main__":
    main()
