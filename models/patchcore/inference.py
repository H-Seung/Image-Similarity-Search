import os
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib
from config import PC_THRESHOLD_PATH


_pc_thresholds_cache = None

def _load_pc_thresholds():
    global _pc_thresholds_cache
    if _pc_thresholds_cache is not None:
        return _pc_thresholds_cache
    if os.path.exists(PC_THRESHOLD_PATH):
        with open(PC_THRESHOLD_PATH, "r") as f:
            _pc_thresholds_cache = json.load(f)
        print(f"threshold 데이터를 불러옵니다 : {PC_THRESHOLD_PATH}")
    else:
        _pc_thresholds_cache = {}
    return _pc_thresholds_cache

def load_patchcore_threshold(category):
    thresholds = _load_pc_thresholds()
    return thresholds.get(category, {}).get("threshold", float('inf'))


def run_patchcore_inference(filepath, category, mb_dir):
    from models.patchcore.model import PatchCoreFeatureExtractor, IMAGE_TRANSFORM
    extractor = PatchCoreFeatureExtractor(device='cpu')
    memory_bank, feature_shape = load_memory_bank(category, mb_dir)
    engine = PatchCoreInference(memory_bank, feature_shape)
    img = Image.open(filepath).convert('RGB')
    tensor = IMAGE_TRANSFORM(img).unsqueeze(0)
    patch_features, _ = extractor.extract(tensor)
    score, patch_scores = engine.score(patch_features)
    threshold = load_patchcore_threshold(category)
    status = "Anomaly" if score > threshold else "Normal"
    heatmap = engine.anomaly_map(patch_scores)
    return score, status, heatmap


def load_memory_bank(category, memory_bank_dir):
    path = os.path.join(memory_bank_dir, f"patchcore_{category}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Memory bank 없음: {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['memory_bank'], data['feature_shape']


class PatchCoreInference:
    def __init__(self, memory_bank: np.ndarray, feature_shape: tuple, k: int = 1):
        self.feature_shape = feature_shape      # (H, W) — 보통 (28, 28)
        self.knn = NearestNeighbors(
            n_neighbors=k, metric='euclidean',
            algorithm='auto', n_jobs=-1
        )
        self.knn.fit(memory_bank)

    def score(self, patch_features: np.ndarray):
        """이미지 level 이상 점수와 패치별 점수 반환"""
        distances, _ = self.knn.kneighbors(patch_features)
        patch_scores = distances[:, 0]
        return float(patch_scores.max()), patch_scores

    def anomaly_map(self, patch_scores: np.ndarray, output_size=(128, 128)):
        """패치 점수를 이미지 크기로 upsample → PIL heatmap 반환"""
        H, W = self.feature_shape
        score_map = patch_scores.reshape(H, W)

        min_v, max_v = score_map.min(), score_map.max()
        if max_v > min_v:
            score_map = (score_map - min_v) / (max_v - min_v)

        score_tensor = torch.tensor(score_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        score_up = F.interpolate(score_tensor, size=output_size,
                                 mode='bilinear', align_corners=False).squeeze().numpy()

        colored = matplotlib.colormaps['hot'](score_up)
        colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
        return Image.fromarray(colored_rgb)
