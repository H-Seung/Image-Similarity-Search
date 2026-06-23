"""
PatchCore AUROC 정량 평가 및 시각화 (score 저장)
사용법: python utils/patchcore_evaluate.py
"""
import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from models.patchcore.model import PatchCoreFeatureExtractor, IMAGE_TRANSFORM
from models.patchcore.inference import PatchCoreInference, load_memory_bank
from utils.common import get_available_categories, unique_path

DATA_DIR   = os.path.join(ROOT_DIR, "data")
MB_DIR     = os.path.join(ROOT_DIR, "models", "patchcore", "memory_bank")
RESULTS_DIR = os.path.join(ROOT_DIR, "results", "patchcore")
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_category(category, extractor):
    memory_bank, feature_shape = load_memory_bank(category, MB_DIR)
    engine = PatchCoreInference(memory_bank, feature_shape)

    test_dir = os.path.join(DATA_DIR, category, "test")
    if not os.path.exists(test_dir):
        return None, None

    labels, scores = [], []
    for defect_type in sorted(os.listdir(test_dir)):
        defect_dir = os.path.join(test_dir, defect_type)
        if not os.path.isdir(defect_dir):
            continue
        label = 0 if defect_type == "good" else 1

        for fname in tqdm(os.listdir(defect_dir),
                          desc=f"  [{category}/{defect_type}]", leave=False):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            try:
                img = Image.open(os.path.join(defect_dir, fname)).convert('RGB')
                tensor = IMAGE_TRANSFORM(img).unsqueeze(0)
                patch_features, _ = extractor.extract(tensor)
                score, _ = engine.score(patch_features)
                labels.append(label)
                scores.append(score)
            except Exception as e:
                print(f"  ⚠️  {fname}: {e}")

    return np.array(labels), np.array(scores)


# 시각화
def plot_score_distributions(all_data):
    cats = list(all_data.keys())
    cols = 4
    rows = (len(cats) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3.5))
    axes = axes.flatten()
    for i, cat in enumerate(cats):
        labels, scores = all_data[cat]
        auroc = roc_auc_score(labels, scores)
        ax = axes[i]
        sns.kdeplot(scores[labels == 0], ax=ax, color='steelblue', fill=True, alpha=0.4, label='Normal')
        sns.kdeplot(scores[labels == 1], ax=ax, color='tomato',    fill=True, alpha=0.4, label='Anomaly')
        ax.set_title(f'{cat}  (AUROC={auroc:.4f})', fontsize=10)
        ax.set_xlabel('PatchCore Distance Score (density)', fontsize=8)
        ax.legend(fontsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Score Distribution: Normal vs Anomaly (PatchCore)', fontsize=13)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, 'score_distributions.png')
    out = unique_path(out)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"저장: {out}")


def plot_roc_curves(all_data):
    fig, ax = plt.subplots(figsize=(8, 7))
    for cat, (labels, scores) in all_data.items():
        fpr, tpr, _ = roc_curve(labels, scores)
        auroc = roc_auc_score(labels, scores)
        ax.plot(fpr, tpr, label=f'{cat} ({auroc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves per Category (PatchCore)')
    ax.legend(loc='lower right', fontsize=9)
    out = os.path.join(RESULTS_DIR, 'roc_curves.png')
    out = unique_path(out)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"저장: {out}")



def main():
    categories = get_available_categories()
    print(f"카테고리: {categories}\n")

    extractor = PatchCoreFeatureExtractor(device='cpu')
    results = {}
    all_data = {}

    for cat in categories:
        print(f"[{cat}] 평가 중...")
        labels, scores = evaluate_category(cat, extractor)
        if labels is None or len(np.unique(labels)) < 2:
            print(f"  ⚠️  스킵\n")
            continue

        auroc = roc_auc_score(labels, scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        precision = tpr / (tpr + fpr + 1e-8)
        f1        = 2 * precision * tpr / (precision + tpr + 1e-8)
        best_idx  = np.argmax(f1)

        results[cat] = {
            'auroc':          auroc,
            'best_f1':        f1[best_idx],
            'best_threshold': thresholds[best_idx],
        }
        all_data[cat] = (labels, scores)
        print(f"  AUROC={auroc:.4f}  best_F1={f1[best_idx]:.4f}  "
              f"best_thresh={thresholds[best_idx]:.4f}\n")

    print("=" * 60)
    print(f"{'Category':<15} {'AUROC':>8} {'Best F1':>9} {'Best Thresh':>13}")
    print("-" * 60)
    for cat, r in results.items():
        print(f"{cat:<15} {r['auroc']:>8.4f} {r['best_f1']:>9.4f} "
              f"{r['best_threshold']:>13.4f}")
    print("=" * 60)
    print(f"\n평균 AUROC: {np.mean([r['auroc'] for r in results.values()]):.4f}")

    # 시각화
    plot_score_distributions(all_data)
    plot_roc_curves(all_data)

    # scores 저장
    save_path = os.path.join(RESULTS_DIR, 'pc_scores.npz')
    save_path = unique_path(save_path)
    np.savez(save_path,
             **{f'{cat}_labels': v[0] for cat, v in all_data.items()},
             **{f'{cat}_scores': v[1] for cat, v in all_data.items()})
    print(f"저장: {save_path}")


if __name__ == "__main__":
    main()
