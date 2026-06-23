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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from models.autoencoder.inference import AutoEncoder, compute_anomaly_score, compute_patch_anomaly_score
from utils.common import get_available_categories, unique_path

DATA_DIR   = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
CATEGORIES_PATH = os.path.join(ROOT_DIR, "models", "used_categories.json")
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')
RESULTS_DIR = os.path.join(ROOT_DIR, "results", "autoencoder")
os.makedirs(RESULTS_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def evaluate_category(category):
    model_path = os.path.join(MODELS_DIR, f"autoencoder_{category}.pth")
    model = AutoEncoder()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    test_dir = os.path.join(DATA_DIR, category, "test")
    if not os.path.exists(test_dir):
        print(f"  ⚠️  {test_dir} 없음, 스킵")
        return None, None

    labels, global_scores, patch_scores = [], [], []

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
                g_score, _ = compute_anomaly_score(model, tensor)
                p_score, _ = compute_patch_anomaly_score(model, tensor)
                labels.append(label)
                global_scores.append(g_score)
                patch_scores.append(p_score)
            except Exception as e:
                print(f"  ⚠️  {fname}: {e}")

    return np.array(labels), np.array(global_scores), np.array(patch_scores)


# 시각화
def plot_score_distributions(all_data, title, xlabel, out_path):
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
        ax.set_xlabel(xlabel, fontsize=8)
        ax.legend(fontsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"저장: {out_path}")


def plot_roc_curves(all_data, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    for cat, (labels, scores) in all_data.items():
        fpr, tpr, _ = roc_curve(labels, scores)
        auroc = roc_auc_score(labels, scores)
        ax.plot(fpr, tpr, label=f'{cat} ({auroc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right', fontsize=9)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"저장: {out_path}")


def main():
    categories = get_available_categories()
    print(f"발견된 카테고리: {categories}\n")

    results = {}
    all_data_global = {}
    all_data_patch  = {}

    for cat in categories:
        print(f"[{cat}] 평가 중...")
        labels, global_scores, patch_scores = evaluate_category(cat)
        if labels is None or len(np.unique(labels)) < 2:
            print(f"  ⚠️  정상/이상 샘플 모두 필요, 스킵\n")
            continue

        auroc_g = roc_auc_score(labels, global_scores)
        auroc_p = roc_auc_score(labels, patch_scores)
        fpr_p, tpr_p, thresholds_p = roc_curve(labels, patch_scores)
        precision_p = tpr_p / (tpr_p + fpr_p + 1e-8)
        f1_p = 2 * precision_p * tpr_p / (precision_p + tpr_p + 1e-8)
        best_idx_p = np.argmax(f1_p)

        results[cat] = {
            "auroc_global": auroc_g,
            "auroc_patch":  auroc_p,
            "best_threshold": thresholds_p[best_idx_p],
            "best_f1": f1_p[best_idx_p],
        }
        all_data_global[cat] = (labels, global_scores)
        all_data_patch[cat]  = (labels, patch_scores)
        print(f"  AUROC(global)={auroc_g:.4f}  AUROC(patch)={auroc_p:.4f}  best_F1={f1_p[best_idx_p]:.4f}\n")

    print("=" * 65)
    print(f"{'Category':<15} {'AUROC(G)':>10} {'AUROC(P)':>10} {'Best F1':>9} {'Best Thresh':>13}")
    print("-" * 65)

    for cat, r in results.items():
        print(f"{cat:<15} {r['auroc_global']:>10.4f} {r['auroc_patch']:>10.4f} {r['best_f1']:>9.4f} {r['best_threshold']:>13.6f}")
    print("=" * 65)

    mean_g = np.mean([r['auroc_global'] for r in results.values()])
    mean_p = np.mean([r['auroc_patch']  for r in results.values()])
    print(f"\n평균 AUROC — Global: {mean_g:.4f}  Patch: {mean_p:.4f}")

    global_path = unique_path(os.path.join(RESULTS_DIR, 'scores_global.npz'))
    patch_path  = unique_path(os.path.join(RESULTS_DIR, 'scores_patch.npz'))
    np.savez(global_path,
             **{f'{cat}_labels': v[0] for cat, v in all_data_global.items()},
             **{f'{cat}_scores': v[1] for cat, v in all_data_global.items()})
    np.savez(patch_path,
             **{f'{cat}_labels': v[0] for cat, v in all_data_patch.items()},
             **{f'{cat}_scores': v[1] for cat, v in all_data_patch.items()})
    print(f"저장: {global_path}")
    print(f"저장: {patch_path}")

    plot_score_distributions(all_data_global,
                             title='Score Distribution: Normal vs Anomaly (AE Global MSE)',
                             xlabel='Global MSE Score (density)',
                             out_path=unique_path(os.path.join(RESULTS_DIR, 'score_distributions_global.png')))
    plot_score_distributions(all_data_patch,
                             title='Score Distribution: Normal vs Anomaly (AE Patch MSE)',
                             xlabel='Patch MSE Score (density)',
                             out_path=unique_path(os.path.join(RESULTS_DIR, 'score_distributions_patch.png')))
    plot_roc_curves(all_data_global,
                    title='ROC Curves per Category (AE Global MSE)',
                    out_path=unique_path(os.path.join(RESULTS_DIR, 'roc_curves_global.png')))
    plot_roc_curves(all_data_patch,
                    title='ROC Curves per Category (AE Patch MSE)',
                    out_path=unique_path(os.path.join(RESULTS_DIR, 'roc_curves_patch.png')))


if __name__ == "__main__":
    main()
