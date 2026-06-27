"""
PatchCore AUROC 정량 평가 및 시각화 (score 저장)
사용법: 
python utils/pc_evaluate.py (전체)
python utils/pc_evaluate.py --categories grid carpet (grid, carpet만)
"""
import os
import sys
import argparse
import csv
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
    """결함 유형별 score 수집 → {defect_type: [scores]} 반환."""
    memory_bank, feature_shape = load_memory_bank(category, MB_DIR)
    engine = PatchCoreInference(memory_bank, feature_shape)

    test_dir = os.path.join(DATA_DIR, category, "test")
    if not os.path.exists(test_dir):
        return {}

    raw = {}
    for defect_type in sorted(os.listdir(test_dir)):
        defect_dir = os.path.join(test_dir, defect_type)
        if not os.path.isdir(defect_dir):
            continue
        scores = []
        for fname in tqdm(sorted(os.listdir(defect_dir)),
                          desc=f"  [{category}/{defect_type}]", leave=False):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            try:
                img = Image.open(os.path.join(defect_dir, fname)).convert('RGB')
                tensor = IMAGE_TRANSFORM(img).unsqueeze(0)
                patch_features, _ = extractor.extract(tensor)
                score, _ = engine.score(patch_features)
                scores.append(score)
            except Exception as e:
                print(f"  ⚠️  {fname}: {e}")
        raw[defect_type] = scores
    return raw


def _aggregate(raw):
    """raw → 카테고리 단위 (labels, scores) 집계."""
    good = raw.get("good", [])
    anomaly = [s for k, v in raw.items() if k != "good" for s in v]
    return np.array([0]*len(good) + [1]*len(anomaly)), np.array(good + anomaly)


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


def plot_defect_auroc(all_raw, out_path):
    cats = list(all_raw.keys())
    fig, axes = plt.subplots(1, len(cats), figsize=(6 * len(cats), max(4, 0.5 * max(
        len([k for k in raw if k != "good"]) for raw in all_raw.values()) + 2)))
    if len(cats) == 1:
        axes = [axes]

    for ax, cat in zip(axes, cats):
        raw = all_raw[cat]
        good = raw.get("good", [])
        defect_types = sorted(k for k in raw if k != "good")
        aurocs = []
        for dt in defect_types:
            ds = raw[dt]
            lbl = [0]*len(good) + [1]*len(ds)
            aurocs.append(roc_auc_score(lbl, good + ds) if ds and len(set(lbl)) == 2 else 0.0)

        colors = ['tomato' if a < 0.95 else 'steelblue' for a in aurocs]
        ax.barh(defect_types, aurocs, color=colors)
        ax.set_xlim(0.5, 1.05)
        ax.axvline(x=0.95, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title(cat, fontsize=12)
        ax.set_xlabel('AUROC')
        for i, v in enumerate(aurocs):
            ax.text(v + 0.003, i, f'{v:.4f}', va='center', fontsize=9)

    plt.suptitle('Per-Defect AUROC', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"저장: {out_path}")


def save_defect_csv(all_raw, out_path):
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['category', 'defect_type', 'auroc', 'n_samples'])
        for cat, raw in all_raw.items():
            good = raw.get("good", [])
            labels, scores = _aggregate(raw)
            if len(set(labels)) == 2:
                writer.writerow([cat, 'ALL', f'{roc_auc_score(labels, scores):.4f}', len(scores) - len(good)])
            for defect_type, defect_scores in sorted(raw.items()):
                if defect_type == "good" or not defect_scores:
                    continue
                lbl = [0]*len(good) + [1]*len(defect_scores)
                scr = good + defect_scores
                if len(set(lbl)) == 2:
                    writer.writerow([cat, defect_type, f'{roc_auc_score(lbl, scr):.4f}', len(defect_scores)])



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--categories', nargs='+', default=None, 
                        metavar='CAT', help='Categories to evaluate(If not specified, applies to All)')
    args = parser.parse_args()

    available = get_available_categories()
    categories = args.categories if args.categories else available
    unknown = [c for c in categories if c not in available]
    if unknown:
        print(f"⚠️  Unknown category: {unknown} ")
    categories = [c for c in categories if c in available]
    print(f"Category: {categories}\n")

    extractor = PatchCoreFeatureExtractor(device='cpu')
    results, all_raw, all_data = {}, {}, {}

    for cat in categories:
        print(f"[{cat}] 평가 중...")
        raw = evaluate_category(cat, extractor)
        if not raw:
            print(f"  ⚠️  스킵\n")
            continue
        labels, scores = _aggregate(raw)

        if len(np.unique(labels)) < 2:
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
        all_raw[cat]  = raw
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

    # 카테고리별 시각화
    plot_score_distributions(all_data)
    plot_roc_curves(all_data)

    # 전체 (카테고리+결함별) auroc score 저장
    csv_path = unique_path(os.path.join(RESULTS_DIR, 'defect_auroc.csv'))
    save_defect_csv(all_raw, csv_path)

    # 결함별 auroc 시각화
    bar_path = unique_path(os.path.join(RESULTS_DIR, 'defect_auroc.png'))
    plot_defect_auroc(all_raw, bar_path)

    # scores 저장
    save_path = unique_path(os.path.join(RESULTS_DIR, 'pc_scores.npz'))
    np.savez(save_path,
             **{f'{cat}_labels': v[0] for cat, v in all_data.items()},
             **{f'{cat}_scores': v[1] for cat, v in all_data.items()})
    print(f"저장: {save_path}")


if __name__ == "__main__":
    main()
