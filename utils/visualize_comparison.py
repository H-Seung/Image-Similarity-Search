"""
AutoEncoder vs PatchCore 결과 비교 시각화
사용법: python utils/visualize_comparison.py
결과: results/comparison_table.csv, results/comparison_roc.png
"""
import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import numpy as np
from PIL import Image
from tqdm import tqdm
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from utils.common import unique_path

AE_GLOBAL_PATH = os.path.join(ROOT_DIR, "results", "autoencoder", "scores_global.npz")
AE_PATCH_PATH  = os.path.join(ROOT_DIR, "results", "autoencoder", "scores_patch.npz")
PC_SCORES_PATH = os.path.join(ROOT_DIR, "results", "patchcore",   "pc_scores.npz")
RESULTS_DIR    = os.path.join(ROOT_DIR, "results", "comparison")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_ae_scores(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"scores npz 없음. 먼저 evaluate_anomaly.py 실행 필요.\n{path}"
        )
    data = np.load(path)
    cats = sorted({k.replace('_labels', '').replace('_scores', '')
                   for k in data.files})
    return {cat: (data[f'{cat}_labels'], data[f'{cat}_scores']) for cat in cats}


def load_patchcore_scores():
    if not os.path.exists(PC_SCORES_PATH):
        raise FileNotFoundError(
            f"scores.npz 없음. 먼저 patchcore_evaluate.py 실행 필요.\n{PC_SCORES_PATH}"
        )
    data = np.load(PC_SCORES_PATH)
    cats = sorted({k.replace('_labels', '').replace('_scores', '')
                   for k in data.files})
    return {cat: (data[f'{cat}_labels'], data[f'{cat}_scores']) for cat in cats}


def plot_comparison_table(ae_global_auroc, ae_patch_auroc, patchcore_auroc):
    cats = sorted(patchcore_auroc.keys())
    rows = []
    for cat in cats:
        g  = ae_global_auroc.get(cat, float('nan'))
        p  = ae_patch_auroc.get(cat, float('nan'))
        pc = patchcore_auroc[cat]
        rows.append([cat, f'{g:.4f}', f'{p:.4f}', f'{pc:.4f}',
                     f'{pc - p:+.4f}'])

    mean_g  = np.mean(list(ae_global_auroc.values()))
    mean_p  = np.mean(list(ae_patch_auroc.values()))
    mean_pc = np.mean(list(patchcore_auroc.values()))
    rows.append(['Mean', f'{mean_g:.4f}', f'{mean_p:.4f}',
                 f'{mean_pc:.4f}', f'{mean_pc - mean_p:+.4f}'])

    out = os.path.join(RESULTS_DIR, 'comparison_table.csv')
    out = unique_path(out)
    col_labels = ['Category', 'AE (Global MSE)', 'AE (Patch)', 'PatchCore', 'AE(Patch)-PC']
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(col_labels)
        writer.writerows(rows)

    print(f"저장: {out}")


def plot_comparison_roc(all_patchcore_data, all_ae_patch_data):
    """PatchCore ROC (실선) vs AE Patch ROC (점선) 비교"""
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_patchcore_data)))
    for (cat, (labels, scores)), color in zip(all_patchcore_data.items(), colors):
        fpr, tpr, _ = roc_curve(labels, scores)
        auroc       = roc_auc_score(labels, scores)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{cat}  PC={auroc:.4f}')
        if cat in all_ae_patch_data:
            ae_labels, ae_scores = all_ae_patch_data[cat]
            ae_fpr, ae_tpr, _    = roc_curve(ae_labels, ae_scores)
            ae_auroc             = roc_auc_score(ae_labels, ae_scores)
            ax.plot(ae_fpr, ae_tpr, color=color, linewidth=1.5, linestyle='--',
                    label=f'{cat}  AE={ae_auroc:.4f}')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — PatchCore (solid) vs AE Patch (dashed)')
    ax.legend(loc='lower right', fontsize=8)
    out = os.path.join(RESULTS_DIR, 'comparison_roc.png')
    out = unique_path(out)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"저장: {out}")


def main():
    all_patchcore_data = load_patchcore_scores()
    all_ae_global_data = load_ae_scores(AE_GLOBAL_PATH)
    all_ae_patch_data  = load_ae_scores(AE_PATCH_PATH)

    patchcore_auroc = {cat: roc_auc_score(v[0], v[1]) for cat, v in all_patchcore_data.items()}
    ae_global_auroc = {cat: roc_auc_score(v[0], v[1]) for cat, v in all_ae_global_data.items()}
    ae_patch_auroc  = {cat: roc_auc_score(v[0], v[1]) for cat, v in all_ae_patch_data.items()}

    plot_comparison_table(ae_global_auroc, ae_patch_auroc, patchcore_auroc)
    plot_comparison_roc(all_patchcore_data, all_ae_patch_data)


if __name__ == "__main__":
    main()
