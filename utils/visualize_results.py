"""
Score 분포 시각화 / ROC Curve / Ablation Table 저장
사용법: python utils/visualize_results.py
결과: results/ 폴더에 PNG 3장 저장
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
from models.anomaly_detector_encoder import AutoEncoder, compute_patch_anomaly_score

DATA_DIR    = os.path.join(ROOT_DIR, "data")
MODELS_DIR  = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')

GLOBAL_MSE_AUROC = {  # 이전 실험 결과
    "bottle": 0.7921, "cable": 0.5260, "carpet": 0.4282,
    "grid": 0.7561, "hazelnut": 0.9146, "leather": 0.5880, "metal_nut": 0.3578,
}

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


def collect_scores(category):
    model_path = os.path.join(MODELS_DIR, f"autoencoder_{category}.pth")
    model = AutoEncoder()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    test_dir = os.path.join(DATA_DIR, category, "test")
    if not os.path.exists(test_dir):
        return None, None

    labels, scores = [], []
    for defect_type in sorted(os.listdir(test_dir)):
        defect_dir = os.path.join(test_dir, defect_type)
        if not os.path.isdir(defect_dir):
            continue
        label = 0 if defect_type == "good" else 1
        for fname in tqdm(os.listdir(defect_dir), desc=f"  [{category}/{defect_type}]", leave=False):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            try:
                img = Image.open(os.path.join(defect_dir, fname)).convert('RGB')
                tensor = transform(img).unsqueeze(0)
                score, _ = compute_patch_anomaly_score(model, tensor)
                labels.append(label)
                scores.append(score)
            except Exception as e:
                print(f"  ⚠️  {fname}: {e}")

    return np.array(labels), np.array(scores)


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
        ax.set_title(f'{cat}  (AUROC={auroc:.3f})', fontsize=10)
        ax.set_xlabel('Patch MSE Score (density)', fontsize=8)
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Score Distribution: Normal vs Anomaly (Patch-based AutoEncoder)', fontsize=13)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, 'score_distributions.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"저장: {out}")


def plot_roc_curves(all_data):
    fig, ax = plt.subplots(figsize=(8, 7))
    for cat, (labels, scores) in all_data.items():
        fpr, tpr, _ = roc_curve(labels, scores)
        auroc = roc_auc_score(labels, scores)
        ax.plot(fpr, tpr, label=f'{cat} ({auroc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves per Category (Patch-based AutoEncoder)')
    ax.legend(loc='lower right', fontsize=9)
    out = os.path.join(RESULTS_DIR, 'roc_curves.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"저장: {out}")


def plot_ablation_table(all_data):
    cats = list(all_data.keys())
    patch_auroc = {cat: roc_auc_score(all_data[cat][0], all_data[cat][1]) for cat in cats}

    rows = []
    for cat in cats:
        g = GLOBAL_MSE_AUROC.get(cat, float('nan'))
        p = patch_auroc[cat]
        rows.append([cat, f'{g:.4f}', f'{p:.4f}', f'{p - g:+.4f}'])
    mean_g = np.mean(list(GLOBAL_MSE_AUROC.values()))
    mean_p = np.mean(list(patch_auroc.values()))
    rows.append(['Mean', f'{mean_g:.4f}', f'{mean_p:.4f}', f'{mean_p - mean_g:+.4f}'])

    fig, ax = plt.subplots(figsize=(7, len(rows) * 0.55 + 1.2))
    ax.axis('off')
    table = ax.table(
        cellText=rows,
        colLabels=['Category', 'Global MSE', 'Patch-based', 'Δ AUROC'],
        loc='center', cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.9)

    for i, row in enumerate(rows):
        delta = float(row[3])
        table[i + 1, 3].set_facecolor('#d4edda' if delta > 0 else '#f8d7da')

    ax.set_title('Ablation: Global MSE vs Patch-based AUROC', fontsize=12, pad=12)
    out = os.path.join(RESULTS_DIR, 'ablation_table.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"저장: {out}")


def main():
    categories = get_available_categories()
    print(f"카테고리: {categories}\n")

    all_data = {}
    for cat in categories:
        print(f"[{cat}] 점수 수집 중...")
        labels, scores = collect_scores(cat)
        if labels is not None and len(np.unique(labels)) >= 2:
            all_data[cat] = (labels, scores)

    plot_score_distributions(all_data)
    # plot_roc_curves(all_data)
    # plot_ablation_table(all_data)


if __name__ == "__main__":
    main()