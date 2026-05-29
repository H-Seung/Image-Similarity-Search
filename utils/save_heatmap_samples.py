"""
카테고리별 정상/이상 샘플 히트맵 저장 (원본 | 히트맵 나란히)
사용법: python utils/save_heatmap_samples.py
결과: results/heatmap_samples/{category}_good.png, {category}_{defect}.png
"""
import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from PIL import Image, ImageDraw
from models.anomaly_detector_encoder import run_anomaly_inference

DATA_DIR   = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUT_DIR    = os.path.join(ROOT_DIR, "results", "heatmap_samples")
os.makedirs(OUT_DIR, exist_ok=True)

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')
IMG_SIZE   = (128, 128)
PAD        = 16


def get_available_categories():
    cats = []
    for fname in os.listdir(MODELS_DIR):
        if fname.startswith("autoencoder_") and fname.endswith(".pth") and "_prep" not in fname:
            cats.append(fname[len("autoencoder_"):-len(".pth")])
    return sorted(cats)


def first_image(folder):
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(IMAGE_EXTS):
            return os.path.join(folder, fname)
    return None


def save_sample(img_path, category, defect_type):
    loss, result, heatmap = run_anomaly_inference(img_path, category)

    original = Image.open(img_path).convert('RGB').resize(IMG_SIZE, Image.Resampling.LANCZOS)
    heatmap  = heatmap.resize(IMG_SIZE, Image.Resampling.LANCZOS)

    w, h     = IMG_SIZE
    label_h  = 16
    canvas_w = w * 2 + PAD * 3
    canvas_h = h + PAD * 2 + label_h * 2

    canvas = Image.new('RGB', (canvas_w, canvas_h), (245, 245, 245))
    canvas.paste(original, (PAD, PAD + label_h))
    canvas.paste(heatmap,  (w + PAD * 2, PAD + label_h))

    draw = ImageDraw.Draw(canvas)
    draw.text((PAD,          PAD), 'Original', fill=(80, 80, 80))
    draw.text((w + PAD * 2,  PAD), 'Error Heatmap', fill=(80, 80, 80))
    draw.text((PAD, h + PAD + label_h + 2),
              f'Score: {loss:.6f}   Result: {result}', fill=(40, 40, 40))

    out_path = os.path.join(OUT_DIR, f"{category}_{defect_type}.png")
    canvas.save(out_path)
    print(f"  저장: {category}_{defect_type}.png  [{result}  score={loss:.6f}]")


def main():
    categories = get_available_categories()
    for cat in categories:
        print(f"\n[{cat}]")
        test_dir = os.path.join(DATA_DIR, cat, "test")
        if not os.path.exists(test_dir):
            print("  test 폴더 없음, 스킵")
            continue

        good_done    = False
        anomaly_done = False

        for defect_type in sorted(os.listdir(test_dir)):
            if good_done and anomaly_done:
                break
            d = os.path.join(test_dir, defect_type)
            if not os.path.isdir(d):
                continue
            img_path = first_image(d)
            if not img_path:
                continue

            if defect_type == "good" and not good_done:
                save_sample(img_path, cat, "good")
                good_done = True
            elif defect_type != "good" and not anomaly_done:
                save_sample(img_path, cat, defect_type)
                anomaly_done = True


if __name__ == "__main__":
    main()