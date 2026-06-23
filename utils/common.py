import os
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATEGORIES_PATH = os.path.join(ROOT_DIR, "models", "used_categories.json")


def get_available_categories():
    """카테고리명 리스트를 가져온다."""
    if not os.path.exists(CATEGORIES_PATH):
        raise FileNotFoundError(f"used_categories.json 없음: {CATEGORIES_PATH}")
    with open(CATEGORIES_PATH, "r") as f:
        return sorted(json.load(f))
    
def unique_path(path):
    """동명 파일이 있으면 _1, _2, ... 접미사를 붙여 고유한 경로를 반환."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while os.path.exists(f"{base}_{i}{ext}"):
        i += 1
    return f"{base}_{i}{ext}"