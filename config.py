# config.py - 향상된 설정 파일
import torch
import os

# 모델 설정
MODEL_NAME = "clip"  # "clip" 또는 "resnet"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
DB_PATH = os.path.join(DATA_DIR, "embeddings_db.pkl")

# 검색 설정
DEFAULT_TOP_K = 5
MAX_TOP_K = 10

# 이미지 설정
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.jfif', '.bmp', '.tiff', '.webp')
THUMBNAIL_SIZE = (224, 224)
DISPLAY_SIZE = (150, 150)

# GUI 설정
WINDOW_SIZE = "900x700"
CANVAS_SIZE = (224, 224)

# 성능 설정
BATCH_SIZE = 32  # 임베딩 생성 시 배치 크기
PROGRESS_UPDATE_INTERVAL = 10  # N개마다 진행상황 출력

# 디버그 설정
VERBOSE = True
LOG_ERRORS = True

def ensure_directories():
    """필요한 디렉토리들을 생성합니다."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

def print_config():
    """현재 설정을 출력합니다."""
    if VERBOSE:
        print(f"🔧 Configuration:")
        print(f"   Model: {MODEL_NAME.upper()}")
        print(f"   Device: {DEVICE}")
        print(f"   Image Dir: {IMAGE_DIR}")
        print(f"   Database: {DB_PATH}")
        print(f"   Supported Formats: {SUPPORTED_FORMATS}")

if __name__ == "__main__":
    ensure_directories()
    print_config()