import os
import pickle
from models.embedder import Embedder
from utils.search import search_similar
from config import *


def build_embedding_db(embedder, image_dir):
    """임베딩 데이터베이스 생성"""
    db = {}
    image_files = [f for f in os.listdir(image_dir)
                   if f.lower().endswith(SUPPORTED_FORMATS)]

    if not image_files:
        print(f"❗ '{image_dir}' 폴더에 이미지가 없습니다.")
        return db

    if VERBOSE:
        print(f"📦 {len(image_files)}개 이미지 임베딩 생성 중...")

    for i, fname in enumerate(image_files):
        path = os.path.join(image_dir, fname)
        try:
            db[fname] = embedder.get_embedding(path)
            if VERBOSE and (i + 1) % PROGRESS_UPDATE_INTERVAL == 0:
                print(f"   진행: {i + 1}/{len(image_files)}")
        except Exception as e:
            if LOG_ERRORS:
                print(f"⚠️ {fname} 처리 중 오류: {e}")

    return db


def save_db(db, path):
    """데이터베이스 저장"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(db, f)
        if VERBOSE:
            print(f"✅ DB 저장 완료: {len(db)}개 임베딩")
    except Exception as e:
        if LOG_ERRORS:
            print(f"❌ DB 저장 실패: {e}")


def load_db(path):
    """데이터베이스 로드"""
    try:
        with open(path, "rb") as f:
            db = pickle.load(f)

        # 차원 문제 수정
        fixed_db = {}
        for fname, embedding in db.items():
            if embedding.dim() > 1:
                embedding = embedding.squeeze().flatten()
            elif embedding.dim() == 1:
                embedding = embedding.flatten()
            fixed_db[fname] = embedding

        if VERBOSE:
            print(f"✅ DB 로드 완료: {len(fixed_db)}개 임베딩")
        return fixed_db

    except Exception as e:
        if LOG_ERRORS:
            print(f"❌ DB 로드 실패: {e}")
        return {}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='이미지 유사도 검색 CLI 도구')
    parser.add_argument("--query", required=True, help="검색할 이미지 경로")
    parser.add_argument("--rebuild", action="store_true", help="데이터베이스 재생성")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"상위 몇 개 결과를 표시할지 (기본값: {DEFAULT_TOP_K}, 최대: {MAX_TOP_K})")
    parser.add_argument("--verbose", action="store_true", help="상세한 출력")
    args = parser.parse_args()

    # 설정 적용
    if args.verbose:
        globals()['VERBOSE'] = True

    # 상위 K개 결과 제한
    top_k = min(args.top_k, MAX_TOP_K)

    # 필요한 디렉토리 생성
    ensure_directories()

    # 설정 정보 출력
    if VERBOSE:
        print_config()

    # 쿼리 이미지 확인
    if not os.path.exists(args.query):
        print(f"❌ 쿼리 이미지를 찾을 수 없습니다: {args.query}")
        exit(1)

    if not args.query.lower().endswith(SUPPORTED_FORMATS):
        print(f"❌ 지원하지 않는 이미지 형식입니다: {args.query}")
        print(f"   지원 형식: {SUPPORTED_FORMATS}")
        exit(1)

    # 모델 로드
    try:
        if VERBOSE:
            print(f"🤖 모델 로딩 중... ({MODEL_NAME.upper()} on {DEVICE})")
        embedder = Embedder(model_name=MODEL_NAME, device=DEVICE)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        exit(1)

    # 임베딩 DB 로드 또는 생성
    if args.rebuild or not os.path.exists(DB_PATH):
        if VERBOSE:
            print("📦 임베딩 DB 생성 중...")
        db = build_embedding_db(embedder, IMAGE_DIR)
        if db:
            save_db(db, DB_PATH)
        else:
            print("❌ DB 생성 실패")
            exit(1)
    else:
        if VERBOSE:
            print("📂 기존 DB 로드 중...")
        db = load_db(DB_PATH)
        if not db:
            print("❌ DB 로드 실패, --rebuild 옵션으로 재생성하세요")