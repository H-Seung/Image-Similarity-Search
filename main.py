import os
import pickle
from models.embedder import Embedder
from utils.search import search_similar
from config import *

def build_embedding_db(embedder, image_dir):
    db = {}
    for fname in os.listdir(image_dir):
        if fname.lower().endswith((".jpg", ".png")):
            path = os.path.join(image_dir, fname)
            db[fname] = embedder.get_embedding(path)
    return db

def save_db(db, path):
    with open(path, "wb") as f:
        pickle.dump(db, f)

def load_db(path):
    with open(path, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="path to query image")
    parser.add_argument("--rebuild", action="store_true", help="rebuild db")
    args = parser.parse_args()

    embedder = Embedder(model_name=MODEL_NAME, device=DEVICE)

    # 임베딩 DB 로드 또는 생성
    if args.rebuild or not os.path.exists(DB_PATH):
        print("📦 Building embedding DB...")
        db = build_embedding_db(embedder, IMAGE_DIR)
        save_db(db, DB_PATH)
    else:
        db = load_db(DB_PATH)

    # 쿼리 이미지 처리
    print(f"🔍 Searching similar images for: {args.query}")
    query_vec = embedder.get_embedding(args.query)
    results = search_similar(query_vec, db)

    print("\n🎯 Top Similar Images:")
    for fname, score in results:
        print(f"- {fname} ({score:.4f})")
