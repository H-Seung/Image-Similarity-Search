import os
import pickle
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import tkinterdnd2 as tkdnd
import torch

from models.embedder import Embedder
from utils.search import search_similar
from config import *


class ImageSearchApp(tkdnd.TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("이미지 유사도 검색기")
        self.geometry(WINDOW_SIZE)
        self.resizable(True, True)

        # 메인 프레임
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 상단 정보 영역
        info_frame = tk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.info_label = tk.Label(info_frame, text="이미지 DB 로딩 중...", font=("Arial", 12))
        self.info_label.pack()

        # 드래그앤드롭 안내
        self.label = tk.Label(main_frame, text="검색할 이미지를 드래그 앤 드롭 하세요",
                              font=("Arial", 16), fg="blue")
        self.label.pack(pady=10)

        # 쿼리 이미지 표시 영역
        query_frame = tk.Frame(main_frame)
        query_frame.pack(pady=10)

        tk.Label(query_frame, text="검색 이미지", font=("Arial", 12, "bold")).pack()
        self.canvas_query = tk.Canvas(query_frame, width=CANVAS_SIZE[0], height=CANVAS_SIZE[1],
                                      bg="white", relief=tk.SUNKEN, bd=2)
        self.canvas_query.pack(pady=5)

        # 결과 영역
        result_label_frame = tk.Frame(main_frame)
        result_label_frame.pack(fill=tk.X, pady=(20, 5))
        tk.Label(result_label_frame, text=f"유사한 이미지 (상위 {DEFAULT_TOP_K}개)",
                 font=("Arial", 12, "bold")).pack()

        # 스크롤 가능한 결과 프레임
        self.result_canvas = tk.Canvas(main_frame, height=300)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.result_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.result_canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all"))
        )

        self.result_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.result_canvas.configure(yscrollcommand=scrollbar.set)

        self.result_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 드래그앤드롭 설정
        self.drop_target_register(tkdnd.DND_FILES)
        self.dnd_bind('<<Drop>>', self.handle_drop)

        # 임베딩 모델 및 DB 초기화
        self.embedder = None
        self.db = None

        # 필요한 디렉토리 생성
        ensure_directories()

        # 설정 정보 출력
        if VERBOSE:
            print_config()

        self.init_components()

    def init_components(self):
        """백그라운드에서 컴포넌트 초기화"""
        try:
            self.info_label.config(text=f"모델 로딩 중... ({MODEL_NAME.upper()} on {DEVICE})")
            self.update()

            self.embedder = Embedder(model_name=MODEL_NAME, device=DEVICE)

            self.info_label.config(text="임베딩 DB 로딩/생성 중...")
            self.update()

            self.db = self.load_or_build_db()

            db_count = len(self.db) if self.db else 0
            self.info_label.config(text=f"✅ 준비 완료! DB에 {db_count}개 이미지 등록됨 (모델: {MODEL_NAME.upper()})")

        except Exception as e:
            error_msg = f"❌ 초기화 실패: {str(e)}"
            self.info_label.config(text=error_msg, fg="red")
            messagebox.showerror("초기화 오류", error_msg)

    def load_or_build_db(self):
        """DB 로드 또는 새로 생성"""
        if not os.path.exists(DB_PATH):
            if VERBOSE:
                print("📦 DB가 없어 새로 생성합니다.")
            return self.build_db()

        try:
            with open(DB_PATH, "rb") as f:
                db = pickle.load(f)
                if VERBOSE:
                    print(f"✅ 임베딩 DB 로드 완료 ({len(db)}개 이미지)")

                # DB 데이터 무결성 검사 및 수정
                db = self.fix_db_dimensions(db)
                return db
        except Exception as e:
            if LOG_ERRORS:
                print(f"⚠️ 기존 DB 로드 실패, 새로 생성: {e}")
            return self.build_db()

    def fix_db_dimensions(self, db):
        """기존 DB의 차원 문제를 수정"""
        fixed_db = {}
        needs_save = False

        for fname, embedding in db.items():
            # 차원 수정
            if embedding.dim() > 1:
                embedding = embedding.squeeze().flatten()
                needs_save = True
            elif embedding.dim() == 1:
                embedding = embedding.flatten()

            fixed_db[fname] = embedding

        # 수정된 내용이 있으면 다시 저장
        if needs_save:
            if VERBOSE:
                print("🔧 DB 차원 문제 수정 및 재저장 중...")
            try:
                with open(DB_PATH, "wb") as f:
                    pickle.dump(fixed_db, f)
            except Exception as e:
                if LOG_ERRORS:
                    print(f"⚠️ DB 재저장 실패: {e}")

        return fixed_db

    def build_db(self):
        """이미지 DB 새로 생성"""
        if not os.path.exists(IMAGE_DIR):
            if VERBOSE:
                print(f"❗ '{IMAGE_DIR}' 폴더가 없습니다.")
            return {}

        db = {}
        image_files = [f for f in os.listdir(IMAGE_DIR)
                       if f.lower().endswith(SUPPORTED_FORMATS)]

        if not image_files:
            if VERBOSE:
                print(f"❗ '{IMAGE_DIR}' 폴더에 이미지가 없습니다.")
            return {}

        if VERBOSE:
            print(f"📦 {len(image_files)}개 이미지 임베딩 생성 중...")

        for i, fname in enumerate(image_files):
            path = os.path.join(IMAGE_DIR, fname)
            try:
                db[fname] = self.embedder.get_embedding(path)
                if VERBOSE and (i + 1) % PROGRESS_UPDATE_INTERVAL == 0:
                    print(f"   진행: {i + 1}/{len(image_files)}")
            except Exception as e:
                if LOG_ERRORS:
                    print(f"⚠️ {fname} 처리 중 오류: {e}")

        # DB 저장
        try:
            with open(DB_PATH, "wb") as f:
                pickle.dump(db, f)
            if VERBOSE:
                print(f"✅ DB 저장 완료: {len(db)}개 임베딩")
        except Exception as e:
            if LOG_ERRORS:
                print(f"⚠️ DB 저장 실패: {e}")

        return db

    def handle_drop(self, event):
        """파일 드롭 처리"""
        if not self.embedder or not self.db:
            messagebox.showerror("오류", "시스템이 아직 준비되지 않았습니다.")
            return

        # 파일 경로 정리
        filepath = event.data.strip('{}').strip('"').strip("'")

        if not os.path.isfile(filepath):
            messagebox.showerror("오류", "유효한 이미지 파일을 드롭하세요")
            return

        # 파일 확장자 확인
        if not filepath.lower().endswith(SUPPORTED_FORMATS):
            messagebox.showerror("오류", f"지원하는 이미지 형식이 아닙니다\n{SUPPORTED_FORMATS}")
            return

        try:
            self.info_label.config(text="검색 중...", fg="orange")
            self.update()

            # 쿼리 이미지 표시
            self.show_query_image(filepath)

            # 임베딩 추출 및 검색
            query_vec = self.embedder.get_embedding(filepath)
            results = search_similar(query_vec, self.db, top_k=DEFAULT_TOP_K)

            # 결과 표시
            self.show_results(results)

            self.info_label.config(text=f"✅ 검색 완료! (DB: {len(self.db)}개 이미지)", fg="green")

        except Exception as e:
            error_msg = f"검색 중 오류가 발생했습니다: {str(e)}"
            messagebox.showerror("오류", error_msg)
            self.info_label.config(text="❌ 검색 실패", fg="red")
            if LOG_ERRORS:
                print(f"❌ 드롭 처리 오류: {e}")

    def show_query_image(self, filepath):
        """쿼리 이미지 표시"""
        try:
            img = Image.open(filepath)
            img.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
            self.query_imgtk = ImageTk.PhotoImage(img)

            self.canvas_query.delete("all")
            # 이미지를 캔버스 중앙에 배치
            canvas_width = self.canvas_query.winfo_width()
            canvas_height = self.canvas_query.winfo_height()
            if canvas_width > 1 and canvas_height > 1:  # 캔버스가 제대로 그려진 후
                x = canvas_width // 2
                y = canvas_height // 2
            else:  # 기본값 사용
                x, y = CANVAS_SIZE[0] // 2, CANVAS_SIZE[1] // 2

            self.canvas_query.create_image(x, y, image=self.query_imgtk)
        except Exception as e:
            if LOG_ERRORS:
                print(f"⚠️ 쿼리 이미지 표시 오류: {e}")

    def show_results(self, results):
        """검색 결과 표시"""
        # 기존 결과 제거
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        if not results:
            no_result_label = tk.Label(self.scrollable_frame, text="검색 결과가 없습니다.",
                                       font=("Arial", 12), fg="gray")
            no_result_label.pack(pady=20)
            return

        # 결과를 가로로 배치
        result_frame = tk.Frame(self.scrollable_frame)
        result_frame.pack(fill=tk.X, padx=10, pady=10)

        for i, (fname, score) in enumerate(results):
            try:
                path = os.path.join(IMAGE_DIR, fname)
                if not os.path.exists(path):
                    continue

                # 개별 결과 프레임
                item_frame = tk.Frame(result_frame, relief=tk.RAISED, bd=1)
                item_frame.grid(row=0, column=i, padx=10, pady=5, sticky="n")

                # 이미지 표시
                img = Image.open(path)
                img.thumbnail(DISPLAY_SIZE, Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(img)

                panel = tk.Label(item_frame, image=imgtk)
                panel.image = imgtk  # 참조 유지
                panel.pack(pady=5)

                # 파일명 및 점수 표시
                info_text = f"{fname}\n유사도: {score:.3f}"
                label = tk.Label(item_frame, text=info_text, font=("Arial", 10),
                                 justify=tk.CENTER, wraplength=150)
                label.pack(pady=5)

            except Exception as e:
                if LOG_ERRORS:
                    print(f"⚠️ 결과 표시 중 오류 ({fname}): {e}")

    def rebuild_database(self):
        """데이터베이스 재구축"""
        try:
            self.info_label.config(text="데이터베이스 재구축 중...", fg="orange")
            self.update()

            # 기존 DB 파일 삭제
            if os.path.exists(DB_PATH):
                os.remove(DB_PATH)

            # 새 DB 생성
            self.db = self.build_db()

            db_count = len(self.db) if self.db else 0
            self.info_label.config(text=f"✅ DB 재구축 완료! {db_count}개 이미지 등록됨", fg="green")

        except Exception as e:
            error_msg = f"DB 재구축 실패: {str(e)}"
            self.info_label.config(text=error_msg, fg="red")
            if LOG_ERRORS:
                print(f"❌ DB 재구축 오류: {e}")


if __name__ == "__main__":
    # 설정 확인
    if not os.path.exists(IMAGE_DIR):
        print(f"❗ '{IMAGE_DIR}' 폴더를 생성합니다.")
        ensure_directories()

    image_files = [f for f in os.listdir(IMAGE_DIR)
                   if f.lower().endswith(SUPPORTED_FORMATS)]

    if not image_files:
        print(f"⚠️ '{IMAGE_DIR}' 폴더에 이미지를 넣고 다시 실행하세요.")
        print(f"   지원 형식: {SUPPORTED_FORMATS}")
        input("Enter를 눌러 종료...")
    else:
        if VERBOSE:
            print(f"🚀 GUI 앱 시작 중... ({len(image_files)}개 이미지 발견)")
        app = ImageSearchApp()
        app.mainloop()