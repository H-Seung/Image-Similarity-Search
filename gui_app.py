"""
사용법: python gui_app.py
"""
import os
import pickle
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import tkinterdnd2 as tkdnd

from config import *
from models.embedder import Embedder
from utils.search import search_similar
from models.autoencoder.inference import run_anomaly_inference
from models.patchcore.inference import run_patchcore_inference


MB_DIR = os.path.join(os.path.dirname(__file__), "models", "patchcore", "memory_bank")

class ImageSearchApp(tkdnd.TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vision Intelligence Platform")
        self.geometry(WINDOW_SIZE)
        self.resizable(True, True)

        self.current_model_name = MODEL_NAME.lower()

        # 1. 메인 탭 위젯(Notebook) 생성
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 2. 각 기능을 위한 독립적인 프레임(탭) 생성
        self.tab_similarity = tk.Frame(self.notebook)
        self.tab_anomaly = tk.Frame(self.notebook)

        self.notebook.add(self.tab_similarity, text=" 유사이미지 검색 ")
        self.notebook.add(self.tab_anomaly, text=" 이상 탐지 ")

        # UI 설정 코드를 각 탭 프레임 내부로 배치
        self.setup_similarity_ui(self.tab_similarity)
        self.setup_anomaly_ui(self.tab_anomaly)

        # 드래그앤드롭 : 앱 전역(self)에 드롭을 바인딩
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
            self.stat_label.config(text=f"모델 로딩 중... ({MODEL_NAME.upper()} on {DEVICE})")
            self.update()

            self.embedder = Embedder(model_name=MODEL_NAME, device=DEVICE)

            self.stat_label.config(text="임베딩 DB 로딩/생성 중...")
            self.update()

            self.db = self.load_db()

            db_count = len(self.db) if self.db else 0
            # 왼쪽 구역 정보 업데이트
            self.stat_label.config(text=f"DB: {db_count}개 이미지")
            self.model_info_label.config(text=f"현재 모델: {MODEL_NAME.upper()} ({DEVICE})")
            # 가운데 구역 업데이트
            self.center_status_label.config(text="✅ 준비 완료!", fg="black")

        except Exception as e:
            error_msg = f"❌ 초기화 실패: {str(e)}"
            self.stat_label.config(text=error_msg, fg="red")
            messagebox.showerror("초기화 오류", error_msg)


    def setup_similarity_ui(self, parent):    
        # 메인 프레임을 부모 탭(parent) 내부에 생성
        main_frame = tk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=12)

        # 헤더
        header = tk.Frame(main_frame)
        header.pack(fill=tk.X, pady=(0, 8))
        tk.Label(header, text="유사 이미지 검색",
                 font=("Arial", 15, "bold"), fg="#1a1a1a").pack(anchor="w")
        tk.Label(header, text="검색하려는 이미지를 드래그 앤 드롭하면, DB 내 이미지에서 유사한 이미지를 검색합니다.",
                 font=("Arial", 11), fg="#555555").pack(anchor="w", pady=(2, 0))
        
        # 컨트롤 바
        ctrl = tk.Frame(main_frame, bg="#f5f5f5",
                        highlightthickness=1, highlightbackground="#e0e0e0")
        ctrl.pack(fill=tk.X, pady=(0, 12), ipady=6)

        left = tk.Frame(ctrl, bg="#f5f5f5")
        left.pack(side=tk.LEFT, padx=12)
        tk.Label(left, text="모델", font=("Arial", 10, "bold"),
                 bg="#f5f5f5", fg="#333333").pack(side=tk.LEFT, padx=(0, 8))
        self.similarity_model_var = tk.StringVar(value=self.current_model_name)
        for val, lbl in (("clip", "CLIP"), ("resnet", "ResNet")):
            tk.Radiobutton(left, text=lbl, variable=self.similarity_model_var,
                           value=val, font=("Arial", 10), bg="#f5f5f5",
                           command=self._on_similarity_model_change).pack(side=tk.LEFT, padx=4)

        self.center_status_label = tk.Label(ctrl, text="⌛ 준비 중...",
                                            font=("Arial", 11, "bold"),
                                            bg="#f5f5f5", fg="#333333")
        self.center_status_label.pack(side=tk.LEFT, expand=True)

        right = tk.Frame(ctrl, bg="#f5f5f5")
        right.pack(side=tk.RIGHT, padx=12)
        self.btn_rebuild = tk.Button(right, text="DB 재생성",
                                     command=self.rebuild_database,
                                     font=("Arial", 10), width=10)
        self.btn_rebuild.pack(anchor="e")
        self.stat_label = tk.Label(right, text="DB: 로딩 중...",
                                   font=("Arial", 10), bg="#f5f5f5", fg="#666666")
        self.stat_label.pack(anchor="e", pady=(3, 0))
        self.model_info_label = tk.Label(right,
                                         text=f"현재 모델: {MODEL_NAME.upper()} ({DEVICE})",
                                         font=("Arial", 10), bg="#f5f5f5", fg="#666666")
        self.model_info_label.pack(anchor="e", pady=(2, 0))

        # 이미지 영역
        img_row = tk.Frame(main_frame)
        img_row.pack(pady=(0, 8))

        q_frame = tk.Frame(img_row, padx=20)
        q_frame.grid(row=0, column=0, sticky="n")
        tk.Label(q_frame, text="검색 이미지",
                 font=("Arial", 13, "bold"), fg="#1a1a1a").pack(pady=(0, 5))
        self.canvas_query = tk.Canvas(q_frame, width=CANVAS_SIZE[0], height=CANVAS_SIZE[1],
                                      bg="white", bd=0,
                                      highlightthickness=1, highlightbackground="#dddddd")
        self.canvas_query.pack()
        self.lbl_query_name = tk.Label(q_frame, text="", font=("Arial", 10), fg="#888888")
        self.lbl_query_name.pack(pady=(3, 0))

        b_frame = tk.Frame(img_row, padx=20)
        b_frame.grid(row=0, column=1, sticky="n", padx=(80, 20))
        tk.Label(b_frame, text="유사 이미지",
                 font=("Arial", 13, "bold"), fg="#1a1a1a").pack(pady=(0, 5))
        self.canvas_best = tk.Canvas(b_frame, width=CANVAS_SIZE[0], height=CANVAS_SIZE[1],
                                     bg="white", bd=0,
                                     highlightthickness=1, highlightbackground="cyan")
        self.canvas_best.pack()
        self.lbl_best_name = tk.Label(b_frame, text="", font=("Arial", 10), fg="#888888")
        self.lbl_best_name.pack(pady=(3, 0))

        # 결과 섹션
        tk.Label(main_frame, text=f"유사한 이미지 (상위 {DEFAULT_TOP_K}개)",
                 font=("Arial", 12, "bold"), fg="#1a1a1a").pack(anchor="w", pady=(0, 4))
        self.result_canvas = tk.Canvas(main_frame, height=220)
        scrollbar = ttk.Scrollbar(main_frame, orient="horizontal",
                                  command=self.result_canvas.xview)
        self.scrollable_frame = ttk.Frame(self.result_canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.result_canvas.configure(
                scrollregion=self.result_canvas.bbox("all"))
        )
        self.result_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.result_canvas.configure(xscrollcommand=scrollbar.set)
        self.result_canvas.pack(side="top", fill=tk.BOTH, expand=True, pady=(2, 0))
        scrollbar.pack(side="top", fill="x")

    def setup_anomaly_ui(self, parent):
        # 메인 프레임을 부모 탭(parent) 내부에 생성 
        main_frame = tk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=12)

        # 헤더
        header = tk.Frame(main_frame)
        header.pack(fill=tk.X, pady=(0, 8))
        tk.Label(header, text="이상 탐지",
                 font=("Arial", 15, "bold"), fg="#1a1a1a").pack(anchor="w")
        tk.Label(header, text="분석하려는 이미지를 드래그 앤 드롭하면, AI 모델이 이미지의 이상(불량) 여부를 탐지합니다.",
                 font=("Arial", 11), fg="#555555").pack(anchor="w", pady=(2, 0))

        # 컨트롤 바
        ctrl = tk.Frame(main_frame, bg="#f5f5f5",
                        highlightthickness=1, highlightbackground="#e0e0e0")
        ctrl.pack(fill=tk.X, pady=(0, 12), ipady=6)
        left = tk.Frame(ctrl, bg="#f5f5f5")
        left.pack(side=tk.LEFT, padx=12)
        tk.Label(left, text="탐지 모델", font=("Arial", 10, "bold"),
                 bg="#f5f5f5", fg="#333333").pack(side=tk.LEFT, padx=(0, 8))
        self.anomaly_model_var = tk.StringVar(value="ae")
        for val, lbl in (("ae", "AutoEncoder"), ("patchcore", "PatchCore")):
            tk.Radiobutton(left, text=lbl, variable=self.anomaly_model_var,
                           value=val, font=("Arial", 10), bg="#f5f5f5",
                           command=self._on_anomaly_model_change).pack(side=tk.LEFT, padx=4)

        # 판정 결과
        self.lbl_anomaly_status = tk.Label(main_frame, text="⌛ 이미지 드롭 대기 중",
                                           font=("Arial", 14, "bold"), fg="gray")
        self.lbl_anomaly_status.pack(pady=(0, 12))

        # 시각화
        visual_container = tk.Frame(main_frame)
        visual_container.pack(expand=False)
        visual_container.columnconfigure(0, weight=1, uniform="anomaly_view")
        visual_container.columnconfigure(1, weight=1, uniform="anomaly_view")

        left_view = tk.Frame(visual_container)
        left_view.grid(row=0, column=0, sticky="n", padx=(20, 80))
        tk.Label(left_view, text="검사 대상 이미지 (Input)",
                 font=("Arial", 13, "bold"), fg="#1a1a1a").pack(pady=(0, 5))
        self.canvas_anomaly_query = tk.Canvas(left_view,
                                              width=CANVAS_SIZE[0], height=CANVAS_SIZE[1],
                                              bg="white", bd=0,
                                              highlightthickness=1, highlightbackground="#dddddd")
        self.canvas_anomaly_query.pack()
        self.lbl_anomaly_query_name = tk.Label(left_view, text="",
                                               font=("Arial", 10), fg="#888888")
        self.lbl_anomaly_query_name.pack(pady=(3, 0))

        right_view = tk.Frame(visual_container)
        right_view.grid(row=0, column=1, sticky="n", padx=(80, 20))
        tk.Label(right_view, text="이상 부위 추적 맵 (Heatmap)",
                 font=("Arial", 13, "bold"), fg="#1a1a1a").pack(pady=(0, 5))
        self.canvas_anomaly_heatmap = tk.Canvas(right_view,
                                                width=CANVAS_SIZE[0], height=CANVAS_SIZE[1],
                                                bg="white", bd=0,
                                                highlightthickness=1, highlightbackground="#dddddd")
        self.canvas_anomaly_heatmap.pack()
        self.lbl_anomaly_heatmap_name = tk.Label(right_view, text="",
                                                 font=("Arial", 10), fg="#888888")
        self.lbl_anomaly_heatmap_name.pack(pady=(3, 0))


    def clear_similarity_display(self):
        """초기화 메서드"""
        self.canvas_query.delete("all")
        self.lbl_query_name.config(text="")
        self.canvas_best.delete("all")
        self.lbl_best_name.config(text="")
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

    def clear_anomaly_display(self):
        """초기화 메서드"""
        self.canvas_anomaly_query.delete("all")
        self.lbl_anomaly_query_name.config(text="")
        self.canvas_anomaly_heatmap.delete("all")
        self.lbl_anomaly_heatmap_name.config(text="")
        self.lbl_anomaly_status.config(text="⌛ 이미지 드롭 대기 중", fg="gray")


    def switch_model(self, new_model_name):
        self.clear_similarity_display()
        self.center_status_label.config(text=f"⌛ {new_model_name.upper()} 로딩 중...", fg="orange")
        self.update()
        try:
            self.embedder = Embedder(model_name=new_model_name, device=DEVICE)
        except Exception as e:
            messagebox.showerror("모델 로드 실패", f"{new_model_name.upper()} 모델을 불러올 수 없습니다.\n{e}")
            self.center_status_label.config(text="❌ 모델 변경 실패", fg="red")
            return
        try:
            base, ext = os.path.splitext(DB_PATH)
            db_path = f"{base}_{new_model_name.lower()}{ext}"
            if not os.path.exists(db_path):
                self.center_status_label.config(text=f"⌛ {new_model_name.upper()} DB 생성 중...", fg="orange")
                self.update()
                self.db = self.build_db()
            else:
                with open(db_path, "rb") as f:
                    self.db = pickle.load(f)
                self.db = self.fix_db_dimensions(self.db)
        except Exception as e:
            messagebox.showerror("DB 로드 실패", f"임베딩 DB를 불러올 수 없습니다.\n{e}")
            self.center_status_label.config(text="❌ DB 로드 실패", fg="red")
            return
        db_count = len(self.db) if self.db else 0
        self.stat_label.config(text=f"DB: {db_count}개 이미지")
        self.model_info_label.config(text=f"현재 모델: {new_model_name.upper()} ({DEVICE})")
        self.center_status_label.config(text="✅ 모델 변경 완료!", fg="green")
        self.current_model_name = new_model_name.lower()

    def _on_similarity_model_change(self):
        new_model = self.similarity_model_var.get()
        if new_model == self.current_model_name:
            return
        if not messagebox.askyesno("모델 변경 확인",
                                   f"모델을 {new_model.upper()}(으)로 변경하시겠습니까?"):
            self.similarity_model_var.set(self.current_model_name)
            return
        self.switch_model(new_model)
        self.similarity_model_var.set(self.current_model_name)
    
    def _on_anomaly_model_change(self):
        self.clear_anomaly_display()

    def clear_similarity_display(self):
        self.canvas_query.delete("all")
        self.lbl_query_name.config(text="")
        self.canvas_best.delete("all")
        self.lbl_best_name.config(text="")
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

    def clear_anomaly_display(self):
        self.canvas_anomaly_query.delete("all")
        self.lbl_anomaly_query_name.config(text="")
        self.canvas_anomaly_heatmap.delete("all")
        self.lbl_anomaly_heatmap_name.config(text="")
        self.lbl_anomaly_status.config(text="⌛ 이미지 드롭 대기 중", fg="gray")


    def get_current_db_path(self):
        """ DB명 처리: 현재 설정된 모델명에 맞춰 고유한 pkl 경로를 반환"""
        base, ext = os.path.splitext(DB_PATH)
        return f"{base}_{MODEL_NAME.lower()}{ext}"
    
    def load_db(self):
        """DB 로드"""
        current_db_path = self.get_current_db_path()
        if not os.path.exists(current_db_path):
            if VERBOSE:
                print(f"현재 모델({MODEL_NAME})의 DB가 없어 새로 생성합니다.")
            return self.build_db()
        try:
            with open(current_db_path, "rb") as f:
                db = pickle.load(f)
                if VERBOSE:
                    print(f"✅ {MODEL_NAME}] 임베딩 DB 로드 완료 ({len(db)}개 이미지)")
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
                print("🔧 [{MODEL_NAME}] DB 차원 문제 수정 및 재저장 중...")
            try:
                with open(self.get_current_db_path, "wb") as f:
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
            print(f"📦 [{MODEL_NAME}] {len(image_files)}개 이미지 임베딩 생성 중...")
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
            current_db_path = self.get_current_db_path()
            with open(current_db_path, "wb") as f:
                pickle.dump(db, f)
            if VERBOSE:
                print(f"✅ DB 저장 완료: {current_db_path} ({len(db)}개 임베딩)")
        except Exception as e:
            if LOG_ERRORS:
                print(f"⚠️ DB 저장 실패: {e}")

        return db

    def rebuild_database(self):
        """데이터베이스 재구축"""
        if not messagebox.askyesno("확인", f"현재 모델({MODEL_NAME})의 기존 DB를 지우고 재구축하시겠습니까?"):
            return
        
        try:
            self.center_status_label.config(text=f"🔄 [{MODEL_NAME}] 데이터베이스 재구축 중...", fg="orange")
            self.update()
            # 기존 DB 파일 삭제
            current_db_path = self.get_current_db_path()
            if os.path.exists(current_db_path):
                os.remove(current_db_path)
            
            # 새 DB 생성
            self.db = self.build_db()
            db_count = len(self.db) if self.db else 0
            self.stat_label.config(text=f"DB: {db_count}개 이미지")
            self.center_status_label.config(text="✅ 재구축 완료!", fg="green")

        except Exception as e:
            error_msg = f"DB 재구축 실패: {str(e)}"
            self.stat_label.config(text=error_msg, fg="red")
            messagebox.showerror("오류", error_msg)
            if LOG_ERRORS:
                print(f"❌ DB 재구축 오류: {e}")   
    

    def handle_drop(self, event):
        """파일 드롭 처리 : 드롭 이벤트를 현재 탭 위치에 따라 분기 처리"""
        filepath = event.data.strip('{}').strip('"').strip("'")

        if not os.path.isfile(filepath) or not filepath.lower().endswith(SUPPORTED_FORMATS):
            messagebox.showerror("오류", "유효한 경로 또는 형식의 이미지 파일을 드롭하세요.")
            return

        # 탭 위치 인덱스 확인 (0: 유사도 검색 탭, 1: 이상 탐지 탭)
        current_tab_idx = self.notebook.index(self.notebook.select())

        # 가동 안내 공통 업데이트
        self.show_query_image(filepath)

        # [유사도 검색 탭 활성화 시 연산]
        if current_tab_idx == 0:
            if not self.embedder or not self.db:
                messagebox.showerror("오류", "유사도 모델이 로드되지 않았습니다.")
                return
            try:
                self.center_status_label.config(text="🔍 유사 이미지 검색 중...", fg="orange")
                self.update()
                query_vec = self.embedder.get_embedding(filepath)
                results = search_similar(query_vec, self.db, top_k=DEFAULT_TOP_K)

                # 상위 탭 분기 호출 (유사도 전용 데이터 전달)
                self.show_results(results=results)
                self.center_status_label.config(text="✅ 검색 완료!", fg="green")
            except Exception as e:
                messagebox.showerror("유사도 오버랩 오류", f"검색 실패: {e}")

        # [이상 탐지 탭 활성화 시 연산]
        else:
            try:
                self.lbl_anomaly_status.config(text="🔍 AI 이상치 탐지 연산 중...", fg="orange")
                self.update()
                
                # 카테고리 추출을 위한 가벼운 유사도 검색 (임시 카테고리 추출 연산을 UI 연동 구조 안으로 내재화)
                temp_query_vec = self.embedder.get_embedding(filepath)
                temp_results = search_similar(temp_query_vec, self.db, top_k=1)
                filename_no_ext = os.path.splitext(temp_results[0][0])[0]  
                category = filename_no_ext.split("_")[0] if "_" in filename_no_ext else filename_no_ext
                
                # AI 이상치 추론 엔진 구동
                selected_model = self.anomaly_model_var.get()
                if selected_model == "patchcore":
                    anomaly_score, anomaly_status, heatmap = run_patchcore_inference(
                        filepath, category, MB_DIR)
                else:
                    anomaly_score, anomaly_status, heatmap = run_anomaly_inference(filepath, category)

                # 상위 탭 분기 호출 (이상치 전용 데이터 전달)
                self.show_results(results=None, anomaly_score=anomaly_score, anomaly_status=anomaly_status)
                
                # 우측 맵 시각화 연동
                self.show_reconstructed_heatmap(heatmap)
                
            except Exception as e:
                self.lift() # 에러 메시지가 항상 앞에 오도록 설정
                self.focus_force()                
                messagebox.showerror("이상치 추론 오류", f"분석 실패: {e}")
                self.lbl_anomaly_status.config(text="❌ 분석 실패", fg="red")


    def show_reconstructed_heatmap(self, heatmap_img):
        """에러맵 PIL Image를 우측 Heatmap 캔버스에 드로잉"""
        try:
            heatmap_img = heatmap_img.resize(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
            w, h = heatmap_img.size
            self.anomaly_heatmap_imgtk = ImageTk.PhotoImage(heatmap_img)
            self.canvas_anomaly_heatmap.delete("all")
            self.canvas_anomaly_heatmap.config(width=w, height=h)
            self.canvas_anomaly_heatmap.create_image(w//2, h//2, image=self.anomaly_heatmap_imgtk, anchor=tk.CENTER)
            self.lbl_anomaly_heatmap_name.config(text="|Input - Reconstructed| Error Map")
        
        except Exception as e:
            print(f"⚠️ Heatmap 시각화 실패: {e}")    
    
    def show_query_image(self, filepath):
        """쿼리 이미지 표시"""
        try:
            img = Image.open(filepath)
            img.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
            thumb_w, thumb_h = img.size
            
            # 현재 선택된 탭이 어디냐에 따라 이미지를 띄워주는 캔버스를 분기
            current_tab_idx = self.notebook.index(self.notebook.select())
            
            if current_tab_idx == 0:  # 1번 유사도 검색 탭일 때
                self.query_imgtk = ImageTk.PhotoImage(img)
                self.canvas_query.delete("all")
                self.canvas_query.config(width=thumb_w, height=thumb_h)
                self.canvas_query.create_image(thumb_w//2, thumb_h//2, image=self.query_imgtk, anchor=tk.CENTER)
                self.lbl_query_name.config(text=os.path.basename(filepath)) # 검색 캔버스 및 파일명
            
            else:                     # 2번 이상치 탐지 탭일 때
                self.anomaly_imgtk = ImageTk.PhotoImage(img) # 별도 이미지 변수 유지
                self.canvas_anomaly_query.delete("all")
                self.canvas_anomaly_query.config(width=thumb_w, height=thumb_h)
                self.canvas_anomaly_query.create_image(thumb_w//2, thumb_h//2, image=self.anomaly_imgtk, anchor=tk.CENTER)
                self.lbl_anomaly_query_name.config(text=os.path.basename(filepath)) # 검색 캔버스 및 파일명

        except Exception as e:
            if LOG_ERRORS:
                print(f"⚠️ 쿼리 이미지 표시 오류: {e}")

    def show_results(self, results, anomaly_score=None, anomaly_status=None):
        """
        통합 결과 표시 함수: 
        전달받은 인자에 따라 현재 활성화된 탭의 UI를 업데이트
        """
        current_tab_idx = self.notebook.index(self.notebook.select()) # 현재 선택된 탭 확인

        if current_tab_idx == 0:  # 유사도 검색 탭일 때
            self.update_similarity_results(results)
        else:  # 이상치 탐지 탭일 때
            self.update_anomaly_results(anomaly_score, anomaly_status)

    def update_anomaly_results(self, anomaly_score, anomaly_status):
        """이상 탐지 label을 결과 구역에 출력 및 갱신 기능"""
        if anomaly_score is None and anomaly_status is None:
            self.lbl_anomaly_status.config(text="❌ 분석 실패 (데이터 없음)", fg="gray")
            return

        # 상단 스탬프 구역에 색상과 스코어를 결합
        if anomaly_status and "anom" in anomaly_status.lower():
            status_text = f"🚨 ANOMALY (불량)  |  Score: {anomaly_score:.5f}"
            status_color = "red"
        else:
            status_text = f"🟢 NORMAL (정상)  |  Score: {anomaly_score:.5f}"
            status_color = "green"

        # 레이블 컴포넌트 실시간 갱신
        self.lbl_anomaly_status.config(text=status_text, fg=status_color)

    def update_similarity_results(self, results):
        if not results:
            no_result_label = tk.Label(self.scrollable_frame, text="검색 결과가 없습니다.",
                                       font=("Arial", 12), fg="gray")
            no_result_label.pack(pady=20)
            self.canvas_best.delete("all")
            self.lbl_best_name.config(text="")
            return

        best_fname, best_score = results[0]
        try:
            best_path = os.path.join(IMAGE_DIR, best_fname)
            if os.path.exists(best_path):
                best_img = Image.open(best_path)
                best_img.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                best_w, best_h = best_img.size

                self.canvas_best.config(width=best_w, height=best_h)
                self.best_imgtk = ImageTk.PhotoImage(best_img)
                self.canvas_best.delete("all")
                
                cx, cy = best_w // 2, best_h // 2
                self.canvas_best.create_image(cx, cy, image=self.best_imgtk, anchor=tk.CENTER)
                self.lbl_best_name.config(text=best_fname)

            else:
                # DB 예외 처리: 실제 이미지가 images 폴더에 없을 경우 텍스트 대체
                self.canvas_best.delete("all")
                self.canvas_best.create_text(CANVAS_SIZE[0]//2, CANVAS_SIZE[1]//2, text="이미지 파일 없음\n(pkl 데이터로 매칭)", justify=tk.CENTER)
                self.lbl_best_name.config(text=best_fname)
        except Exception as e:
            print(f"우측 베스트 결과 표시 오류: {e}")
        
        # 하단 전체 결과 청소 후 새로 고침 고정
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # 하단 전체 결과 리스트 가로 배치
        result_frame = tk.Frame(self.scrollable_frame)
        result_frame.pack(fill=tk.X, padx=10, pady=10)

        for i, (fname, score) in enumerate(results):
            try:
                # 개별 결과 프레임
                item_frame = tk.Frame(result_frame, relief=tk.RAISED, bd=1)
                item_frame.grid(row=0, column=i, padx=10, pady=2, sticky="n")

                # 이미지 위에 순위 표시 (1부터 시작하므로 i + 1)
                rank_label = tk.Label(item_frame, text=str(i + 1), font=("Arial", 12, "bold"))
                rank_label.pack(pady=(2, 1))

                # 이미지 표시
                path = os.path.join(IMAGE_DIR, fname)
                if os.path.exists(path):
                    img = Image.open(path)
                    img.thumbnail(DISPLAY_SIZE, Image.Resampling.LANCZOS) # 썸네일 크기 조정
                    imgtk = ImageTk.PhotoImage(img) # 썸네일로 만든 이미지를 ImageTk로 변환

                    panel = tk.Label(item_frame, image=imgtk)
                    panel.image = imgtk  # 참조 유지
                    panel.pack(pady=1)
                else:
                    # DB 예외 처리: 하단 목록에서도 원본 이미지가 없으면 대체 텍스트 상자 배치
                    panel = tk.Label(item_frame, text="No Image\nFile", width=12, height=6, bg="lightgray", fg="black")
                    panel.pack(pady=1)

                # 파일명 및 유사도 점수 표시
                info_text = f"{fname}\n유사도: {score:.3f}"
                label = tk.Label(item_frame, text=info_text, font=("Arial", 10),
                                 justify=tk.CENTER, wraplength=150)
                label.pack(pady=1)

            except Exception as e:
                if LOG_ERRORS:
                    print(f"⚠️ 결과 표시 중 오류 ({fname}): {e}")


if __name__ == "__main__":
    # 설정 확인
    if not os.path.exists(IMAGE_DIR):
        print(f"❗ '{IMAGE_DIR}' 폴더를 생성합니다.")
        ensure_directories()

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(SUPPORTED_FORMATS)]
    
    base, ext = os.path.splitext(DB_PATH)
    current_model_db_path = f"{base}_{MODEL_NAME.lower()}{ext}"

    # DB 예외 처리: 이미지가 없어도 .pkl 캐시 파일이 있다면 그대로 진행하도록 조건 수정
    if not image_files and not os.path.exists(current_model_db_path):
        print(f"⚠️ '{IMAGE_DIR}' 폴더에 이미지가 없고, 현재 모델({MODEL_NAME})의 기존 pkl 파일도 없습니다.")
        print(f"   폴더에 이미지를 넣거나 백업 DB를 두고 다시 실행하세요.")
        print(f"   지원 형식: {SUPPORTED_FORMATS}")
        input("Enter를 눌러 종료...")
    else:
        if VERBOSE:
            print(f"🚀 GUI 앱 시작 중... (모델: {MODEL_NAME} / 발견된 이미지 파일: {len(image_files)}개)")
        app = ImageSearchApp()
        app.mainloop()