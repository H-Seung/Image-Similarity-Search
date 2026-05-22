import os
import pickle
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import tkinterdnd2 as tkdnd
import torch

from config import *
from models.embedder import Embedder
from utils.search import search_similar
from models.anomaly_detector_encoder import run_anomaly_inference


class ImageSearchApp(tkdnd.TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vision Intelligence Platform")
        self.geometry(WINDOW_SIZE)
        self.resizable(True, True)

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
        
    def setup_similarity_ui(self, parent):    
        # 1. 메인 프레임을 부모 탭(parent) 내부에 생성
        main_frame = tk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 2. 드래그앤드롭 안내 
        self.label = tk.Label(main_frame, 
                              text="검색하려는 이미지를 드래그 앤 드롭하면,\nDB 내 이미지에서 유사한 이미지를 검색합니다.",
                              font=("Arial", 12), fg="black", justify=tk.CENTER)
        self.label.pack(pady=10)
        
        # 3. 상단 정보 및 제어 영역 프레임 (3분할 grid 구조)
        info_frame = tk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 11))

        # 좌, 우, 가운데가 완벽한 비율을 갖도록 열(column) 가중치 설정
        info_frame.columnconfigure(0, weight=1, uniform="top_info")  # 왼쪽 구역
        info_frame.columnconfigure(1, weight=1, uniform="top_info")  # 가운데 구역
        info_frame.columnconfigure(2, weight=1, uniform="top_info")  # 오른쪽 구역

        # [왼쪽 구역] DB 개수 및 현재 모델 정보 
        left_info_subframe = tk.Frame(info_frame)
        left_info_subframe.grid(row=0, column=0, sticky="w", padx=10)

        # 상태 및 DB 개수 레이블
        self.stat_label = tk.Label(left_info_subframe, text="DB: 로딩 중...", font=("Arial", 11), fg="black")
        self.stat_label.pack(anchor="w")

        # 현재 선택한 모델 정보 표시 레이블
        self.model_info_label = tk.Label(left_info_subframe, text=f"현재 모델: {MODEL_NAME.upper()} ({DEVICE})", 
                                         font=("Arial", 11), fg="black")
        self.model_info_label.pack(anchor="w", pady=(2, 0))

        # [가운데 구역] 상태 문구
        center_info_subframe = tk.Frame(info_frame)
        center_info_subframe.grid(row=0, column=1, sticky="nsew") 

        # 상태 알림용 중앙 레이블 (초기에는 빈 값 혹은 로딩 상태, 나중에 "✅ 검색 완료!" 등으로 변경)
        self.center_status_label = tk.Label(center_info_subframe, text="⌛ 준비 중...", font=("Arial", 12, "bold"), fg="black")
        self.center_status_label.pack(expand=True)

        # [오른쪽 구역] 버튼 제어 영역 
        right_btn_subframe = tk.Frame(info_frame)
        right_btn_subframe.grid(row=0, column=2, sticky="e", padx=10)

        # DB 재생성 버튼
        self.btn_rebuild = tk.Button(right_btn_subframe, text="DB 재생성", command=self.rebuild_database, font=("Arial", 11), width=12)
        self.btn_rebuild.pack(anchor="e", pady=(0, 4))

        # 모델 재설정 버튼
        self.btn_change_model = tk.Button(right_btn_subframe, text="모델 재설정", command=self.open_model_selection_popup, font=("Arial", 11), width=12)
        self.btn_change_model.pack(anchor="e")
        
        # 4. 쿼리(검색대상) 이미지 및 매칭 결과 2분할 표시 컨테이너 프레임
        query_match_container = tk.Frame(main_frame)
        query_match_container.pack(pady=10)

        # [왼쪽] 쿼리 이미지 표시 영역
        query_frame = tk.Frame(query_match_container, padx=20)
        query_frame.grid(row=0, column=0, sticky="n")

        tk.Label(query_frame, text="검색 이미지", font=("Arial", 14, "bold")).pack(pady=5)
        self.canvas_query = tk.Canvas(query_frame, width=CANVAS_SIZE[0], height=CANVAS_SIZE[1],
                                      bg="white", bd=0, highlightthickness=0)
        self.canvas_query.pack()
        self.lbl_query_name = tk.Label(query_frame, text="", font=("Arial", 10)) # 파일명 레이블
        self.lbl_query_name.pack(pady=2)

        # [오른쪽] 가장 유사한 이미지 표시 영역
        best_match_frame = tk.Frame(query_match_container, padx=20)
        best_match_frame.grid(row=0, column=1, sticky="n", padx=(80, 20))

        tk.Label(best_match_frame, text="유사 이미지", font=("Arial", 14, "bold")).pack(pady=5)
        self.canvas_best = tk.Canvas(best_match_frame, width=CANVAS_SIZE[0], height=CANVAS_SIZE[1],
                                     bg="white", bd=0, highlightthickness=1, highlightbackground="cyan")
        self.canvas_best.pack()
        self.lbl_best_name = tk.Label(best_match_frame, text="", font=("Arial", 10)) # 파일명 레이블
        self.lbl_best_name.pack(pady=2)

        # 5. 결과 하단 스크롤 영역
        result_label_frame = tk.Frame(main_frame)
        result_label_frame.pack(fill=tk.X, pady=(16, 1))
        tk.Label(result_label_frame, text=f"유사한 이미지 (상위 {DEFAULT_TOP_K}개)",
                 font=("Arial", 12, "bold")).pack()

        # 스크롤 가능한 결과 프레임
        self.result_canvas = tk.Canvas(main_frame, height=220)
        scrollbar = ttk.Scrollbar(main_frame, orient="horizontal", command=self.result_canvas.xview)
        self.scrollable_frame = ttk.Frame(self.result_canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all"))
        )

        self.result_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.result_canvas.configure(xscrollcommand=scrollbar.set)

        self.result_canvas.pack(side="top", fill="both", expand=True)
        scrollbar.pack(side="top", fill="x")

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

    def setup_anomaly_ui(self, parent):
        # 1. 메인 프레임을 부모 탭(parent) 내부에 생성 
        main_frame = tk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 2. 상단 안내 문구 
        anomaly_label = tk.Label(main_frame, 
                                 text="분석하려는 이미지를 드래그 앤 드롭하면,\nAI 모델이 이미지의 이상(불량) 여부를 탐지합니다.",
                                 font=("Arial", 12), fg="black", justify=tk.CENTER)
        anomaly_label.pack(pady=10)

        # 3. 드롭된 이미지를 보여줄 중앙 쿼리 이미지 영역
        img_frame = tk.Frame(main_frame)
        img_frame.pack(pady=10)

        tk.Label(img_frame, text="분석 대상 이미지", font=("Arial", 14, "bold")).pack(pady=5)
        self.canvas_anomaly_query = tk.Canvas(img_frame, width=CANVAS_SIZE[0], height=CANVAS_SIZE[1],
                                              bg="white", bd=0, highlightthickness=0)
        self.canvas_anomaly_query.pack()
        
        self.lbl_anomaly_query_name = tk.Label(img_frame, text="", font=("Arial", 10)) # 파일명 레이블
        self.lbl_anomaly_query_name.pack(pady=2)

        # 4. 하단 결과 표시 영역 
        result_label_frame = tk.Frame(main_frame)
        result_label_frame.pack(fill=tk.X, pady=(20, 3))
        tk.Label(result_label_frame, text="이상치 분석 결과", font=("Arial", 12, "bold")).pack()

        self.anomaly_canvas = tk.Canvas(main_frame, height=150)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.anomaly_canvas.yview)
        
        # 중요 변수명: 유사도 탭과 겹치지 않게 'self.anomaly_scrollable_frame'으로 명명
        self.anomaly_scrollable_frame = ttk.Frame(self.anomaly_canvas)

        self.anomaly_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.anomaly_canvas.configure(scrollregion=self.anomaly_canvas.bbox("all"))
        )

        self.anomaly_canvas.create_window((0, 0), window=self.anomaly_scrollable_frame, anchor="nw")
        self.anomaly_canvas.configure(yscrollcommand=scrollbar.set)

        self.anomaly_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


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

    def open_model_selection_popup(self):
        """⚙️ 모델 재설정: CLIP / ResNet을 선택할 수 있는 팝업창을 띄웁니다."""
        popup = tk.Toplevel(self)
        popup.title("모델 변경")
        popup.geometry("300x150")
        popup.resizable(False, False)
        
        # 팝업을 메인 창 중앙 근처에 띄우기
        popup.grab_set() 
        
        lbl = tk.Label(popup, text="변경할 모델을 선택하세요:", font=("Arial", 11, "bold"))
        lbl.pack(pady=15)
        
        # 라디오 버튼 변수 (현재 활성화된 모델을 기본값으로 선택)
        selected_model = tk.StringVar(value=MODEL_NAME.lower())
        
        rb_frame = tk.Frame(popup)
        rb_frame.pack(pady=5)
        
        rb_clip = tk.Radiobutton(rb_frame, text="CLIP", variable=selected_model, value="clip", font=("Arial", 10))
        rb_clip.pack(side=tk.LEFT, padx=20)
        
        rb_resnet = tk.Radiobutton(rb_frame, text="ResNet", variable=selected_model, value="resnet", font=("Arial", 10))
        rb_resnet.pack(side=tk.LEFT, padx=20)
        
        def on_confirm():
            new_model = selected_model.get()
            popup.destroy() # 팝업 닫기
            
            # 사용자가 현재 모델과 다른 것을 골랐을 때만 변경 진행
            if new_model != MODEL_NAME.lower():
                if messagebox.askyesno("모델 변경 확인", f"모델을 {new_model.upper()}(으)로 변경하시겠습니까?\n변경 후 앱이 자동으로 재시작됩니다."):
                    self.update_config_file(new_model)
            else:
                messagebox.showinfo("알림", "현재 이미 선택되어 있는 모델입니다.")

        btn_confirm = tk.Button(popup, text="적용", command=on_confirm, font=("Arial", 10, "bold"), width=10, bg="lightgray")
        btn_confirm.pack(pady=10)

    def update_config_file(self, new_model_name):
        """config.py 파일을 읽어서 MODEL_NAME 설정을 변경하고 앱을 재시작합니다."""
        config_path = "config.py"
        if not os.path.exists(config_path):
            messagebox.showerror("오류", "config.py 파일을 찾을 수 없습니다.")
            return

        try:
            # 1. config.py 파일 읽기
            with open(config_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # 2. MODEL_NAME이 선언된 줄을 찾아 치환
            updated = False
            for i, line in enumerate(lines):
                # 주석이 아니고 MODEL_NAME 변수 설정 선언문인 경우 검색
                if line.strip().startswith("MODEL_NAME") and "=" in line:
                    lines[i] = f"MODEL_NAME = \"{new_model_name}\"\n"
                    updated = True
                    break

            if not updated:
                messagebox.showerror("오류", "config.py 내에서 MODEL_NAME 변수를 찾지 못했습니다.")
                return

            # 3. 변경 내용 반영하여 다시 쓰기
            with open(config_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

            # 4. 앱 프로세스 자동 재시작
            import sys
            import subprocess
            
            messagebox.showinfo("재시작", "설정이 저장되었습니다. 앱을 재시작합니다.")
            self.destroy() # 현재 tkinter 창을 완전히 닫음
            
            # 현재 실행 중인 파이썬 스크립트 파일 경로를 가져와 새 프로세스로 재실행
            subprocess.Popen([sys.executable] + sys.argv)
            sys.exit() # 현재 스크립트 프로세스 종료

        except Exception as e:
            messagebox.showerror("오류", f"설정 변경 중 오류가 발생했습니다:\n{e}")
    
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
        # DB 재생성 확인 메세지창 알림
        if not messagebox.askyesno("확인", "현재 모델({MODEL_NAME})의 기존 pkl DB를 지우고 새로 빌드하시겠습니까?"):
            return
        
        try:
            self.center_status_label.config(text="🔄 [{MODEL_NAME}] 데이터베이스 재구축 중...", fg="orange")
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
        """파일 드롭 처리"""
        if not self.embedder or not self.db:
            messagebox.showerror("오류", "시스템이 아직 준비되지 않았습니다.")
            return

        # 파일 경로 정리
        filepath = event.data.strip('{}').strip('"').strip("'")

        if not os.path.isfile(filepath):
            messagebox.showerror("오류", "유효한 경로의 이미지 파일을 드롭하세요")
            return

        # 파일 확장자 확인
        if not filepath.lower().endswith(SUPPORTED_FORMATS):
            messagebox.showerror("오류", f"지원하는 이미지 형식이 아닙니다\n{SUPPORTED_FORMATS}")
            return

        try:
            self.center_status_label.config(text="🔍 검색 중...", fg="orange")
            self.update()

            # 쿼리 이미지 표시
            self.show_query_image(filepath)

            # 임베딩 추출 및 유사 이미지 검색
            query_vec = self.embedder.get_embedding(filepath)
            results = search_similar(query_vec, self.db, top_k=DEFAULT_TOP_K)

            # 결과에서 이미지의 파일명만 가져오기 (예: "grid_000.png" -> "grid_000")
            filename_no_ext = os.path.splitext(results[0][0])[0]  
            if "_" in filename_no_ext:
                category = filename_no_ext.split("_")[0]
            else:
                category = filename_no_ext
            # 이상치 탐지
            results_anomaly = run_anomaly_inference(filepath, category)

            # 결과 표시
            self.show_results(results, anomaly_score=results_anomaly[0], anomaly_status=results_anomaly[1])
            self.center_status_label.config(text=f"✅ 검색 완료!", fg="green")

        except Exception as e:
            error_msg = f"검색 중 오류가 발생했습니다: {str(e)}"
            messagebox.showerror("오류", error_msg)
            self.stat_label.config(text="❌ 검색 실패", fg="red")
            if LOG_ERRORS:
                print(f"❌ 드롭 처리 오류: {e}")

    def show_query_image(self, filepath):
        """쿼리 이미지 표시"""
        try:
            img = Image.open(filepath)
            img.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
            thumb_w, thumb_h = img.size
            
            # 현재 선택된 탭이 어디냐에 따라 이미지를 띄워주는 캔버스를 분기합니다.
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
        # 현재 선택된 탭 확인
        current_tab_idx = self.notebook.index(self.notebook.select())

        if current_tab_idx == 0:  # 유사도 검색 탭일 때
            self.update_similarity_results(results)
        else:  # 이상치 탐지 탭일 때
            self.update_anomaly_results(anomaly_score, anomaly_status)

    def update_anomaly_results(self, anomaly_score, anomaly_status):
        """이상치 탐지용 label을 이상치 전용 스크롤 프레임에 출력"""
        # 기존 이상치 결과 컴포넌트 청소
        for widget in self.anomaly_scrollable_frame.winfo_children():
            widget.destroy()

        if anomaly_score is not None or anomaly_status is not None:
            text = ""
            if anomaly_score is not None:
                text += f"이상치 점수: {anomaly_score:.3f}   "
            if anomaly_status is not None:
                text += f"상태: {anomaly_status}"

            big_label = tk.Label(
                self.anomaly_scrollable_frame,
                text=text,
                font=("Arial", 15, "bold"),  
                fg="red" if (anomaly_status and "anom" in anomaly_status.lower()) else "green"
            )
            big_label.pack(pady=(20, 20))

    def update_similarity_results(self, results):
        """1순위 유사 이미지를 상단 우측 '유사 이미지' 영역에 배치"""
        # 기존 이상치 결과 컴포넌트 청소
        for widget in self.anomaly_scrollable_frame.winfo_children():
            widget.destroy()

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
        
        # 하단 전체 결과 리스트 가로 배치
        result_frame = tk.Frame(self.scrollable_frame)
        result_frame.pack(fill=tk.X, padx=10, pady=10)

        for i, (fname, score) in enumerate(results):
            try:
                # 개별 결과 프레임
                item_frame = tk.Frame(result_frame, relief=tk.RAISED, bd=1)
                item_frame.grid(row=0, column=i, padx=10, pady=5, sticky="n")

                # 이미지 위에 순위 표시 (1부터 시작하므로 i + 1)
                rank_label = tk.Label(item_frame, text=str(i + 1), font=("Arial", 12, "bold"))
                rank_label.pack(pady=(2, 1))

                path = os.path.join(IMAGE_DIR, fname)
                if os.path.exists(path):
                    # 이미지 표시
                    img = Image.open(path)
                    img.thumbnail(DISPLAY_SIZE, Image.Resampling.LANCZOS) # 썸네일 크기 조정
                    imgtk = ImageTk.PhotoImage(img) # 썸네일로 만든 이미지를 ImageTk로 변환

                    panel = tk.Label(item_frame, image=imgtk)
                    panel.image = imgtk  # 참조 유지
                    panel.pack(pady=3)
                else:
                    # DB 예외 처리: 하단 목록에서도 원본 이미지가 없으면 대체 텍스트 상자 배치
                    panel = tk.Label(item_frame, text="No Image\nFile", width=12, height=6, bg="lightgray", fg="black")
                    panel.pack(pady=3)

                # 파일명 및 점수 표시
                info_text = f"{fname}\n유사도: {score:.3f}"
                label = tk.Label(item_frame, text=info_text, font=("Arial", 10),
                                 justify=tk.CENTER, wraplength=150)
                label.pack(pady=3)

            except Exception as e:
                if LOG_ERRORS:
                    print(f"⚠️ 결과 표시 중 오류 ({fname}): {e}")


if __name__ == "__main__":
    # 설정 확인
    if not os.path.exists(IMAGE_DIR):
        print(f"❗ '{IMAGE_DIR}' 폴더를 생성합니다.")
        ensure_directories()

    image_files = [f for f in os.listdir(IMAGE_DIR)
                   if f.lower().endswith(SUPPORTED_FORMATS)]
    
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