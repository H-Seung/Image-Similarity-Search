# 이미지 유사도 검색기 (Image Similarity Search)

OpenAI CLIP 또는 ResNet-50 모델을 기반으로 한 이미지 유사도 검색기입니다. </br>
Drag & Drop 방식 GUI 를 통해 유사한 이미지를 빠르게 검색할 수 있습니다.
![result](assets/result_similarity.png)

## 1. 주요 기능

- **Drag & Drop  인터페이스**: 간단하게 이미지를 끌어다 놓아 검색
- **OpenAI CLIP / ResNet 기반**: OpenAI의 CLIP 또는 ResNet 모델 선택
- **빠른 검색**: 사전 계산된 임베딩 데이터베이스로 빠른 유사도 검색
- **다양한 이미지 포맷 지원**: JPG, PNG, JPEG, JFIF, BMP, TIFF
- **GPU 가속**: CUDA 지원 시 자동으로 GPU 사용

## 2. 프로젝트 구조

```
Parts_checker/
├── data/
│   ├── images/              # 이미지 DB
│   └── embeddings_db_{model_name}.pkl    # 임베딩 데이터베이스 (자동 생성)
├── models/
│   └── embedder.py          # 이미지 임베딩 추출기
├── utils/
│   └── search.py            # 유사도 검색 함수
├── search_gui_app.py        # GUI 기반 검색 앱
├── config.py                # 설정 파일
├── requirements.txt         # 의존성 패키지 목록
└── README.md
```

## 3.️ 설치 및 설정

### 1. 필수 요구사항

- Python 3.10
- CUDA 12.1 (GPU 사용 시)

### 2. 패키지 설치

GPU 사용 시:
```bash
# PyTorch (CUDA 버전)
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# 나머지 패키지들
pip install -r requirements.txt
```

CPU만 사용하는 경우:
```bash
pip install torch torchvision
pip install -r requirements.txt
```

### 3. 이미지 DB 준비

`data/images/` 폴더에 DB로 쓸 이미지를 넣어주세요:

```bash
mkdir -p data/images
# 이미지 파일들을 data/images/ 폴더에 복사
```

## 4. 사용 방법

### GUI 앱 실행

```bash
python search_gui_app.py
```

1. 애플리케이션이 시작되면 자동으로 임베딩 데이터베이스를 생성합니다
2. 검색할 이미지를 드래그 앤 드롭하세요
3. 상위 k개(default=5, config.py에서 변경 가능)의 유사한 이미지가 표시됩니다


## 5. 설정 옵션

`config.py`에서 다음 설정을 변경할 수 있습니다:

- 모델 및 디바이스 설정
```python
MODEL_NAME = "clip"  # 또는 "resnet"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```
- 경로 설정
```python
IMAGE_DIR = "data/images"
DB_PATH = "data/embeddings_db.pkl"
```
- 이미지/GUI 설정
```python
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.jfif', '.bmp', '.tiff', '.webp')
THUMBNAIL_SIZE = (224, 224)
DISPLAY_SIZE = (150, 150)
WINDOW_SIZE = "900x700"
```
- 검색 결과 설정
```python
DEFAULT_TOP_K = 5
MAX_TOP_K = 10
```


## 6. 성능 최적화

### GPU 사용
- NVIDIA GPU가 있는 경우 자동으로 CUDA를 사용합니다
- GPU 메모리가 부족한 경우 `BATCH_SIZE`를 조정하세요

### 대용량 이미지 처리
- 이미지는 자동으로 224x224로 리사이즈됩니다
- 원본 이미지 품질에는 영향을 주지 않습니다

## 7. 업그레이드 및 유지보수

### 데이터베이스 재생성
gui 앱에서 우측 상단 'DB 재생성' 버튼 클릭

### 모델 변경
다른 모델을 사용하려면 `config.py`에서 `MODEL_NAME`을 변경하거나,</br>
gui 앱에서 우측 상단 '모델 재설정' 버튼 클릭:
- `"clip"`: OpenAI CLIP 
- `"resnet"`: ResNet-50

## 8. 문제 해결

### 일반적인 오류들

**"only integer tensors of a single element can be converted to an index"**
- 빈 데이터베이스이거나 손상된 이미지 파일이 원인
- `data/images/` 폴더에 유효한 이미지가 있는지 확인

**메모리 부족 오류**
- GPU 메모리가 부족한 경우 CPU 모드로 전환: `DEVICE = "cpu"`
- 큰 이미지 파일들을 사전에 압축

**느린 검색 속도**
- 첫 실행 시 임베딩 생성에 시간이 걸립니다
- 이후 실행에서는 저장된 데이터베이스를 사용해 빠릅니다

### 로그 확인

애플리케이션 실행 시 콘솔에 다음과 같은 정보가 출력됩니다:
- 모델 로딩 상태
- 데이터베이스 생성/로드 진행상황
- 검색 결과 및 오류 메시지
