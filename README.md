# 설치 명령어

`python -m pip install --upgrade pip setuptools wheel`

`pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 --index-url https://download.pytorch.org/whl/cu121 
&& pip install git+https://github.com/openai/CLIP.git "numpy<2.0" "scikit-learn>=1.3.0" "matplotlib>=3.7.0" Pillow tqdm ftfy regex`

## 설치 확인 코드

`python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
`
-> </br> 
2.3.0 </br>
12.1 </br>
True

---
# 실행 예시 (CLI)

### 1. DB 생성 + 유사 이미지 검색
`python main.py --query data/images/test.png --rebuild`

### 2. 기존 DB 재사용
`python main.py --query data/images/011.png`