# CUDA + Python 베이스 이미지
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Python 설치
RUN apt update && apt install -y python3 python3-pip ffmpeg

# pip 최신화
RUN python3 -m pip install --upgrade pip

# 작업 디렉토리 설정
WORKDIR /app

# requirements 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Whisper 모델 미리 다운로드
RUN python3 -c "import whisper; whisper.load_model('large')"

# 앱 복사
COPY ./app ./app

# FastAPI 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
