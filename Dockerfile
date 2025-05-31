# CUDA + Python 베이스 이미지
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Python 설치
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt update && \
    apt install -y software-properties-common ffmpeg git curl tzdata && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.11 python3.11-distutils python3.11-venv python3.11-dev && \
    curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.11 get-pip.py && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/local/bin/pip /usr/bin/pip && \
    rm get-pip.py
    
# pip 최신화
RUN python -m pip install --upgrade pip

# 작업 디렉토리 설정
WORKDIR /app

# requirements 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Whisper 모델 미리 다운로드
RUN python -c "import whisper; whisper.load_model('large')"

# 앱 복사
COPY ./app .

# FastAPI 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
