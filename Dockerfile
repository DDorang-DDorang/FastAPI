# CUDA 기반 이미지 사용
FROM nvidia/cuda:12.2.0-base

# Python 설치
RUN apt update && \
    apt install -y python3 python3-pip && \
    ln -sf python3 /usr/bin/python && \
    ln -sf pip3 /usr/bin/pip

# Whisper 모델 미리 다운로드
RUN pip install --no-cache-dir git+https://github.com/openai/whisper && \
    python -c "import whisper; whisper.load_model('large')"

# 작업 디렉토리 설정
WORKDIR /app

# requirements 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 코드 복사
COPY ./app ./app

# FastAPI 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
