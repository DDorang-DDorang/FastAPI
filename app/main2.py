import json
import os
import uuid
from typing import Dict

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from pydub.utils import mediainfo

from anxiety.anxiety_score import anxiety_analysis
from voice_analysis import SoundAnalyzer
from whisper_utils import transcribe_audio, calculate_pronunciation_score, calculate_wpm
from gpt import correct_stt_result, get_chat_response, get_compare_result

app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: Dict[str, dict] = {}


# ----------------- 유틸 함수 -----------------

def compute_pronunciation_score(logprobs, threshold=-1.0):
    filtered = [lp for lp in logprobs if lp > threshold]
    score = np.mean([np.exp(lp) for lp in filtered]) if filtered else 0.0
    return score, len(filtered), threshold


def save_upload_file(upload_file: UploadFile, filename: str) -> str:
    """업로드 파일 저장"""
    save_path = os.path.join(UPLOAD_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(upload_file.file.read())
    return save_path


def convert_to_wav(file_path: str) -> str:
    """mp4 -> wav 변환, wav는 그대로 반환"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".mp4":
        wav_path = file_path.rsplit(".", 1)[0] + ".wav"
        audio = AudioSegment.from_file(file_path, format="mp4")
        audio.export(wav_path, format="wav")
        return wav_path
    elif ext == ".wav":
        return file_path
    else:
        raise ValueError("지원되지 않는 파일 형식입니다. wav 또는 mp4만 허용됩니다.")


def merge_chunks(original_filename: str, total_chunks: int) -> str:
    """조각난 파일 합치기"""
    merged_path = os.path.join(UPLOAD_DIR, f"merged_{original_filename}")
    with open(merged_path, "wb") as merged:
        for i in range(total_chunks):
            part_path = os.path.join(UPLOAD_DIR, f"{original_filename}_chunk_{i}")
            with open(part_path, "rb") as part:
                merged.write(part.read())
            os.remove(part_path)
    return merged_path


# ----------------- API -----------------

@app.post("/compare")
async def compare_scripts(script1: str = Form(...), script2: str = Form(...)):
    try:
        result = get_compare_result(script1, script2)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/stt")
async def transcribe(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    metadata: str = Form(...),
    chunk_index: int = Form(default=None),
    total_chunks: int = Form(default=None),
    original_filename: str = Form(default=None),
):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing"}

    # ----------------- 파일 처리 -----------------
    if chunk_index is not None and total_chunks is not None and original_filename:
        # chunked upload
        chunk_filename = f"{original_filename}_chunk_{chunk_index}"
        chunk_path = save_upload_file(video, chunk_filename)
        uploaded_chunks = [
            name for name in os.listdir(UPLOAD_DIR) if name.startswith(original_filename + "_chunk_")
        ]
        if len(uploaded_chunks) < total_chunks:
            return {"status": "chunk_received", "chunk_index": chunk_index}

        save_path = merge_chunks(original_filename, total_chunks)
    else:
        # 일반 업로드
        save_path = save_upload_file(video, f"temp_{job_id}{os.path.splitext(video.filename)[1]}")

    # wav 변환
    wav_path = convert_to_wav(save_path)

    # 백그라운드 처리
    background_tasks.add_task(process_audio_job, job_id, save_path, wav_path, metadata)
    return {"job_id": job_id, "status": "processing"}


# ----------------- 백그라운드 작업 -----------------

def process_audio_job(job_id: str, save_path: str, wav_path: str, metadata: str):
    try:
        target_time = "6:00"
        if metadata:
            meta_data = json.loads(metadata)
            target_time = meta_data.get("target_time", "6:00")

        # ----------------- Whisper -----------------
        result = transcribe_audio(wav_path, language="ko")
        segments = result.get("segments", [])
        all_text = result.get("text", "")

        pron_score, used, threshold = calculate_pronunciation_score(segments)
        wpm, wpm_grade, wpm_comment = calculate_wpm(segments)

        # ----------------- SoundAnalyzer -----------------
        analyzer = SoundAnalyzer(wav_path, threshold=60)
        intensity_grade, avg_db, intensity_comment = analyzer.evaluate_intensity()
        pitch_grade, avg_pitch, pitch_comment = analyzer.evaluate_pitch_score()

        # ----------------- STT 보정 & GPT -----------------
        corrected_transcription = correct_stt_result(all_text.strip())
        analysis_result = get_chat_response(corrected_transcription, current_time="0:00", target_time=target_time)

        # ----------------- 불안 분석 -----------------
        anxiety_grade, final_score, anxiety_series, strong_events_ratio = anxiety_analysis(save_path, wav_path)

        jobs[job_id] = {
            "status": "completed",
            "result": {
                "transcription": all_text.strip(),
                "pronunciation_score": round(pron_score, 4),
                "intensity_grade": intensity_grade,
                "intensity_db": round(avg_db, 2),
                "intensity_text": intensity_comment,
                "pitch_grade": pitch_grade,
                "pitch_avg": round(avg_pitch, 2),
                "pitch_text": pitch_comment,
                "wpm_grade": wpm_grade,
                "wpm_avg": round(wpm, 2),
                "wpm_comment": wpm_comment,
                "corrected_transcription": corrected_transcription,
                "adjusted_script": analysis_result.get("adjusted_script"),
                "feedback": analysis_result.get("feedback"),
                "predicted_questions": analysis_result.get("predicted_questions"),
                "anxiety_analysis": anxiety_grade,
                "anxiety_ratio": round(strong_events_ratio, 4),
            },
        }

    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e)}

    finally:
        # 파일 정리
        for path in [save_path, wav_path]:
            if os.path.exists(path):
                os.remove(path)


@app.get("/result/{job_id}")
def get_result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return {"status": "not_found"}
    return job
