import json
import os
import uuid
from typing import Dict

from fastapi import FastAPI, File, Form, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from anxiety.anxiety_score import anxiety_analysis
from config import UPLOAD_DIR, DEFAULT_TARGET_TIME
from gpt import correct_stt_result, get_chat_response, get_compare_result
from utils.file_handler import save_upload_file, merge_chunks, convert_to_wav
from voice_analysis import SoundAnalyzer
from whisper_utils import transcribe_audio, calculate_pronunciation_score, calculate_wpm

app = FastAPI()
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: Dict[str, dict] = {}


# ----------------- API -----------------

@app.post("/compare")
async def compare_scripts(script1: str = Form(...), script2: str = Form(...)):
    try:
        result = get_compare_result(script1, script2)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/analysis")
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


    ext = os.path.splitext(video.filename)[1].lower()

    # 청크 병합
    if chunk_index is not None and total_chunks is not None and original_filename:
        # chunked upload
        chunk_filename = f"{original_filename}_chunk_{chunk_index}{ext}"
        save_upload_file(video, chunk_filename, UPLOAD_DIR)
        uploaded_chunks = [
            name for name in os.listdir(UPLOAD_DIR) if name.startswith(original_filename + "_chunk_")
        ]
        if len(uploaded_chunks) < total_chunks:
            return {"status": "chunk_received", "chunk_index": chunk_index}

        save_path = merge_chunks(original_filename, total_chunks, ext, UPLOAD_DIR)
    else:
        # 일반 업로드
        save_path = save_upload_file(video, f"temp_{job_id}{ext}", UPLOAD_DIR)

    # wav 변환
    wav_path = convert_to_wav(save_path)

    # 백그라운드 분석 작업 추가
    background_tasks.add_task(process_audio_job, job_id, save_path, wav_path, metadata)
    return {"job_id": job_id, "status": "processing", "save_path": save_path}


# ----------------- 백그라운드 작업 -----------------

def process_audio_job(job_id: str, save_path: str, wav_path: str, metadata: str):
    try:
        target_time = DEFAULT_TARGET_TIME
        if metadata:
            meta_data = json.loads(metadata)
            target_time = meta_data.get("target_time", DEFAULT_TARGET_TIME)

        # ----------------- Whisper -----------------
        result = transcribe_audio(wav_path, language="ko")
        segments = result.get("segments", [])
        all_text = result.get("text", "")

        pron_score, pron_grade, pron_comment = calculate_pronunciation_score(segments)
        wpm, wpm_grade, wpm_comment = calculate_wpm(segments)

        # ----------------- SoundAnalyzer -----------------
        analyzer = SoundAnalyzer(wav_path, threshold=60)
        intensity_grade, avg_db, intensity_comment = analyzer.evaluate_intensity()
        pitch_grade, avg_pitch, pitch_comment = analyzer.evaluate_pitch_score()

        # ----------------- STT 보정 & GPT -----------------
        corrected_transcription = correct_stt_result(all_text.strip())
        analysis_result = get_chat_response(corrected_transcription, current_time="0:00", target_time=target_time)

        # ----------------- 불안 분석 -----------------
        anxiety_grade, anxiety_comment, final_score, anxiety_series, strong_events_ratio = anxiety_analysis(save_path, wav_path)

        jobs[job_id] = {
            "status": "completed",
            "result": {
                "transcription": all_text.strip(),
                "corrected_transcription": corrected_transcription,
                "pronounciation_grade" : pron_grade, #추가
                "pronounciation_score": round(pron_score, 4),
                "pronounciation_text": pron_comment, #추가
                "intensity_grade": intensity_grade,
                "intensity_db": round(avg_db, 2),
                "intensity_text": intensity_comment,
                "pitch_grade": pitch_grade,
                "pitch_avg": round(avg_pitch, 2),
                "pitch_text": pitch_comment,
                "wpm_grade": wpm_grade,
                "wpm_avg": round(wpm, 2),
                "wpm_comment": wpm_comment,
                "anxiety_grade": anxiety_grade,
                "anxiety_ratio": round(strong_events_ratio, 6),
                "anxiety_comment": anxiety_comment,
                "adjusted_script": analysis_result.get("adjusted_script"),
                "feedback": analysis_result.get("feedback"),
                "predicted_questions": analysis_result.get("predicted_questions"),
            },
        }

    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e)}

    finally:
        # 파일 정리
        for path in [wav_path]:
            if os.path.exists(path):
                os.remove(path)


@app.get("/result/{job_id}")
def get_result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return {"status": "not_found"}
    return job
