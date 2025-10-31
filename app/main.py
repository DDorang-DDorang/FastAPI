import json
import whisper
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
from pydub import AudioSegment
from pydub.utils import mediainfo
from anxiety.anxiety_score import anxiety_analysis
from voice_analysis import SoundAnalyzer  # Assuming you have a voice_analysis module
from whisper_utils import transcribe_audio, calculate_pronunciation_score, calculate_wpm  # Assuming you have a whisper_utils module
from gpt import correct_stt_result, get_chat_response, get_compare_result
# from FER import analyze_emotion
from typing import Dict


app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 React 도메인
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def compute_pronunciation_score(logprobs, threshold=-1.0):
    filtered = [lp for lp in logprobs if lp > threshold]
    score = np.mean([np.exp(lp) for lp in filtered]) if filtered else 0.0
    return score, len(filtered), threshold



jobs: Dict[str, dict] = {}


@app.post("/compare")
async def compare_scripts(script1: str = Form(...), script2: str = Form(...)):
    try:
        result = get_compare_result(script1, script2)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# @app.post("/stt")
# async def transcribe(background_tasks: BackgroundTasks, video: UploadFile = File(...), metadata: str = Form(...)):
#     job_id = str(uuid.uuid4())
#     jobs[job_id] = {"status": "processing"}

#     ext = os.path.splitext(video.filename)[1].lower()
#     save_path = f"temp_{job_id}{os.path.splitext(video.filename)[1]}"

#     if ext == ".mp4":
#         with open(save_path, "wb") as f:
#             f.write(await video.read())
#         audio = AudioSegment.from_file(save_path, format="mp4")
#         audio.export(f"temp_{job_id}.wav", format="wav")
#     elif ext == ".wav":
#         pass
#     else:
#         raise ValueError("지원되지 않는 파일 형식입니다. wav 또는 mp4만 허용됩니다.")

#     background_tasks.add_task(process_audio_job, job_id, save_path, metadata)
#     return {"job_id": job_id}

@app.post("/stt")
async def transcribe_chunked(
    background_tasks: BackgroundTasks,
    video: UploadFile,
    metadata: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    original_filename: str = Form(...) 
):
    job_id = str(uuid.uuid4())

    ext = os.path.splitext(video.filename)[1].lower()

    chunk_filename = f"{original_filename}_chunk_{chunk_index}{ext}"
    chunk_path = os.path.join(UPLOAD_DIR, chunk_filename)

    with open(chunk_path, "wb") as f:
        f.write(await video.read())

    uploaded_chunks = [
        name for name in os.listdir(UPLOAD_DIR)
        if name.startswith(original_filename + "_chunk_")
    ]
    if len(uploaded_chunks) < total_chunks:
        return {"status": "chunk_received", "chunk_index": chunk_index}

    merged_path = os.path.join(UPLOAD_DIR, f"merged_{original_filename}{ext}")
    with open(merged_path, "wb") as merged:
        for i in range(total_chunks):
            part_path = os.path.join(UPLOAD_DIR, f"{original_filename}_chunk_{i}{ext}")
            with open(part_path, "rb") as part:
                merged.write(part.read())
            os.remove(part_path)  # 임시 조각 삭제

    ext = os.path.splitext(merged_path)[1].lower()
    save_path = f"temp_{job_id}{ext}"

    os.rename(merged_path, save_path)

    if ext == ".mp4":
        audio = AudioSegment.from_file(save_path, format="mp4")
        audio.export(f"temp_{job_id}.wav", format="wav")
    elif ext == ".wav":
        pass  # 이미 wav면 그대로 사용
    else:
        raise ValueError("지원되지 않는 파일 형식입니다. wav 또는 mp4만 허용됩니다. 현재 형식 : " + ext)

    jobs[job_id] = {"status": "processing"}
    background_tasks.add_task(process_audio_job, job_id, save_path, metadata)

    return {"job_id": job_id, "status": "merged_and_processing"}

def process_audio_job(job_id: str, save_path: str, metadata: str):
    try :
        if metadata is None:
            target_time = "6:00"
        else :
            meta_data = json.loads(metadata)

            target_time = meta_data.get("target_time")

        wav_path = f"temp_{job_id}.wav"

    
        audio_info = mediainfo(wav_path)
        duration = float(audio_info["duration"])   # 초 단위 길이
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        current_time = f"{minutes}:{seconds:02d}"

        # Whisper 음성 처리
        result = transcribe_audio(wav_path, language="ko")
        segments = result.get("segments", [])
        all_text = result.get("text", "")

        pronounciation_score, used, threshold = calculate_pronunciation_score(segments)
        wpm, wpm_grade, wpm_comment = calculate_wpm(segments)

        # SoundAnalyzer 음성 분석
        analyzer = SoundAnalyzer(wav_path, threshold=60)
        intensity_grade, avg_db, intensity_comment = analyzer.evaluate_intensity()
        pitch_grade, avg_pitch, pitch_comment = analyzer.evaluate_pitch_score()

        transcription = all_text.strip()

        corrected_transcription = correct_stt_result(transcription)

        analysis_result = get_chat_response(corrected_transcription, current_time=current_time, target_time=target_time)
        
        # 표정 분석 (삭제)
        # emotion_analysis = analyze_emotion(save_path, frame_skip=120)

        # 불안 분석
        anxiety_grade, final_score, anxiety_series, strong_events_ratio = anxiety_analysis(
            save_path, wav_path
        )

        jobs[job_id] = {
            "status" : "completed",
            "result" : {
                "transcription": all_text.strip(),
                "pronunciation_score": round(pronounciation_score, 4),
                "intensity_grade": intensity_grade,
                "intensity_db": round(avg_db, 2),
                "intensity_text": intensity_comment,
                "pitch_grade": pitch_grade,
                "pitch_avg": round(avg_pitch, 2),
                "pitch_text": pitch_comment,
                "wpm_grade": wpm_grade,
                "wpm_avg": round(wpm, 2),
                "wpm_comment": wpm_comment,
                "anxiety_analysis": anxiety_grade,
                "anxiety_ratio": round(strong_events_ratio, 4),

                # 분석 결과 추가
                "corrected_transcription": corrected_transcription,
                "adjusted_script": analysis_result["adjusted_script"],
                "feedback": analysis_result["feedback"], 
                "predicted_questions": analysis_result["predicted_questions"],
                

            }
        }
    
    except Exception as e :
        jobs[job_id] = {"status": "error", "error" : str(e)}


    finally:
        if os.path.exists(save_path):
            os.remove(save_path)
        if os.path.exists(wav_path) and wav_path != save_path:
            os.remove(wav_path) 

@app.get("/result/{job_id}")
def get_result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return {"status": "not_found"}
    return job