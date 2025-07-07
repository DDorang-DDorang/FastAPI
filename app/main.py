import whisper
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uuid
import os
from pydub import AudioSegment
from voice_analysis import SoundAnalyzer  # Assuming you have a voice_analysis module
from whisper_utils import transcribe_audio, calculate_pronunciation_score, calculate_wpm  # Assuming you have a whisper_utils module
from gpt import correct_stt_result, get_chat_response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

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

#

@app.post("/stt")
async def transcribe(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    save_path = f"temp_{file_id}{ext}"



    with open(save_path, "wb") as f:
        f.write(await file.read())

    wav_path = f"temp_{file_id}.wav"

    if ext == ".mp4":
        audio = AudioSegment.from_file(save_path, format="mp4")
        audio.export(wav_path, format="wav")
    elif ext == ".wav":
        wav_path = save_path
    else:
        raise ValueError("지원되지 않는 파일 형식입니다. wav 또는 mp4만 허용됩니다.")

    try:
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

        analysis_result = get_chat_response(corrected_transcription)

        return JSONResponse(content={
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

            # 분석 결과 추가
            "adjusted_script": analysis_result["adjusted_script"],
            "feedback": analysis_result["feedback"],
            "predicted_questions": analysis_result["predicted_questions"]
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if os.path.exists(save_path):
            os.remove(save_path)
        if os.path.exists(wav_path) and wav_path != save_path:
            os.remove(wav_path) 