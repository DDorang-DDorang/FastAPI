import whisper
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uuid
import os
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import torchaudio

app = FastAPI()

# large로 수정 필요
model = whisper.load_model("medium")

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
        video = VideoFileClip(save_path)
        audio = video.audio
        audio.write_audiofile(wav_path, codec='pcm_s16le')
        video.close()
    elif ext == ".wav":
        wav_path = save_path
    else:
        raise ValueError("지원되지 않는 파일 형식입니다. wav 또는 mp4만 허용됩니다.")


    try:
        # Whisper 음성 처리
        result = model.transcribe(save_path, language="ko", word_timestamps=False)

        segments = result.get("segments", [])
        all_text = result.get("text", "")
        all_logprobs = []

        for seg in segments:
            if "avg_logprob" in seg:
                all_logprobs.append(seg["avg_logprob"])

        score, used, threshold = compute_pronunciation_score(all_logprobs)

        return JSONResponse(content={
            "transcription": all_text.strip(),
            "pronunciation_score": round(score, 4),
            # "used_segment_count": used,
            # "logprob_threshold": round(threshold, 4)
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if os.path.exists(save_path):
            os.remove(save_path)
