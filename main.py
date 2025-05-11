import whisper
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uuid
import os
import torchaudio

app = FastAPI()

model = whisper.load_model("medium")

def compute_pronunciation_score(logprobs, threshold=-1.0):
    filtered = [lp for lp in logprobs if lp > threshold]
    score = np.mean([np.exp(lp) for lp in filtered]) if filtered else 0.0
    return score, len(filtered), threshold

@app.post("/stt")
async def transcribe(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    save_path = f"temp_{file_id}.wav"

    with open(save_path, "wb") as f:
        f.write(await file.read())

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
            "final_score": round(score, 4),
            "used_segment_count": used,
            "logprob_threshold": round(threshold, 4)
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if os.path.exists(save_path):
            os.remove(save_path)
