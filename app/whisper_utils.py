import whisper
import numpy as np


model = whisper.load_model("medium")

def transcribe_audio(wav_path: str, language: str = "ko") -> dict:
    return model.transcribe(wav_path, language=language, word_timestamps=False)

def calculate_pronunciation_score(segments, threshold=-1.0):
    logprobs = [seg["avg_logprob"] for seg in segments if "avg_logprob" in seg]
    filtered = [lp for lp in logprobs if lp > threshold]
    score = np.mean([np.exp(lp) for lp in filtered]) if filtered else 0.0
    grade, comment = grade_pronounciation_score(score)
    return score, grade, comment

def grade_pronounciation_score(score):
    pronounciation_comment = ""
    if score >= 0.88:
        pronounciation_comment = "명확하고 자연스러운 발음"
        return "A", pronounciation_comment  
    elif score >= 0.83:
        pronounciation_comment = "대체로 명확한 발음"
        return "B", pronounciation_comment  
    elif score >= 0.78:
        pronounciation_comment = "전달은 되지만 다소 부정확한 발음"
        return "C", pronounciation_comment  
    else :
        pronounciation_comment = "발음이 불분명하고 전달 어려움"
        return "D", pronounciation_comment  

def calculate_wpm(segments):
    """
    Whisper segments 기반 WPM 계산

    - segments: Whisper가 반환한 segment 리스트
    - 각 segment에서 단어 수와 발화 길이를 구해서 전체 WPM 계산
    """
    total_words = 0
    total_time = 0.0  # 초 단위

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        word_count = len(text.split())
        segment_duration = seg.get("end", 0) - seg.get("start", 0)
        
        total_words += word_count
        total_time += segment_duration

    if total_time == 0:
        return 0.0

    wpm = (total_words / total_time) * 60  # 초 -> 분 환산
    wpm_grade, wpm_comment = grade_wpm_korean(wpm)
    return wpm, wpm_grade, wpm_comment


def grade_wpm_korean(wpm):
    wpm_comment = ""
    if 75 <= wpm <= 115:
        wpm_comment = "적절함"
        return "A", wpm_comment  
    elif 70 <= wpm <= 120:
        if wpm < 75:
            wpm_comment = "조금 느림"
        else:
            wpm_comment = "조금 빠름"
        return "B", wpm_comment
    elif 65 <= wpm <= 125:
        if wpm < 70:
            wpm_comment = "느림"
        else:
            wpm_comment = "빠름"
        return "C", wpm_comment
    else :
        if wpm < 65:
            wpm_comment = "많이 느림"
        else:
            wpm_comment = "많이 빠름"
        return "D", wpm_comment