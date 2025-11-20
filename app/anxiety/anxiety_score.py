import numpy as np
from .voice_feature import extract_features_by_window
from .facial_feature import (
    extract_visual_features,
    analyze_blinks_from_ear_series,
    analyze_head_movement_spikes
)

# 상수 정의
PERCENTILE_THRESHOLD = 85
F0_SPIKE_WEIGHT = 2.0
JITTER_SPIKE_WEIGHT = 1.5
SHIMMER_SPIKE_WEIGHT = 1.5
BLINK_SCORE_WEIGHT = 1.0
HEAD_SCORE_WEIGHT = 1.0
MAX_SCORE_WITH_VIDEO = 8.0
MAX_SCORE_AUDIO_ONLY = 6.0
STRONG_THRESHOLD_WITH_VIDEO = 4.5
STRONG_THRESHOLD_AUDIO_ONLY = 3.0
DEFAULT_FPS = 30


def calculate_anxiety_scores(
    blinks: np.ndarray,
    f0: np.ndarray,
    jitter: np.ndarray,
    shimmer: np.ndarray,
    head_movement: np.ndarray,
    is_audio_only: bool = False
):

    # --- 유효 데이터 필터링 ---
    valid_blinks = blinks[blinks > 0]
    valid_f0 = f0[f0 > 0]
    valid_jitter = jitter[jitter > 0]
    valid_shimmer = shimmer[shimmer > 0]
    valid_head = head_movement[head_movement > 0]

    # --- 임계값 계산 (상위 15%) ---
    blink_thresh = np.percentile(valid_blinks, PERCENTILE_THRESHOLD) if len(valid_blinks) > 1 else 1
    f0_thresh = np.percentile(valid_f0, PERCENTILE_THRESHOLD) if len(valid_f0) > 0 else 9999
    jitter_thresh = np.percentile(valid_jitter, PERCENTILE_THRESHOLD) if len(valid_jitter) > 0 else 99
    shimmer_thresh = np.percentile(valid_shimmer, PERCENTILE_THRESHOLD) if len(valid_shimmer) > 0 else 99
    head_thresh = np.percentile(valid_head, PERCENTILE_THRESHOLD) if len(valid_head) > 1 else 1

    num_windows = len(blinks)
    blink_scores = np.zeros(num_windows)
    head_scores = np.zeros(num_windows)

    # 음성 지표 스파이크 계산
    f0_spikes = (f0 >= f0_thresh).astype(float) * F0_SPIKE_WEIGHT
    jitter_spikes = (jitter >= jitter_thresh).astype(float) * JITTER_SPIKE_WEIGHT
    shimmer_spikes = (shimmer >= shimmer_thresh).astype(float) * SHIMMER_SPIKE_WEIGHT

    for i in range(num_windows):
        if blinks[i] >= blink_thresh and blink_thresh > 0:
            score = max(1.0, min(2.0, blinks[i] / blink_thresh))
            blink_scores[i] = score * BLINK_SCORE_WEIGHT

        if head_movement[i] >= head_thresh and head_thresh > 0:
            score = max(1.0, min(2.0, head_movement[i] / head_thresh))
            head_scores[i] = score * HEAD_SCORE_WEIGHT

    # --- 침묵 구간 처리 ---
    silent_mask = (f0 == 0)
    f0_spikes[silent_mask] = 0
    jitter_spikes[silent_mask] = 0
    shimmer_spikes[silent_mask] = 0

    # --- 불안 점수 합산 ---
    anxiety_score_series = (
        blink_scores +
        f0_spikes +
        jitter_spikes +
        shimmer_spikes +
        head_scores
    )

    speaking_mask = (f0 > 0)
    speaking_anxiety_scores = anxiety_score_series[speaking_mask]

    if len(speaking_anxiety_scores) == 0:
        return "N/A", "음성 구간 감지 안 됨", 0, anxiety_score_series, 0.0

    max_score = MAX_SCORE_AUDIO_ONLY if is_audio_only else MAX_SCORE_WITH_VIDEO
    average_score = np.mean(speaking_anxiety_scores)
    final_score_100 = (average_score / max_score) * 100

    strong_threshold = STRONG_THRESHOLD_AUDIO_ONLY if is_audio_only else STRONG_THRESHOLD_WITH_VIDEO

    strong_events_count = np.sum(speaking_anxiety_scores >= strong_threshold)
    strong_events_ratio = strong_events_count / len(speaking_anxiety_scores)

    grade, comment = get_anxiety_grade(strong_events_ratio)

    return grade, comment, final_score_100, anxiety_score_series, strong_events_ratio


def get_anxiety_grade(density_ratio: float):
    density_ratio = max(0, min(1.0, density_ratio))
    
    if density_ratio >= 0.06:   
        return "E", "매우 불안"
    elif density_ratio >= 0.05:
        return "D", "불안"
    elif density_ratio >= 0.04: 
        return "C", "약간 불안"
    elif density_ratio >= 0.03:
        return "B", "안정"
    else:                       
        return "A", "매우 안정"
    
def anxiety_analysis(video_file_path: str, audio_path: str, window_size: float = 1.0):
    """불안도 분석 메인 함수"""
    is_audio_only = video_file_path.lower().endswith(".wav")

    try:
        # --- 1. 음성 특징 추출 ---
        f0_series, jitter_series, shimmer_series = extract_features_by_window(
            audio_path, window_size=window_size
        )

        # --- 2. 시각 특징 추출 ---
        if not is_audio_only:
            ear_series, head_movement_per_frame, fps = extract_visual_features(video_file_path)
            blink_series, _, _ = analyze_blinks_from_ear_series(ear_series, fps, window_size=window_size)
            head_spikes_series, _ = analyze_head_movement_spikes(head_movement_per_frame, fps, window_size=window_size)
        else:
            blink_series = np.zeros(len(f0_series))
            head_spikes_series = np.zeros(len(f0_series))

        min_len = min(len(f0_series), len(blink_series), len(head_spikes_series))

        # --- 3. 불안 점수 측정 ---
        result = calculate_anxiety_scores(
            blink_series[:min_len],
            f0_series[:min_len],
            jitter_series[:min_len],
            shimmer_series[:min_len],
            head_spikes_series[:min_len],
            is_audio_only=is_audio_only
        )

        return result

    except Exception as e:
        print(f"불안도 분석 중 오류 발생: {e}")
        return "N/A", "분석 실패", 0, np.array([]), 0.0


if __name__ == "__main__":

    import os
    from pydub import AudioSegment
    
    VIDEO_FILE_PATH = "./sample_voices/FER_sample.mp4"
    WINDOW_SIZE = 1.0
    TEMP_AUDIO_PATH = "temp_audio_for_analysis.wav"

    try:
        # 오디오 추출
        audio = AudioSegment.from_file(VIDEO_FILE_PATH, format="mp4")
        audio = audio.set_channels(1)
        audio.export(TEMP_AUDIO_PATH, format="wav")
        
        anxiety_grade, anxiety_comment, final_score, anxiety_series, strong_events_ratio = anxiety_analysis(
            VIDEO_FILE_PATH, TEMP_AUDIO_PATH, window_size=WINDOW_SIZE
        )

        print("\n--- 최종 분석 결과 ---")
        print(f"불안 등급: {anxiety_grade} ({anxiety_comment})")
        print(f"불안 지수 (100점 만점): {final_score:.2f}점")
        print(f"강한 불안 비율: {strong_events_ratio:.6f}")



    except Exception as e:
        print(f"\n프로세스 중 심각한 오류 발생: {e}")

    finally:
        # --- [단계 7] 임시 오디오 파일 삭제 ---
        if os.path.exists(TEMP_AUDIO_PATH):
            os.remove(TEMP_AUDIO_PATH)
            print(f"\n임시 오디오 파일 '{TEMP_AUDIO_PATH}' 삭제 완료.")