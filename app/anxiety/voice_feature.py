import parselmouth
from parselmouth.praat import call # ★ 'call' 함수를 임포트해야 합니다 ★
import numpy as np
import os

# Praat 분석을 위한 상수 정의
PITCH_FLOOR = 75.0
PITCH_CEILING = 500.0

# Jitter/Shimmer 계산은 '주파수(Hz)'가 아닌 '주기(sec)'를 인수로 받습니다.
SHORTEST_PERIOD = 1.0 / PITCH_CEILING  # (1 / 500Hz)
LONGEST_PERIOD = 1.0 / PITCH_FLOOR     # (1 / 75Hz)

def extract_features_by_window(audio_path, window_size=1.0):
    """
    오디오 파일을 window_size(초) 단위로 분할하여
    각 구간의 평균 F0, Jitter(local), Shimmer(local)를 추출합니다.
    """
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return np.array([]), np.array([]), np.array([])
        
    try:
        snd = parselmouth.Sound(audio_path)
    except parselmouth.PraatError as e:
        print(f"Error loading audio file: {e}")
        return np.array([]), np.array([]), np.array([])

    duration = snd.get_total_duration()
    num_windows = int(duration // window_size)
    
    f0_series, jitter_series, shimmer_series = [], [], []

    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        segment = snd.extract_part(from_time=start, to_time=end)
        
        if segment.get_intensity() < 20:
            f0_series.append(0.0)
            jitter_series.append(0.0)
            shimmer_series.append(0.0)
            continue

        # --- F0 (평균 음높이) ---
        # (이 부분은 올바르게 작동합니다)
        pitch = segment.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
        f0_values = pitch.selected_array['frequency']
        f0_voiced_values = f0_values[f0_values != 0]
        
        if len(f0_voiced_values) > 0:
            f0_mean = np.mean(f0_voiced_values)
        else:
            f0_mean = 0.0
        
        # --- Jitter & Shimmer ---
        # (★오류 수정 지점★)
        # 'call' 함수를 사용하도록 코드를 복원하고, 인수를 수정합니다.
        
        # 1. PointProcess 생성:
        point_process = call(segment, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)
        
        # 2. Jitter 계산:
        jitter = call(point_process, "Get jitter (local)", 0, 0, SHORTEST_PERIOD, LONGEST_PERIOD, 1.3)
        
        # 3. Shimmer 계산:
        shimmer = call([segment, point_process], "Get shimmer (local)", 0, 0, SHORTEST_PERIOD, LONGEST_PERIOD, 1.3, 1.6)

        # NaN 값 처리 (음성 구간이 너무 짧으면 NaN이 뜰 수 있음)
        if isinstance(jitter, float) and np.isnan(jitter):
            jitter = 0.0
        if isinstance(shimmer, float) and np.isnan(shimmer):
            shimmer = 0.0

        f0_series.append(f0_mean)
        jitter_series.append(jitter)
        shimmer_series.append(shimmer)

    return np.array(f0_series), np.array(jitter_series), np.array(shimmer_series)

if __name__ == "__main__":
    # --- 예시 실행 ---
    voice_file_path = "./sample_voices/sample.wav" 

    if os.path.exists(voice_file_path):
        f0_series, jitter_series, shimmer_series = extract_features_by_window(voice_file_path, window_size=1.0)

        print("--- F0 (음높이) Series ---")
        print(f0_series)
        print("\n--- Jitter (음높이 떨림) Series ---")
        print(jitter_series)
        print("\n--- Shimmer (소리크기 떨림) Series ---")
        print(shimmer_series)
    else:
        print(f"'{voice_file_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")