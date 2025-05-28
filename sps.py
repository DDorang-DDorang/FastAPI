import parselmouth
import numpy as np
import matplotlib.pyplot as plt

# 음성 파일 불러오기
snd = parselmouth.Sound("./sample_voices/sample.wav")

# 1. 무음 제거를 위한 intensity 분석
intensity = snd.to_intensity()
times = intensity.xs()
values = intensity.values[0]

# 음성 활동 구간만 추출 (무음 threshold: 50 dB)
speech_mask = values > 50
speech_durations = np.diff(times)[speech_mask[:-1]]
total_speech_time = np.sum(speech_durations)

print(f"말한 시간 (무음 제외): {total_speech_time:.2f}초")

# 2. Pitch 분석 → 음절 수 추정
pitch = snd.to_pitch()
pitch_values = pitch.selected_array['frequency']
pitch_values[pitch_values == 0] = np.nan  # 무음 부분 제외

# 피치에서 음절 수 추정: 음절 ≈ 피크 개수
# 간단하게 NaN 아닌 부분의 변화량을 카운트
syllable_like = np.count_nonzero(~np.isnan(pitch_values))

# 3. 말 빠르기 계산 (syllables per second)
sps = syllable_like / total_speech_time
print(f"추정 음절 수: {syllable_like}")
print(f"말 빠르기 (Syllables per Second): {sps:.2f}")
