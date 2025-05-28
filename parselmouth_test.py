import parselmouth
import numpy as np
import matplotlib.pyplot as plt

# 음성 파일 불러오기
snd = parselmouth.Sound("./sample_voices/sample.wav")  # 경로는 본인 파일로 바꾸세요

# Intensity 객체 얻기
intensity = snd.to_intensity()

# 시간별 intensity 정보 얻기
times = intensity.xs()
values = intensity.values[0]  # 2차원 array이므로 첫 번째 값 사용

# 평균, 최대, 최소 데시벨
avg_db = np.mean(values)
max_db = np.max(values)
min_db = np.min(values)

print(f"평균 dB: {avg_db:.2f}")
print(f"최대 dB: {max_db:.2f}")
print(f"최소 dB: {min_db:.2f}")

# 시각화
plt.figure(figsize=(10, 4))
plt.plot(times, values, label="Intensity (dB)")
plt.xlabel("Time (s)")
plt.ylabel("dB")
plt.title("음성 크기(Intensity) 분석")
plt.grid(True)
plt.show()

pitch = snd.to_pitch()

# 시간과 주파수 정보 가져오기
pitch_values = pitch.selected_array['frequency']  # pitch in Hz
pitch_values[pitch_values==0] = np.nan  # 무음 구간은 NaN 처리
times = pitch.xs()

# 평균, 최대, 최소 pitch (무음 제외)
mean_pitch = np.nanmean(pitch_values)
max_pitch = np.nanmax(pitch_values)
min_pitch = np.nanmin(pitch_values)

print(f"평균 Pitch: {mean_pitch:.2f} Hz")
print(f"최대 Pitch: {max_pitch:.2f} Hz")
print(f"최소 Pitch: {min_pitch:.2f} Hz")

# 시각화
plt.figure(figsize=(10, 4))
plt.plot(times, pitch_values, label="Pitch (Hz)", color="orange")
plt.xlabel("Time (s)")
plt.ylabel("Pitch (Hz)")
plt.title("음성 높낮이(Pitch) 분석")
plt.grid(True)
plt.show()