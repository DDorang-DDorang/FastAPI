import parselmouth
import numpy as np
import matplotlib.pyplot as plt



class SoundAnalyzer:
    def __init__(self, snd, threshold=60):
        self.snd = snd
        self.threshold = threshold
        self.intensity = self.snd.to_intensity()
        self.pitch = self.snd.to_pitch()
    
    def evaluate_intensity(self):
        values = self.intensity.values[0]

        q1 = np.percentile(values, 10)
        q3 = np.percentile(values, 90)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        filtered_values = values[(values >= lower_bound) & (values <= upper_bound)]
        avg_db = np.mean(filtered_values)

        ratio = avg_db / self.threshold

        if 0.97 <= ratio <= 1.03:
            grade = "A"
        elif 0.92 <= ratio <= 1.08:
            grade = "B"
        elif 0.85 <= ratio <= 1.15:
            grade = "C"
        elif 0.75 <= ratio <= 1.25:
            grade = "D"
        else:
            grade = "E"

        return grade, avg_db

    def evaluate_syllables_per_second(self):
        times = self.intensity.xs()
        values = self.intensity.values[0]

        speech_mask = values > 40
        speech_durations = np.diff(times)[speech_mask[:-1]]
        total_speech_time = np.sum(speech_durations)

        pitch_values = self.pitch.selected_array['frequency']
        pitch_values[pitch_values == 0] = np.nan

        syllable_like = np.count_nonzero(~np.isnan(pitch_values))
        sps = syllable_like / total_speech_time if total_speech_time > 0 else 0

        return syllable_like, sps
    

    def evaluate_pitch_score(self):
        pitch_values = self.pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]

        q1 = np.percentile(pitch_values, 25)
        q3 = np.percentile(pitch_values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_pitch = pitch_values[(pitch_values >= lower_bound) & (pitch_values <= upper_bound)]

        if len(filtered_pitch) == 0:
            return "E", 0  # 이상치 제거 후 데이터 없음

        pitch_std = np.std(filtered_pitch)
        pitch_range = np.max(filtered_pitch) - np.min(filtered_pitch)

        # 기준 예시 (단위: Hz, 발표 환경에 따라 조절 가능)
        if pitch_std > 50 and pitch_range > 100:
            grade = "A"  # 다양한 톤, 생동감 있음
        elif pitch_std > 30 and pitch_range > 70:
            grade = "B"
        elif pitch_std > 15 and pitch_range > 40:
            grade = "C"
        elif pitch_std > 5 and pitch_range > 20:
            grade = "D"
        else:
            grade = "E"  # 너무 monotone

        return grade, pitch_std