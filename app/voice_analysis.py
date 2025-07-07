import parselmouth
import numpy as np

class SoundAnalyzer:
    def __init__(self, snd_path, threshold=60):
        self.snd = parselmouth.Sound(snd_path)
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

        # IQR를 사용하여 이상치 제거
        filtered_values = values[(values >= lower_bound) & (values <= upper_bound)]
        avg_db = np.mean(filtered_values)

        ratio = avg_db / self.threshold

        intensity_grade = "D" 
        intensity_comment = "N/A"

        if 0.95 <= ratio:
            intensity_grade = "A"
            intensity_comment = "적절한 목소리 크기"
        elif 0.90 <= ratio:
            intensity_grade = "B"
            intensity_comment = "조금 작은 목소리"
        elif 0.85 <= ratio:
            intensity_grade = "C"
            intensity_comment = "작은 목소리"
        else:
            intensity_grade = "D"
            intensity_comment = "너무 작은 목소리"

        return intensity_grade, avg_db, intensity_comment


    def evaluate_pitch_score(self):
        # TODO : 좀 더 정교한 피치 분석 및 기준 필요
        pitch_values = self.pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]

        q1 = np.percentile(pitch_values, 25)
        q3 = np.percentile(pitch_values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_pitch = pitch_values[(pitch_values >= lower_bound) & (pitch_values <= upper_bound)]

        pitch_grade = "D"
        pitch_comment = "N/A"

        if len(filtered_pitch) == 0:   
            return pitch_grade, 0, "N/A"

        pitch_std = np.std(filtered_pitch)
        pitch_range = np.max(filtered_pitch) - np.min(filtered_pitch)

        # 기준 예시 (단위: Hz, 발표 환경에 따라 조절 가능)
        if pitch_std > 50 and pitch_range > 100:
            pitch_grade = "A"
            pitch_comment = "표현력 우수"
        elif pitch_std > 30 and pitch_range > 70:
            pitch_grade = "B"
            pitch_comment = "표현력 적절"
        elif pitch_std > 15 and pitch_range > 40:
            pitch_grade = "C"
            pitch_comment = "조금 단조로움"
        else:
            pitch_grade = "D"
            pitch_comment = "단조로움"
        
        return pitch_grade, np.mean(filtered_pitch) if len(filtered_pitch) > 0 else 0, pitch_comment