import os
import numpy as np
from pydub import AudioSegment
import cv2 # 시각화 함수 사용 시 필요

try:
    from voice_feature import extract_features_by_window
    from facial_feature import (
        extract_visual_features,
        analyze_blinks_from_ear_series,
        analyze_head_movement_spikes
        # , visualize_events_on_video # 시각화 필요 시 주석 해제
    )
except ImportError:
    print("오류: 'test.py' 또는 'blinking_test.py' 파일을 찾을 수 없습니다.")
    exit()


def extract_audio_pydub(video_path, audio_path):
    try:
        print(f"--- 1. '{video_path}'에서 오디오 추출 시작 (Pydub) ---")
        audio = AudioSegment.from_file(video_path, format="mp4")
        audio = audio.set_channels(1)
        audio.export(audio_path, format="wav")
        print(f"--- 1a. 오디오 추출 완료 ---")
        return True
    except Exception as e:
        print(f"Pydub 오디오 추출 중 오류 발생: {e}")
        return False

def calculate_anxiety_scores(blinks, f0, jitter, shimmer, head_movement):
    print("\n--- 3. 불안 점수 계산 시작 ---")

    # --- 유효 데이터 필터링 ---
    valid_blinks = blinks[blinks > 0]
    valid_f0 = f0[f0 > 0]
    valid_jitter = jitter[jitter > 0]
    valid_shimmer = shimmer[shimmer > 0]
    valid_head = head_movement[head_movement > 0]

    # --- 임계값 계산 (상위 15%) ---
    blink_thresh = np.percentile(valid_blinks, 85) if len(valid_blinks) > 1 else 1 # 데이터 1개면 최소 1회
    f0_thresh = np.percentile(valid_f0, 85) if len(valid_f0) > 0 else 9999
    jitter_thresh = np.percentile(valid_jitter, 85) if len(valid_jitter) > 0 else 99
    shimmer_thresh = np.percentile(valid_shimmer, 85) if len(valid_shimmer) > 0 else 99
    head_thresh = np.percentile(valid_head, 85) if len(valid_head) > 1 else 1 # 데이터 1개면 최소 1회

    print(f"Blink 임계값 (상위 15% 시작): {blink_thresh:.2f} 회/초")
    print(f"F0 임계값 (상위 15%): {f0_thresh:.2f} Hz")
    print(f"Jitter 임계값 (상위 15%): {jitter_thresh:.4f}")
    print(f"Shimmer 임계값 (상위 15%): {shimmer_thresh:.4f}")
    print(f"Head Mov 임계값 (상위 15% 시작): {head_thresh:.2f} 회/초")

    # --- 스파이크 점수 계산 ---
    num_windows = len(blinks)
    blink_scores = np.zeros(num_windows)
    head_scores = np.zeros(num_windows)

    # 음성 지표는 기존 방식
    f0_spikes = (f0 >= f0_thresh).astype(float) * 2.0
    jitter_spikes = (jitter >= jitter_thresh).astype(float) * 1.5
    shimmer_spikes = (shimmer >= shimmer_thresh).astype(float) * 1.5

    for i in range(num_windows):
        # Blink 점수 (최소 thresh 이상일 때, 횟수에 비례, 최대 2점)
        if blinks[i] >= blink_thresh and blink_thresh > 0:
            score = max(1.0, min(2.0, blinks[i] / blink_thresh))
            blink_scores[i] = score * 1.0 # Blink 가중치 = 1.0

        # Head Movement 점수 (최소 thresh 이상일 때, 횟수에 비례, 최대 2점)
        if head_movement[i] >= head_thresh and head_thresh > 0:
            score = max(1.0, min(2.0, head_movement[i] / head_thresh))
            head_scores[i] = score * 1.0 # Head Movement 가중치 = 1.0

    # --- 침묵 구간 처리 ---
    silent_mask = (f0 == 0)
    f0_spikes[silent_mask] = 0
    jitter_spikes[silent_mask] = 0
    shimmer_spikes[silent_mask] = 0
    # (Blink, Head는 침묵 중에도 점수 유지)

    # F0(2) + Jitter(1.5) + Shimmer(1.5) + Blink(최대 2) + Head(최대 2) = 8.0
    anxiety_score_series = (
        blink_scores +
        f0_spikes +
        jitter_spikes +
        shimmer_spikes +
        head_scores
    )

    # --- 최종 Metric 계산 ---
    speaking_mask = (f0 > 0)
    speaking_anxiety_scores = anxiety_score_series[speaking_mask]

    if len(speaking_anxiety_scores) == 0:
        print("경고: 음성 구간이 감지되지 않았습니다.")
        return anxiety_score_series, "N/A", 0.0, 0 # strong_events 0 반환

    MAX_SCORE = 8.0
    average_score = np.mean(speaking_anxiety_scores)
    final_score_100 = (average_score / MAX_SCORE) * 100

    grade = get_anxiety_grade(final_score_100) # 등급 함수 호출

    strong_threshold = 5.0
    strong_events_count = np.sum(speaking_anxiety_scores >= strong_threshold)

    return anxiety_score_series, grade, final_score_100, strong_events_count

# --- 3. 불안 등급 변환 함수 ---
def get_anxiety_grade(score_100):
    """(수정됨) 100점 만점 불안 점수를 A-E 등급으로 변환 (총점 8.0 기준)"""
    score_100 = max(0, min(100, score_100))
    # Threshold 예시 (총점이 높아졌으므로 기준 약간 상향 또는 유지)
    if score_100 >= 40: # 예: 45점 이상 (평균 3.6점 이상)
        return "E (매우 불안)"
    elif score_100 >= 30: # 예: 30점 이상 (평균 2.4점 이상)
        return "D (불안)"
    elif score_100 >= 20: # 예: 18점 이상 (평균 1.44점 이상)
        return "C (약간 불안)"
    elif score_100 >= 10:  # 예: 8점 이상 (평균 0.64점 이상)
        return "B (안정)"
    else:
        return "A (매우 안정)"
    
def anxiety_analysis(video_file_path, audio_path, window_size=1.0):
    fps_from_video = 30 # 기본값

    try:
        # --- [단계 2] 오디오 & 비디오 특징 추출 (★수정★) ---
        print("\n--- 2. 오디오 및 비디오 특징 추출 중 ---")

        f0_series, jitter_series, shimmer_series = extract_features_by_window(
            audio_path, window_size=WINDOW_SIZE
        )

        # (★수정★) extract_visual_features 호출
        ear_series, head_movement_per_frame, fps_from_video = extract_visual_features(
            video_file_path
        )
        if ear_series is None: # 추출 실패 시 종료
             raise Exception("시각 특징(EAR, Head Movement) 추출에 실패했습니다.")

        # EAR 시리즈 -> Blinks 분석 (윈도우별 횟수)
        blink_series, _, _ = analyze_blinks_from_ear_series(
            ear_series, fps_from_video, window_size=window_size
        )
        # (★신규★) Head Movement -> Head Spikes 분석 (윈도우별 횟수)
        head_spikes_series, _ = analyze_head_movement_spikes(
            head_movement_per_frame, fps_from_video, window_size=window_size
        )

        if len(f0_series)==0 or len(blink_series)==0 or len(head_spikes_series)==0:
            raise Exception("특징 추출 실패 또는 길이가 0입니다 (오디오, 블링크, 헤드 중 하나).")

        # --- [단계 3] 데이터 동기화 (★수정★) ---
        min_len = min(len(f0_series), len(blink_series), len(head_spikes_series))

        f0_aligned = f0_series[:min_len]
        jitter_aligned = jitter_series[:min_len]
        shimmer_aligned = shimmer_series[:min_len]
        blink_aligned = blink_series[:min_len]
        head_aligned = head_spikes_series[:min_len] # Head Movement 데이터 추가


        # --- [단계 4] 불안 점수 계산 (★수정★) ---
        anxiety_series, anxiety_grade, final_score, strong_events = calculate_anxiety_scores(
            blink_aligned, f0_aligned, jitter_aligned, shimmer_aligned, head_aligned # head 전달
        )


        return anxiety_grade, final_score, anxiety_series

        # # --- 시간대별 불안 정보 출력 (기존 로직 유지) ---
        # time = []
        # severe_time = []
        # for i in range(len(anxiety_series)) :
        #     # (★수정★) 강한 불안 기준 점수 변경 (strong_threshold 값과 일치)
        #     if anxiety_series[i] >= 2.0 : # 예: 2점 이상 구간 기록
        #         # 초를 "분:초" 형식으로 변환 (00:00)
        #         minutes = i // 60
        #         seconds = i % 60
        #         time_str = f"{minutes:02d}:{seconds:02d}"
        #         time.append(time_str)
        #         if anxiety_series[i] >= 4.5 : # 강한 불안 기준
        #             severe_time.append(time_str)

        # print("\n불안 감지 구간 (2.0점 이상):")
        # print(time if time else "없음")
        # print("\n강한 불안 감지 구간 (4.5점 이상):")
        # print(severe_time if severe_time else "없음")

    except Exception as e:
        print(f"\n프로세스 중 심각한 오류 발생: {e}")

    finally:
        # --- [단계 7] 임시 오디오 파일 삭제 ---
        if os.path.exists(TEMP_AUDIO_PATH):
            os.remove(TEMP_AUDIO_PATH)
            print(f"\n임시 오디오 파일 '{TEMP_AUDIO_PATH}' 삭제 완료.")
    
    
# --- 4. 메인 실행 로직 (★수정됨★) ---
if __name__ == "__main__":

    # --- 설정 ---
    VIDEO_FILE_PATH = "./sample_voices/FER_sample.mp4" # 파일 경로 확인!
    WINDOW_SIZE = 1.0
    TEMP_AUDIO_PATH = "temp_audio_for_analysis.wav"

    fps_from_video = 30 # 기본값

    try:
        # --- [단계 1] 오디오 추출 ---
        extraction_success = extract_audio_pydub(VIDEO_FILE_PATH, TEMP_AUDIO_PATH)
        if not extraction_success:
            raise Exception("Pydub 오디오 추출에 실패했습니다.")

        # --- [단계 2] 오디오 & 비디오 특징 추출 (★수정★) ---
        print("\n--- 2. 오디오 및 비디오 특징 추출 중 ---")

        f0_series, jitter_series, shimmer_series = extract_features_by_window(
            TEMP_AUDIO_PATH, window_size=WINDOW_SIZE
        )

        # (★수정★) extract_visual_features 호출
        ear_series, head_movement_per_frame, fps_from_video = extract_visual_features(
            VIDEO_FILE_PATH
        )
        if ear_series is None: # 추출 실패 시 종료
             raise Exception("시각 특징(EAR, Head Movement) 추출에 실패했습니다.")

        # EAR 시리즈 -> Blinks 분석 (윈도우별 횟수)
        blink_series, _, _ = analyze_blinks_from_ear_series(
            ear_series, fps_from_video, window_size=WINDOW_SIZE
        )
        # (★신규★) Head Movement -> Head Spikes 분석 (윈도우별 횟수)
        head_spikes_series, _ = analyze_head_movement_spikes(
            head_movement_per_frame, fps_from_video, window_size=WINDOW_SIZE
        )

        if len(f0_series)==0 or len(blink_series)==0 or len(head_spikes_series)==0:
            raise Exception("특징 추출 실패 또는 길이가 0입니다 (오디오, 블링크, 헤드 중 하나).")

        # --- [단계 3] 데이터 동기화 (★수정★) ---
        min_len = min(len(f0_series), len(blink_series), len(head_spikes_series))

        f0_aligned = f0_series[:min_len]
        jitter_aligned = jitter_series[:min_len]
        shimmer_aligned = shimmer_series[:min_len]
        blink_aligned = blink_series[:min_len]
        head_aligned = head_spikes_series[:min_len] # Head Movement 데이터 추가

        print("--- f0 ----")
        print(f0_aligned)
        print()
        print("--- jitter ---")
        print(jitter_aligned)
        print()
        print("--- shimmer ---")
        print(shimmer_aligned)
        print()
        print("--- blink ---")
        print(blink_aligned)
        print()
        print("--- head movement ---")
        print(head_aligned)
        print()


        print(f"데이터 동기화 완료: 총 {min_len}초 분량 분석")

        # --- [단계 4] 불안 점수 계산 (★수정★) ---
        anxiety_series, anxiety_grade, final_score, strong_events = calculate_anxiety_scores(
            blink_aligned, f0_aligned, jitter_aligned, shimmer_aligned, head_aligned # head 전달
        )

        # --- [단계 5] 최종 결과 출력 (★수정됨★) ---
        print("\n--- 🏁 최종 분석 결과 ---")
        print(f"전체 불안 지수 (100점 만점): {final_score:.2f} 점")
        print(f"종합 불안 등급: {anxiety_grade}")
        # (★수정★) 강한 불안 이벤트 횟수 출력 (strong_events 변수 사용)
        print(f"강한 불안 이벤트 횟수 (4.5점+): {strong_events} 회")

        print()

        print("\n--- 시간대별 불안 점수 (0~8점) ---") # 최대 점수 변경
        print(anxiety_series)

        # --- 시간대별 불안 정보 출력 (기존 로직 유지) ---
        time = []
        severe_time = []
        for i in range(len(anxiety_series)) :
            # (★수정★) 강한 불안 기준 점수 변경 (strong_threshold 값과 일치)
            if anxiety_series[i] >= 2.0 : # 예: 2점 이상 구간 기록
                # 초를 "분:초" 형식으로 변환 (00:00)
                minutes = i // 60
                seconds = i % 60
                time_str = f"{minutes:02d}:{seconds:02d}"
                time.append(time_str)
                if anxiety_series[i] >= 4.5 : # 강한 불안 기준
                    severe_time.append(time_str)

        print("\n강한 불안 감지 구간 (4.5점 이상):")
        print(severe_time if severe_time else "없음")

        # --- [단계 6] 시각/청각 검증 (현재 코드에서는 제거됨) ---
        # print("\n--- 5. 불안 구간 시각적/청각적 검증 시작 ---")
        # visualize_anxiety_events(...) # 필요 시 blinking_test.py에서 가져와 사용

    except Exception as e:
        print(f"\n프로세스 중 심각한 오류 발생: {e}")

    finally:
        # --- [단계 7] 임시 오디오 파일 삭제 ---
        if os.path.exists(TEMP_AUDIO_PATH):
            os.remove(TEMP_AUDIO_PATH)
            print(f"\n임시 오디오 파일 '{TEMP_AUDIO_PATH}' 삭제 완료.")