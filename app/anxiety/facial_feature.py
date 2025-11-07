import cv2
import mediapipe as mp
import numpy as np
import math
import os
from scipy.signal import find_peaks

# --- MediaPipe 설정 ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# --- 랜드마크 인덱스 ---
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
NOSE_TIP_INDEX = 1

# --- 헬퍼 함수 ---
def get_landmark_point_2d(landmarks, idx, frame_shape):
    if not landmarks or idx >= len(landmarks): return None
    landmark = landmarks[idx]
    x = landmark.x * frame_shape[1]
    y = landmark.y * frame_shape[0]
    return (int(x), int(y))

def euclidean_distance(point1, point2):
    if point1 is None or point2 is None: return 0.0
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_ear(face_landmarks, frame_shape):
    if not face_landmarks: return 0.0
    try:
        p1_right = get_landmark_point_2d(face_landmarks, RIGHT_EYE_INDICES[3], frame_shape)
        p4_right = get_landmark_point_2d(face_landmarks, RIGHT_EYE_INDICES[0], frame_shape)
        p2_right = get_landmark_point_2d(face_landmarks, RIGHT_EYE_INDICES[1], frame_shape)
        p6_right = get_landmark_point_2d(face_landmarks, RIGHT_EYE_INDICES[5], frame_shape)
        p3_right = get_landmark_point_2d(face_landmarks, RIGHT_EYE_INDICES[2], frame_shape)
        p5_right = get_landmark_point_2d(face_landmarks, RIGHT_EYE_INDICES[4], frame_shape)
        if None in [p1_right, p2_right, p3_right, p4_right, p5_right, p6_right]: right_ear = 0.0
        else:
            right_vertical_dist1 = euclidean_distance(p2_right, p6_right)
            right_vertical_dist2 = euclidean_distance(p3_right, p5_right)
            right_horizontal_dist = euclidean_distance(p1_right, p4_right)
            right_ear = (right_vertical_dist1 + right_vertical_dist2) / (2.0 * right_horizontal_dist) if right_horizontal_dist else 0.0

        p1_left = get_landmark_point_2d(face_landmarks, LEFT_EYE_INDICES[3], frame_shape)
        p4_left = get_landmark_point_2d(face_landmarks, LEFT_EYE_INDICES[0], frame_shape)
        p2_left = get_landmark_point_2d(face_landmarks, LEFT_EYE_INDICES[1], frame_shape)
        p6_left = get_landmark_point_2d(face_landmarks, LEFT_EYE_INDICES[5], frame_shape)
        p3_left = get_landmark_point_2d(face_landmarks, LEFT_EYE_INDICES[2], frame_shape)
        p5_left = get_landmark_point_2d(face_landmarks, LEFT_EYE_INDICES[4], frame_shape)
        if None in [p1_left, p2_left, p3_left, p4_left, p5_left, p6_left]: left_ear = 0.0
        else:
            left_vertical_dist1 = euclidean_distance(p2_left, p6_left)
            left_vertical_dist2 = euclidean_distance(p3_left, p5_left)
            left_horizontal_dist = euclidean_distance(p1_left, p4_left)
            left_ear = (left_vertical_dist1 + left_vertical_dist2) / (2.0 * left_horizontal_dist) if left_horizontal_dist else 0.0

        if right_ear > 0 and left_ear > 0: ear = (left_ear + right_ear) / 2.0
        elif right_ear > 0: ear = right_ear
        elif left_ear > 0: ear = left_ear
        else: ear = 0.0
    except IndexError: ear = 0.0
    return ear

# --- 시각 특징 추출 함수 ---
def extract_visual_features(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None, None, 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    ear_series = []
    head_movement_per_frame = []
    with mp_face_mesh.FaceMesh(...) as face_mesh: # 설정 생략
        last_good_ear = 0.3
        prev_nose_tip = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_shape = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = face_mesh.process(rgb_frame)
            rgb_frame.flags.writeable = True
            current_ear = last_good_ear
            current_nose_tip = None
            displacement = 0.0
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                current_ear = calculate_ear(landmarks, frame_shape)
                last_good_ear = current_ear if current_ear > 0 else last_good_ear
                current_nose_tip = get_landmark_point_2d(landmarks, NOSE_TIP_INDEX, frame_shape)
                if current_nose_tip and prev_nose_tip:
                    displacement = euclidean_distance(current_nose_tip, prev_nose_tip)
            ear_series.append(current_ear)
            head_movement_per_frame.append(displacement)
            prev_nose_tip = current_nose_tip
    cap.release()
    return np.array(ear_series), np.array(head_movement_per_frame), fps

# --- (★ 수정됨) 깜빡임 분석 함수 ---
def analyze_blinks_from_ear_series(ear_series, fps, window_size=1.0): # window_size 추가
    """EAR 시계열에서 깜빡임 프레임 인덱스 + 윈도우별 횟수 찾기"""
    PROMINENCE_THRESHOLD = 0.05
    min_width_frames = int(0.05 * fps)
    max_width_frames = int(0.5 * fps)
    inverted_ear_series = -ear_series
    inverted_ear_series[ear_series <= 0] = np.nan
    peaks, properties = find_peaks(
        inverted_ear_series, prominence=PROMINENCE_THRESHOLD, width=(min_width_frames, max_width_frames)
    )

    # --- 윈도우별 횟수 계산 추가 ---
    frames_per_window = int(fps * window_size)
    total_frames = len(ear_series)
    # 마지막 불완전한 윈도우도 포함하기 위해 올림 계산
    num_windows = math.ceil(total_frames / frames_per_window)
    blink_counts_per_window = np.zeros(num_windows, dtype=int)

    for peak_frame_index in peaks:
        window_index = peak_frame_index // frames_per_window
        # 배열 인덱스 초과 방지
        if window_index < num_windows:
            blink_counts_per_window[window_index] += 1
    # --- 계산 끝 ---

    # 반환값 3개로 변경
    return blink_counts_per_window, peaks, properties

# --- (★ 수정됨) 머리 움직임 스파이크 분석 함수 ---
def analyze_head_movement_spikes(head_movement_per_frame_series, fps, window_size=1.0): # window_size 추가
    """프레임별 Head Movement 스파이크 프레임 인덱스 + 윈도우별 횟수 찾기"""
    HEAD_MOVEMENT_PERCENTILE = 90
    MIN_SPIKE_DISTANCE_FRAMES = int(0.3 * fps)
    valid_movement = head_movement_per_frame_series[head_movement_per_frame_series > 0.1]
    if len(valid_movement) < 20:
        # 데이터 부족 시 빈 배열 반환
        total_frames = len(head_movement_per_frame_series)
        frames_per_window = int(fps * window_size)
        num_windows = math.ceil(total_frames / frames_per_window)
        return np.zeros(num_windows, dtype=int), np.array([]) # 빈 횟수 배열, 빈 인덱스 배열

    movement_threshold = np.percentile(valid_movement, HEAD_MOVEMENT_PERCENTILE)
    print(f"   > 머리 움직임 임계값 (상위 10%): {movement_threshold:.2f} pixels/frame")
    spike_indices, _ = find_peaks(
        head_movement_per_frame_series, height=movement_threshold, distance=MIN_SPIKE_DISTANCE_FRAMES
    )

    # --- 윈도우별 횟수 계산 추가 ---
    frames_per_window = int(fps * window_size)
    total_frames = len(head_movement_per_frame_series)
    num_windows = math.ceil(total_frames / frames_per_window)
    spike_counts_per_window = np.zeros(num_windows, dtype=int)

    for spike_frame_index in spike_indices:
        window_index = spike_frame_index // frames_per_window
        if window_index < num_windows:
            spike_counts_per_window[window_index] += 1
    # --- 계산 끝 ---

    # 반환값 2개로 변경
    return spike_counts_per_window, spike_indices

# --- 메인 실행 블록 (★ 수정됨 ★) ---
if __name__ == "__main__":
    VIDEO_FILE_PATH = "./sample_voices/FER_sample.mp4"
    WINDOW_SIZE = 1.0 # 집계 단위 (이제 분석 함수에 전달됨)

    if os.path.exists(VIDEO_FILE_PATH):
        print("1. 시각 특징(EAR, Head Movement) 추출 중...")
        ear_series, head_movement_per_frame, fps = extract_visual_features(VIDEO_FILE_PATH)

        if ear_series is not None and head_movement_per_frame is not None:
            print("2. 눈깜빡임 분석 중...")
            # (★ 수정 ★) 윈도우별 횟수 배열 받기
            blinks_per_window, blink_peaks, _ = analyze_blinks_from_ear_series(ear_series, fps, window_size=WINDOW_SIZE)
            print(f"   > 감지된 깜빡임 수: {len(blink_peaks)}")
            print("\n--- Blink Counts per Window ---")
            print(blinks_per_window) # 윈도우별 횟수 출력

            print("\n3. 머리 움직임 스파이크 분석 중...")
            # (★ 수정 ★) 윈도우별 횟수 배열 받기
            head_spikes_per_window, head_spike_indices = analyze_head_movement_spikes(head_movement_per_frame, fps, window_size=WINDOW_SIZE)
            print(f"   > 감지된 머리 움직임 스파이크 수: {len(head_spike_indices)}")
            print("\n--- Head Movement Spikes per Window ---")
            print(head_spikes_per_window) # 윈도우별 횟수 출력
    else:
        print(f"Error: '{VIDEO_FILE_PATH}' 파일을 찾을 수 없습니다.")