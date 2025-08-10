import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
cap = cv2.VideoCapture(0)


previous_center = None

def get_eye_center(landmarks, eye_indices):
    eye_points = np.array([landmarks[i] for i in eye_indices])
    return np.mean(eye_points, axis=0)

LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지 전처리
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

            # 눈동자 중심 좌표
            left_eye = np.array(landmarks[LEFT_IRIS_CENTER])
            right_eye = np.array(landmarks[RIGHT_IRIS_CENTER])

            eye_center = ((left_eye + right_eye) / 2).astype(int)

            # 시선 튐 감지
            if previous_center is not None:
                dx = eye_center[0] - previous_center[0]
                dy = eye_center[1] - previous_center[1]
                movement = np.linalg.norm([dx, dy])

                if movement > 5:
                    print(f"이동량: {movement:.2f}")

            previous_center = eye_center
            cv2.circle(frame, left_eye, 3, (255, 0, 0), -1)
            cv2.circle(frame, right_eye, 3, (0, 0, 255), -1)

    cv2.imshow("Gaze Tracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()