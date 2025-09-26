import cv2
from fer import FER

def analyze_emotion(video_path, frame_skip=120):
    """
    영상에서 긍정/중립/부정 비율을 계산하는 함수

    Args:
        video_path (str): 분석할 영상 파일 경로
        frame_skip (int): 분석할 프레임 간격 (기본 120)

    Returns:
        dict: {'positive': %, 'neutral': %, 'negative': %}
    """
    cap = cv2.VideoCapture(video_path)
    detector = FER(mtcnn=True)

    frame_count = 0
    positive_count = 0
    neutral_count = 0
    negative_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % frame_skip == 0:
            detected = detector.detect_emotions(frame)
            if detected:
                emotions = detected[0]["emotions"]
                dominant_emotion = max(emotions, key=emotions.get)

                # 감정 분류 매핑
                if dominant_emotion in ['happy', 'surprise']:
                    positive_count += 1
                elif dominant_emotion in ['neutral', 'fear']:
                    neutral_count += 1
                else:
                    negative_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # 비율 계산
    total = positive_count + neutral_count + negative_count
    if total > 0:
        pos_ratio = (positive_count / total) * 100
        neutral_ratio = (neutral_count / total) * 100
        neg_ratio = (negative_count / total) * 100
    else:
        pos_ratio = neutral_ratio = neg_ratio = 0

    return {
        "positive": pos_ratio,
        "neutral": neutral_ratio,
        "negative": neg_ratio
    }


# 사용 예시
if __name__ == "__main__":
    result = analyze_expression(r"sample_voices\FER_sample_myself_smile.mp4", frame_skip=120)
    print("\n영상 전체 비율:")
    print(f"긍정: {result['positive']:.2f}%, 중립: {result['neutral']:.2f}%, 부정: {result['negative']:.2f}%")
