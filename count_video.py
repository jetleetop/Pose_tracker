import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# exercise_model 로드
exercise_model = load_model("exercise_model.h5")

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# 관절 좌표 추출 함수 (팔꿈치와 무릎을 기준으로)
def extract_joint_coordinates(frame):
    # MediaPipe를 사용하여 관절 좌표 추출
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # 4개의 주요 관절 좌표 (팔꿈치, 무릎)만 추출
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        joint_coords = [
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y,
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y
        ]
        return joint_coords
    return None


# 프레임을 모델 입력 형식에 맞게 시퀀스로 전처리하는 함수
def preprocess_frame(frame):
    joint_coords = extract_joint_coordinates(frame)
    if joint_coords:
        return np.array(joint_coords)
    return None


# 비디오 분석 함수
def analyze_video_and_count(model, video_path):
    cap = cv2.VideoCapture(video_path)

    # 초기 상태 및 카운트
    current_state = "idle"
    completed_count = 0
    joint_sequence = []  # 시퀀스 저장용 리스트

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임을 모델 입력 형식에 맞게 전처리
        joint_coords = preprocess_frame(frame)
        if joint_coords is not None:
            joint_sequence.append(joint_coords)

        # 30프레임마다 모델 입력에 적합한 시퀀스를 생성
        if len(joint_sequence) == 30:
            X_frame = np.expand_dims(np.array(joint_sequence), axis=0)  # 배치 차원 추가
            phase_preds, count_preds = model.predict(X_frame)

            # phase_preds에서 운동 상태 출력
            predicted_phase = np.argmax(phase_preds)

            # 상태 전환 추적
            if predicted_phase == 0:  # "idle" 상태
                current_state = "idle"
            elif predicted_phase == 1:  # "crunching" 상태
                if current_state == "idle":
                    current_state = "crunching"
            elif predicted_phase == 2:  # "top" 상태
                if current_state == "crunching":
                    current_state = "top"
                    completed_count += 1  # 운동 완료 -> 카운트 증가
                    current_state = "idle"  # 운동 완료 후 "idle" 상태로 돌아가기

            # 시퀀스 초기화
            joint_sequence = joint_sequence[1:]  # 첫 번째 프레임을 삭제하고 새로운 프레임 추가

        # 비디오 프레임에 카운트 출력
        cv2.putText(frame, f"Completed Count: {completed_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 비디오 프레임 표시
        cv2.imshow('Exercise Video', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# exercise_model을 사용하여 비디오 분석 실행
analyze_video_and_count(exercise_model, "test_video2.mp4")