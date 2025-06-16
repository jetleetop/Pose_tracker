import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque

# 1. 모델 및 MediaPipe 초기화
model = load_model('exercise_phase_count_model.keras')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# 2. 비디오 설정
cap = cv2.VideoCapture('test_video.mp4')
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 3. 분석 변수 초기화
sequence = deque(maxlen=30)  # 30프레임 시퀀스
count = 0
state_history = []
phase_names = ['idle', 'crunching', 'top']

# 4. 상태 머신 설정
STATE = {
    "prev_phase": 0,
    "cool_down": 0,
    "current_count": 0
}
current_phase = 0
count_prob = 0.0
# 5. 메인 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 5-1. 관절 추정
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        # 5-2. 키포인트 추출 (RIGHT_ELBOW, LEFT_KNEE 등 4개 관절)
        keypoints = []
        for joint in [mp_pose.PoseLandmark.RIGHT_ELBOW,
                      mp_pose.PoseLandmark.RIGHT_KNEE,
                      mp_pose.PoseLandmark.LEFT_ELBOW,
                      mp_pose.PoseLandmark.LEFT_KNEE]:
            lm = results.pose_landmarks.landmark[joint]
            keypoints.extend([lm.x, lm.y])

        sequence.append(keypoints)

        # 5-3. 예측 수행 (30프레임 채워질 때마다)
        if len(sequence) == 30:
            input_data = np.array(sequence).reshape(1, 30, 8)
            phase_pred, count_pred = model.predict(input_data, verbose=0)

            current_phase = np.argmax(phase_pred[0])
            count_prob = count_pred[0][0]

            # 5-4. 상태 전이에 따른 카운트 처리
            if (STATE["cool_down"] <= 0 and
                    current_phase == 2 and  # top 상태
                    STATE["prev_phase"] == 1):  # crunching → top 전환

                STATE["current_count"] += 1
                STATE["cool_down"] = int(fps * 0.5)  # 0.5초 쿨다운

            STATE["prev_phase"] = current_phase
            STATE["cool_down"] = max(0, STATE["cool_down"] - 1)
            state_history.append(current_phase)

    # 5-5. 시각화
    # 관절 연결선 그리기
    if results.pose_landmarks:
        for connection in mp_pose.POSE_CONNECTIONS:
            start = connection[0]
            end = connection[1]
            if start in [mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.LEFT_ELBOW,
                         mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.LEFT_KNEE]:
                start_point = (int(results.pose_landmarks.landmark[start].x * WIDTH),
                               int(results.pose_landmarks.landmark[start].y * HEIGHT))
                end_point = (int(results.pose_landmarks.landmark[end].x * WIDTH),
                             int(results.pose_landmarks.landmark[end].y * HEIGHT))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

    # 상태 및 카운트 표시
    cv2.putText(frame, f"Phase: {phase_names[current_phase]}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Count: {STATE['current_count']}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    cv2.putText(frame, f"Confidence: {count_prob:.2f}", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

    # 5-6. 출력
    cv2.imshow('Exercise Analysis', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6. 종료 처리
cap.release()
cv2.destroyAllWindows()

# 7. 분석 결과 리포트
print(f"\n📊 최종 분석 결과")
print(f"- 총 운동 횟수: {STATE['current_count']}")
print(f"- 상태 분포: idle({state_history.count(0)}), crunching({state_history.count(1)}), top({state_history.count(2)})")