import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import json

# 모델 및 인덱스 로드
model = load_model('pose_condition_exercise_model.keras')
with open('exercise_index.json', 'r', encoding='utf-8') as f:
    exercise_to_index = json.load(f)
index_to_exercise = {v: k for k, v in exercise_to_index.items()}

# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# 관절 매핑 (MediaPipe ↔ 사용자 정의)
JOINT_NAMES = [
    'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
    'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
    'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle',
    'Neck', 'Left Palm', 'Right Palm', 'Back', 'Waist',
    'Left Foot', 'Right Foot'
]

JOINT_MAPPING = {
    'Nose': mp_pose.PoseLandmark.NOSE,
    'Left Eye': mp_pose.PoseLandmark.LEFT_EYE,
    'Right Eye': mp_pose.PoseLandmark.RIGHT_EYE,
    'Left Ear': mp_pose.PoseLandmark.LEFT_EAR,
    'Right Ear': mp_pose.PoseLandmark.RIGHT_EAR,
    'Left Shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
    'Right Shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
    'Left Elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
    'Right Elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
    'Left Wrist': mp_pose.PoseLandmark.LEFT_WRIST,
    'Right Wrist': mp_pose.PoseLandmark.RIGHT_WRIST,
    'Left Hip': mp_pose.PoseLandmark.LEFT_HIP,
    'Right Hip': mp_pose.PoseLandmark.RIGHT_HIP,
    'Left Knee': mp_pose.PoseLandmark.LEFT_KNEE,
    'Right Knee': mp_pose.PoseLandmark.RIGHT_KNEE,
    'Left Ankle': mp_pose.PoseLandmark.LEFT_ANKLE,
    'Right Ankle': mp_pose.PoseLandmark.RIGHT_ANKLE,
}


# 계산이 필요한 관절 처리 함수
def calculate_additional_joints(landmarks):
    joints = {}

    # Neck: 어깨 중간점
    left_shoulder = np.array([landmarks[JOINT_MAPPING['Left Shoulder']].x,
                              landmarks[JOINT_MAPPING['Left Shoulder']].y])
    right_shoulder = np.array([landmarks[JOINT_MAPPING['Right Shoulder']].x,
                               landmarks[JOINT_MAPPING['Right Shoulder']].y])
    joints['Neck'] = (left_shoulder + right_shoulder) / 2

    # Waist: 힙 중간점
    left_hip = np.array([landmarks[JOINT_MAPPING['Left Hip']].x,
                         landmarks[JOINT_MAPPING['Left Hip']].y])
    right_hip = np.array([landmarks[JOINT_MAPPING['Right Hip']].x,
                          landmarks[JOINT_MAPPING['Right Hip']].y])
    joints['Waist'] = (left_hip + right_hip) / 2

    # Palm: 손목과 손가락 끝의 중간점 (단순화)
    joints['Left Palm'] = np.array([landmarks[mp_pose.PoseLandmark.LEFT_INDEX].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_INDEX].y])
    joints['Right Palm'] = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].y])

    # Foot: 발목 좌표 재사용
    joints['Left Foot'] = np.array([landmarks[JOINT_MAPPING['Left Ankle']].x,
                                    landmarks[JOINT_MAPPING['Left Ankle']].y])
    joints['Right Foot'] = np.array([landmarks[JOINT_MAPPING['Right Ankle']].x,
                                     landmarks[JOINT_MAPPING['Right Ankle']].y])

    # Back: 어깨와 힙 중간점
    joints['Back'] = (joints['Neck'] + joints['Waist']) / 2

    return joints


# 비디오 처리
cap = cv2.VideoCapture('test_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 관절 추출
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        # 기본 관절 좌표 추출
        landmarks = results.pose_landmarks.landmark
        keypoints = []
        valid = True

        # 추가 관절 계산
        extra_joints = calculate_additional_joints(landmarks)

        # 전체 관절 처리
        for joint in JOINT_NAMES:
            if joint in JOINT_MAPPING:  # MediaPipe 기본 관절
                x = landmarks[JOINT_MAPPING[joint]].x
                y = landmarks[JOINT_MAPPING[joint]].y
                if x < 0 or y < 0:
                    valid = False
                    break
            else:  # 계산된 관절
                x, y = extra_joints[joint]

            keypoints.extend([x, y])

        if valid:
            # 모델 예측
            input_data = np.array([keypoints])
            pred_conditions, pred_exercise = model.predict(input_data, verbose=0)

            # 결과 해석
            exercise = index_to_exercise[np.argmax(pred_exercise[0])]
            conditions = (pred_conditions[0] > 0.5).astype(int)

            # 화면 표시
            cv2.putText(frame, f"Exercise: {exercise}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Conditions: {', '.join(map(str, conditions))}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 관절 시각화
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 화면 출력
    cv2.imshow('Pose Analysis', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()