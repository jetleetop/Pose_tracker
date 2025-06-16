import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import json

# 모델 및 인덱스 로드
pose_condition_exercise_model = load_model('pose_condition_exercise_model.keras')
exercise_model = load_model("exercise_model.h5")

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


# 비디오 분석 함수
def analyze_video_and_count(model, video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    # 비디오 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 비디오 형식 설정 (XVID 포맷)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 원본 영상의 FPS
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))  # 출력 파일 설정

    # 초기 상태 및 카운트
    current_state = "idle"
    prev_state = "idle"  # 이전 상태 추적용
    completed_count = 0
    joint_sequence = []  # 시퀀스 저장용 리스트

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 관절 추출 및 계산된 추가 관절 처리
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
                pred_conditions, pred_exercise = pose_condition_exercise_model.predict(input_data, verbose=0)

                # 결과 해석
                exercise = index_to_exercise[np.argmax(pred_exercise[0])]
                conditions = (pred_conditions[0] > 0.5).astype(int)

                # 화면 표시
                cv2.putText(frame, f"Exercise: {exercise}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Conditions: {', '.join(map(str, conditions))}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 30프레임마다 모델 입력에 적합한 시퀀스를 생성
            joint_coords = extract_joint_coordinates(frame)
            if joint_coords:
                joint_sequence.append(joint_coords)

            if len(joint_sequence) == 30:
                X_frame = np.expand_dims(np.array(joint_sequence), axis=0)
                phase_preds, count_preds = exercise_model.predict(X_frame)
                predicted_phase = np.argmax(phase_preds)

                # 상태 전환 로직 (수정된 부분)
                if predicted_phase == 0:  # "idle"
                    if current_state == "crunching" and prev_state == "top":
                        completed_count += 1  # top → crunching → idle 완료 시 카운트
                    current_state = "idle"
                elif predicted_phase == 1:  # "crunching"
                    if current_state == "idle" or current_state == "top":
                        prev_state = current_state  # 이전 상태 저장
                        current_state = "crunching"
                elif predicted_phase == 2:  # "top"
                    if current_state == "crunching":
                        current_state = "top"

                joint_sequence = joint_sequence[1:]  # 시퀀스 갱신

            # 비디오 프레임에 카운트 출력
            frame_height, frame_width, _ = frame.shape  # 프레임 크기
            text = f"Completed Count: {completed_count}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = frame_width - text_size[0] - 10  # 오른쪽 여백 10px
            text_y = 50  # 상단에 50px 간격

            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 관절 시각화
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 처리된 비디오 프레임 저장
        out.write(frame)

        # 비디오 프레임 표시
        cv2.imshow('Pose Analysis', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()  # 비디오 저장 종료
    cv2.destroyAllWindows()

# exercise_model을 사용하여 비디오 분석 실행
analyze_video_and_count(exercise_model, "test_33.mp4", "output_test_video4.mp4")