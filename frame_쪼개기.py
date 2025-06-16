import cv2
import mediapipe as mp
import json
import os

# 경로 설정
video_path = 'test_video.mp4'
output_image_dir = 'save_frames'  # ← 여기 수정
output_json_dir = 'output_json'

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_json_dir, exist_ok=True)

# MediaPipe 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 비디오 열기
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"[INFO] FPS: {fps:.2f}")

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # BGR → RGB 변환
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    # 이미지 저장
    frame_filename = f"frame_{frame_idx:04}.jpg"
    frame_path = os.path.join(output_image_dir, frame_filename)
    cv2.imwrite(frame_path, frame)

    # 좌표 저장
    keypoints = {}
    if result.pose_landmarks:
        for i, lm in enumerate(result.pose_landmarks.landmark):
            keypoints[f'{mp_pose.PoseLandmark(i).name}'] = {
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility
            }

    json_filename = f"frame_{frame_idx:04}.json"
    json_path = os.path.join(output_json_dir, json_filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(keypoints, f, indent=2, ensure_ascii=False)

    frame_idx += 1

cap.release()
pose.close()
print(f"[DONE] {frame_idx} 프레임 처리 완료")
