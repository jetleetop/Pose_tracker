import cv2
import os

# === 경로 설정 ===
video_path = 'output.avi'         # 입력 영상 경로
output_image_dir = 'save_frames'  # 프레임 이미지 저장 폴더

os.makedirs(output_image_dir, exist_ok=True)

# === 비디오 열기 ===
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"[INFO] FPS: {fps:.2f}")

frame_idx = 0

# === 프레임별 저장 루프 ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_filename = f"frame_{frame_idx:04}.jpg"
    frame_path = os.path.join(output_image_dir, frame_filename)
    cv2.imwrite(frame_path, frame)

    frame_idx += 1

cap.release()
print(f"[DONE] 총 {frame_idx} 프레임 저장 완료! (폴더: {output_image_dir})")
