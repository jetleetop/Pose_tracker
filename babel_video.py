import os
import cv2
import json
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# ───────────────────────── 설정 ─────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, 'aimodel', 'pose_condition_exercise_babel_model.keras')
IDX_PATH        = os.path.join(BASE_DIR, 'condition', 'exercise_index_babel.json')
COND_MAP_PATH   = os.path.join(BASE_DIR, 'condition', 'exercise_condition_babel_map.json')
VIDEO_IN        = os.path.join(BASE_DIR, 'video', 'test_squat.mp4')
VIDEO_OUT       = os.path.join(BASE_DIR, 'video', 'output_squat123.mp4')

# ───────────────── 모델 & 매핑 로드 ─────────────────
model = load_model(MODEL_PATH)
with open(IDX_PATH, 'r', encoding='utf-8') as f:
    exercise_to_index = json.load(f)
index_to_exercise = {v: k for k, v in exercise_to_index.items()}
with open(COND_MAP_PATH, 'r', encoding='utf-8') as f:
    _ = json.load(f)  # unused

# English condition names mapping
english_cond_map = {
    'deadlift': [
        'spine neutral',
        'foot-knee alignment',
        'bar-body closeness',
        'simultaneous knee-hip extension',
        'feet flat'
    ],
    'squat': [
        'spine neutral',
        'head forward',
        'foot-knee alignment',
        'feet flat'
    ]
}

# ───────────────── MediaPipe 초기화 ─────────────────
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose    = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# condition to landmarks mapping
delta_map = {
    'spine neutral': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
                      mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP],
    'foot-knee alignment': [mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE,
                             mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE],
    'bar-body closeness': [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST],
    'simultaneous knee-hip extension': [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE,
                                         mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE],
    'feet flat': [mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.RIGHT_HEEL]
}

# ──────────────── 카운터 & 다운 플래그 ────────────────
down_flag       = {'squat': False, 'deadlift': False}
active_exercise = None
counts          = {'squat': 0, 'deadlift': 0}

# ──────────────── 각도 계산 함수 ────────────────
def calc_angle(a, b, c):
    va = np.array([a.x, a.y]); vb = np.array([b.x, b.y]); vc = np.array([c.x, c.y])
    ba, bc = va - vb, vc - vb
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

# ──────────────── 랜드마크 추출 ────────────────
def extract(frame):
    res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return res, (res.pose_landmarks.landmark if res.pose_landmarks else None)

# ──────────────── 비디오 처리 ────────────────
cap = cv2.VideoCapture(VIDEO_IN)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps    = cap.get(cv2.CAP_PROP_FPS)
w, h   = int(cap.get(3)), int(cap.get(4))
out    = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results, lm = extract(frame)
    if lm is None:
        out.write(frame)
        cv2.imshow('Feedback', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # 1) 모델 예측
    inp = np.array([[p.x, p.y] for p in lm[:24]]).flatten().reshape(1, -1)
    conds, exe_pred = model.predict(inp, verbose=0)
    exe_key = index_to_exercise[int(np.argmax(exe_pred[0]))].lower()
    cond_scores = conds[0]

    # 2) phase & count (down flag)
    phase = None
    if exe_key == 'squat':
        angs = [
            calc_angle(lm[mp_pose.PoseLandmark.RIGHT_HIP], lm[mp_pose.PoseLandmark.RIGHT_KNEE], lm[mp_pose.PoseLandmark.RIGHT_ANKLE]),
            calc_angle(lm[mp_pose.PoseLandmark.LEFT_HIP],  lm[mp_pose.PoseLandmark.LEFT_KNEE],  lm[mp_pose.PoseLandmark.LEFT_ANKLE])
        ]
        avg_ang = sum(angs) / 2
        phase = 2 if avg_ang < 100 else (0 if avg_ang > 160 else 1)
        key = 'squat'
    elif exe_key == 'deadlift' or active_exercise == 'deadlift':
        angs = [
            calc_angle(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER], lm[mp_pose.PoseLandmark.RIGHT_HIP], lm[mp_pose.PoseLandmark.RIGHT_KNEE]),
            calc_angle(lm[mp_pose.PoseLandmark.LEFT_SHOULDER],  lm[mp_pose.PoseLandmark.LEFT_HIP],  lm[mp_pose.PoseLandmark.LEFT_KNEE])
        ]
        avg_ang = sum(angs) / 2
        phase = 2 if avg_ang < 140 else (0 if avg_ang > 170 else 1)
        key = 'deadlift'
        exe_key = 'deadlift'
    else:
        key = None

    if key and phase is not None:
        if phase == 2 and not down_flag[key]:
            down_flag[key] = True
            active_exercise = key
        if phase == 0 and down_flag.get(active_exercise, False):
            counts[active_exercise] += 1
            down_flag[active_exercise] = False
            active_exercise = None

    # 3) English conditions & flags
    cond_names = english_cond_map.get(exe_key, [])
    ok_flags = [cond_scores[i] > 0.5 for i in range(len(cond_names))]

    # 4) Draw skeleton
    mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 5) Highlight faulty areas
    for name, ok in zip(cond_names, ok_flags):
        color = (0,255,0) if ok else (0,0,255)
        for lm_idx in delta_map.get(name, []):
            coord = results.pose_landmarks.landmark[lm_idx]
            x, y = int(coord.x * w), int(coord.y * h)
            cv2.circle(frame, (x, y), 8, color, -1)

    # 6) Overlay counts
    cv2.putText(frame, f"Squat Count: {counts['squat']}",    (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, f"Deadlift Count: {counts['deadlift']}",(20, 80),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    out.write(frame)
    cv2.imshow('Feedback', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
