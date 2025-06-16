# ┌───────────────────────────────────────────────────────────────┐
# │                  REAL-TIME FEEDBACK (v1.1)                  │
# └───────────────────────────────────────────────────────────────┘

import os
import cv2
import json
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# 1) 설정
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, 'aimodel', 'pose_condition_exercise_babel_model.keras')
IDX_PATH        = os.path.join(BASE_DIR, 'condition', 'exercise_index_babel.json')

# ────────────────────────────────────────────────────────────────
# 2) 임계치
CONDITION_THRESHOLD = 0.4
ANGLE_THRESH = {
    'squat':    {'down': 100, 'up': 160},
    'deadlift': {'down': 140, 'up': 170}
}

# 3) 모델 및 매핑 로드
model = load_model(MODEL_PATH)
with open(IDX_PATH, 'r', encoding='utf-8') as f:
    exercise_to_index = json.load(f)
index_to_exercise = {v: k for k, v in exercise_to_index.items()}

# 4) Condition ↔ Landmark 매핑
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

# ────────────────────────────────────────────────────────────────
# TEMPORARY: 'head forward' 조건 제외 (머리 정면 검출 오류 대응)
if 'head forward' in english_cond_map['squat']:
    english_cond_map['squat'].remove('head forward')
# ────────────────────────────────────────────────────────────────

mp_pose  = mp.solutions.pose
delta_map = {
    'spine neutral':                   [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
                                        mp_pose.PoseLandmark.LEFT_HIP,      mp_pose.PoseLandmark.RIGHT_HIP],
    'foot-knee alignment':             [mp_pose.PoseLandmark.LEFT_KNEE,     mp_pose.PoseLandmark.LEFT_ANKLE,
                                        mp_pose.PoseLandmark.RIGHT_KNEE,    mp_pose.PoseLandmark.RIGHT_ANKLE],
    'feet flat':                       [mp_pose.PoseLandmark.LEFT_HEEL,     mp_pose.PoseLandmark.RIGHT_HEEL],
    'bar-body closeness':              [mp_pose.PoseLandmark.LEFT_WRIST,    mp_pose.PoseLandmark.RIGHT_WRIST],
    'simultaneous knee-hip extension': [mp_pose.PoseLandmark.LEFT_HIP,      mp_pose.PoseLandmark.LEFT_KNEE,
                                        mp_pose.PoseLandmark.RIGHT_HIP,     mp_pose.PoseLandmark.RIGHT_KNEE]
}

# 5) MediaPipe 초기화
mp_draw = mp.solutions.drawing_utils
pose    = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# 6) 카운터 초기화
down_flag       = {'squat': False, 'deadlift': False}
active_exercise = None
total_reps      = {'squat': 0, 'deadlift': 0}
correct_reps    = {'squat': 0, 'deadlift': 0}
incorrect_reps  = {'squat': 0, 'deadlift': 0}

# 7) 각도 계산 함수
def calc_angle(a, b, c):
    va, vb, vc = np.array([a.x, a.y]), np.array([b.x, b.y]), np.array([c.x, c.y])
    ba, bc    = va - vb, vc - vb
    cosang    = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

# 8) 웹캠 열기 & 전체화면
for idx in range(4):
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        print("Camera index:", idx)
        break
cv2.namedWindow('Feedback', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Feedback', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 9) 실시간 루프
while True:
    ret, frame = cap.read()
    if not ret: break
    h, w, _ = frame.shape

    # 9.1) 랜드마크
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    if not result.pose_landmarks:
        cv2.imshow('Feedback', frame)
        if cv2.waitKey(1)==ord('q'): break
        continue
    lm = result.pose_landmarks.landmark

    # 9.2) 모델 예측
    inp       = np.array([[p.x, p.y] for p in lm[:24]]).flatten().reshape(1, -1)
    conds, pr = model.predict(inp, verbose=0)
    exe_key   = index_to_exercise[int(np.argmax(pr[0]))].lower()
    scores    = conds[0]

    # 9.3) Phase & 카운트
    phase, key = None, None
    if exe_key == 'squat':
        key = 'squat'
        angs = [
            calc_angle(lm[mp_pose.PoseLandmark.RIGHT_HIP],
                       lm[mp_pose.PoseLandmark.RIGHT_KNEE],
                       lm[mp_pose.PoseLandmark.RIGHT_ANKLE]),
            calc_angle(lm[mp_pose.PoseLandmark.LEFT_HIP],
                       lm[mp_pose.PoseLandmark.LEFT_KNEE],
                       lm[mp_pose.PoseLandmark.LEFT_ANKLE])
        ]
        avg   = sum(angs)/2
        phase = 2 if avg < ANGLE_THRESH['squat']['down'] else (0 if avg > ANGLE_THRESH['squat']['up'] else 1)

    elif exe_key == 'deadlift' or active_exercise=='deadlift':
        key = 'deadlift'; exe_key='deadlift'
        angs = [
            calc_angle(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                       lm[mp_pose.PoseLandmark.RIGHT_HIP],
                       lm[mp_pose.PoseLandmark.RIGHT_KNEE]),
            calc_angle(lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
                       lm[mp_pose.PoseLandmark.LEFT_HIP],
                       lm[mp_pose.PoseLandmark.LEFT_KNEE])
        ]
        avg   = sum(angs)/2
        phase = 2 if avg < ANGLE_THRESH['deadlift']['down'] else (0 if avg > ANGLE_THRESH['deadlift']['up'] else 1)

    if key and phase is not None:
        if phase == 2 and not down_flag[key]:
            down_flag[key] = True
            active_exercise = key

        if phase == 0 and down_flag.get(active_exercise, False):
            cond_names = english_cond_map[exe_key]
            ok_flags   = [scores[i] > CONDITION_THRESHOLD for i in range(len(cond_names))]
            if all(ok_flags):
                total_reps[key]   += 1
                correct_reps[key] += 1
            else:
                incorrect_reps[key] += 1
            down_flag[key] = False
            active_exercise = None

    # 9.4) 시각화
    mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    for cond, score in zip(english_cond_map[exe_key], scores):
        color = (0,255,0) if score > CONDITION_THRESHOLD else (0,0,255)
        for lm_idx in delta_map[cond]:
            x, y = int(lm[lm_idx].x*w), int(lm[lm_idx].y*h)
            cv2.circle(frame, (x, y), 8, color, -1)

    # 9.5) 텍스트
    cv2.putText(frame,
                f"Squat  T={total_reps['squat']}  C={correct_reps['squat']}  W={incorrect_reps['squat']}",
                (10,30),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame,
                f"Deadlift  T={total_reps['deadlift']}  C={correct_reps['deadlift']}  W={incorrect_reps['deadlift']}",
                (10,60),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"Exercise: {exe_key}", (10,90),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"Phase: {phase}",      (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow('Feedback', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
