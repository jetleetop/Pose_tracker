# 1) 라이브러리 임포트
import os
import cv2
import json
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# 2) 설정
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, 'aimodel', 'pose_condition_exercise_babel_model.keras')
IDX_PATH        = os.path.join(BASE_DIR, 'condition', 'exercise_index_babel.json')

# ─── Thresholds ───────────────────────────────────────
CONDITION_THRESHOLD = 0.4   # condition OK 임계치 (0.0 ~ 1.0)
# 운동별 Phase 각도 임계치
ANGLE_THRESH = {
    'squat':   {'down': 100, 'up': 160},   # 무릎(hip–knee–ankle) 각도
    'deadlift':{'down': 140, 'up': 170}    # 힙힌지(shoulder–hip–knee) 각도
}

# 3) 모델 & 매핑 로드
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
mp_pose  = mp.solutions.pose
delta_map = {
    'spine neutral':                   [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
                                         mp_pose.PoseLandmark.LEFT_HIP,      mp_pose.PoseLandmark.RIGHT_HIP],
    'head forward':                    [mp_pose.PoseLandmark.NOSE],
    'foot-knee alignment':            [mp_pose.PoseLandmark.LEFT_KNEE,     mp_pose.PoseLandmark.LEFT_ANKLE,
                                         mp_pose.PoseLandmark.RIGHT_KNEE,    mp_pose.PoseLandmark.RIGHT_ANKLE],
    'feet flat':                       [mp_pose.PoseLandmark.LEFT_HEEL,     mp_pose.PoseLandmark.RIGHT_HEEL],
    'bar-body closeness':              [mp_pose.PoseLandmark.LEFT_WRIST,    mp_pose.PoseLandmark.RIGHT_WRIST],
    'simultaneous knee-hip extension': [mp_pose.PoseLandmark.LEFT_HIP,      mp_pose.PoseLandmark.LEFT_KNEE,
                                         mp_pose.PoseLandmark.RIGHT_HIP,     mp_pose.PoseLandmark.RIGHT_KNEE]
}

# 5) MediaPipe 초기화
mp_draw = mp.solutions.drawing_utils
pose    = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# 6) 카운터 & 상태 초기화
down_flag       = {'squat': False, 'deadlift': False}
active_exercise = None
total_reps      = {'squat': 0, 'deadlift': 0}
correct_reps    = {'squat': 0, 'deadlift': 0}
incorrect_reps  = {'squat': 0, 'deadlift': 0}

# 7) 헬퍼 함수: 관절 각도 계산
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

# 9) 메인 루프
while True:
    ret, frame = cap.read()
    if not ret: break
    h, w, _ = frame.shape

    # 9.1) PoseLandmarks 추출
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result  = pose.process(rgb)
    if not result.pose_landmarks:
        cv2.imshow('Feedback', frame)
        if cv2.waitKey(1)==ord('q'): break
        continue
    lm = result.pose_landmarks.landmark

    # 9.2) 모델 예측 (conditions + exercise)
    inp         = np.array([[p.x, p.y] for p in lm[:24]]).flatten().reshape(1, -1)
    conds, pred = model.predict(inp, verbose=0)
    exe_key     = index_to_exercise[int(np.argmax(pred[0]))].lower()
    scores      = conds[0]

    # 9.3) Phase 계산 & 반복 카운팅
    phase, key = None, None
    if exe_key=='squat':
        key   = 'squat'
        hip_angs = [calc_angle(lm[mp_pose.PoseLandmark.RIGHT_HIP],
                               lm[mp_pose.PoseLandmark.RIGHT_KNEE],
                               lm[mp_pose.PoseLandmark.RIGHT_ANKLE]),
                    calc_angle(lm[mp_pose.PoseLandmark.LEFT_HIP],
                               lm[mp_pose.PoseLandmark.LEFT_KNEE],
                               lm[mp_pose.PoseLandmark.LEFT_ANKLE])]
        avg    = sum(hip_angs)/2
        phase  = 2 if avg < ANGLE_THRESH['squat']['down'] else (0 if avg > ANGLE_THRESH['squat']['up'] else 1)
    elif exe_key=='deadlift' or active_exercise=='deadlift':
        key   = 'deadlift'; exe_key='deadlift'
        hip_hinge = [calc_angle(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                                lm[mp_pose.PoseLandmark.RIGHT_HIP],
                                lm[mp_pose.PoseLandmark.RIGHT_KNEE]),
                     calc_angle(lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                lm[mp_pose.PoseLandmark.LEFT_HIP],
                                lm[mp_pose.PoseLandmark.LEFT_KNEE])]
        avg    = sum(hip_hinge)/2
        phase  = 2 if avg < ANGLE_THRESH['deadlift']['down'] else (0 if avg > ANGLE_THRESH['deadlift']['up'] else 1)

    # Down-Flag & 카운트 (조건 OK일 때만 total 증가)
    if key and phase is not None:
        if phase==2 and not down_flag[key]:
            down_flag[key] = True
            active_exercise = key

        if phase==0 and down_flag.get(active_exercise, False):
            # condition threshold 비교
            cond_names = english_cond_map[exe_key]
            ok_flags   = [scores[i] > CONDITION_THRESHOLD for i in range(len(cond_names))]
            if all(ok_flags):
                total_reps[key]   += 1
                correct_reps[key] += 1
            else:
                incorrect_reps[key] += 1
            down_flag[active_exercise] = False
            active_exercise = None

    # 9.4) Skeleton + Condition 시각화
    mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    for cond, score in zip(english_cond_map[exe_key], scores):
        color = (0,255,0) if score>CONDITION_THRESHOLD else (0,0,255)
        for lm_idx in delta_map[cond]:
            x, y = int(lm[lm_idx].x*w), int(lm[lm_idx].y*h)
            cv2.circle(frame, (x,y), 8, color, -1)

    # 9.5) 텍스트 오버레이
    cv2.putText(frame, f"Squat T={total_reps['squat']} C={correct_reps['squat']} W={incorrect_reps['squat']}",
                (10,30),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"Deadlift T={total_reps['deadlift']} C={correct_reps['deadlift']} W={incorrect_reps['deadlift']}",
                (10,60),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"Exercise: {exe_key}",    (10,90),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"Phase: {phase}",         (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow('Feedback', frame)
    if cv2.waitKey(1)==ord('q'): break

cap.release()
cv2.destroyAllWindows()
