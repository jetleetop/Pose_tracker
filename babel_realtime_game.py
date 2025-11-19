import os
import cv2
import json
import csv
import time
import numpy as np
import pandas as pd
import mediapipe as mp
from datetime import datetime
from tensorflow.keras.models import load_model

# 1) 설정
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, 'aimodel', 'pose_condition_exercise_babel_model.keras')
IDX_PATH        = os.path.join(BASE_DIR, 'condition', 'exercise_index_babel.json')
CONDITION_THRESHOLD = 0.375  # 판정 완화
ANGLE_THRESH = {
    'squat':   {'down': 100, 'up': 160},
    'deadlift':{'down': 140, 'up': 170}
}

# 2) 모델 및 매핑
model = load_model(MODEL_PATH)
with open(IDX_PATH, 'r', encoding='utf-8') as f:
    exercise_to_index = json.load(f)
index_to_exercise = {v: k for k, v in exercise_to_index.items()}

# 3) 조건-관절 맵핑
mp_pose = mp.solutions.pose
english_cond_map = {
    'deadlift': ['spine neutral','foot-knee alignment','bar-body closeness','simultaneous knee-hip extension','feet flat'],
    'squat':    ['spine neutral','head forward','foot-knee alignment','feet flat']
}
delta_map = {
    'spine neutral':                   [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP],
    'head forward':                    [mp_pose.PoseLandmark.NOSE],
    'foot-knee alignment':            [mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE],
    'feet flat':                       [mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.RIGHT_HEEL],
    'bar-body closeness':              [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST],
    'simultaneous knee-hip extension':[mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE]
}

# 4) 초기화
mp_draw = mp.solutions.drawing_utils
pose    = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
down_flag       = {'squat': False, 'deadlift': False}
active_exercise = None
total_reps      = {'squat': 0, 'deadlift': 0}
correct_reps    = {'squat': 0, 'deadlift': 0}
incorrect_reps  = {'squat': 0, 'deadlift': 0}
combo_streak    = 0
bonus_score     = 0
combo_start_time = None

# 5) 헬퍼 함수
def calc_angle(a, b, c):
    va, vb, vc = np.array([a.x, a.y]), np.array([b.x, b.y]), np.array([c.x, c.y])
    ba, bc    = va - vb, vc - vb
    cosang    = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

def show_store(score):
    print("\n\U0001f6d2 [상점] 구매 가능 아이템")
    if score >= 300: print("✅ PT 1회권 (300점)")
    if score >= 200: print("✅ 단백질 음료 쿠폰 (200점)")
    if score >= 100: print("✅ 손목 스트랩 (100점)")
    if score < 100:  print("❌ 아직 구매 가능한 아이템이 없습니다. 더 운동하세요!")

def save_result_csv(file_path='results.csv'):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([now, 'squat', total_reps['squat'], correct_reps['squat'], incorrect_reps['squat']])
        writer.writerow([now, 'deadlift', total_reps['deadlift'], correct_reps['deadlift'], incorrect_reps['deadlift']])
    print(f"[✅ 저장 완료] {file_path}에 기록됨.")

    if os.path.exists(file_path):
        df = pd.read_csv(file_path, header=None, names=['Timestamp','Exercise','Total','Correct','Incorrect'])
        score = df['Correct'].sum() * 5 + bonus_score
        level = score // 100
        print(f"[📊 누적 점수] {score} 점")
        print(f"[🧬 현재 레벨] Lv.{level}")
        show_store(score)

# 6) 웹캠 열기
for idx in range(4):
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        print("Camera index:", idx)
        break

# 저장할 영상 설정 (avi, XVID 코덱 사용)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps    = cap.get(cv2.CAP_PROP_FPS) or 30   # FPS 못 읽으면 기본 30
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out    = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

cv2.namedWindow('Feedback', cv2.WINDOW_NORMAL)
# 필요시 전체화면: cv2.setWindowProperty('Feedback', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 7) 메인 루프
while True:
    ret, frame = cap.read()
    if not ret: break
    h, w, _ = frame.shape

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result  = pose.process(rgb)
    if not result.pose_landmarks:
        cv2.imshow('Feedback', frame)
        out.write(frame)
        if cv2.waitKey(1)==ord('q'): break
        continue

    lm = result.pose_landmarks.landmark
    inp = np.array([[p.x, p.y] for p in lm[:24]]).flatten().reshape(1, -1)
    conds, pred = model.predict(inp, verbose=0)
    exe_key     = index_to_exercise[int(np.argmax(pred[0]))].lower()

    # (1) KeyError 방지
    if exe_key not in english_cond_map:
        cv2.imshow('Feedback', frame)
        out.write(frame)
        if cv2.waitKey(1)==ord('q'): break
        continue

    scores      = conds[0]

    phase, key = None, None
    if exe_key=='squat':
        key   = 'squat'
        hip_angs = [calc_angle(lm[mp_pose.PoseLandmark.RIGHT_HIP], lm[mp_pose.PoseLandmark.RIGHT_KNEE], lm[mp_pose.PoseLandmark.RIGHT_ANKLE]),
                    calc_angle(lm[mp_pose.PoseLandmark.LEFT_HIP], lm[mp_pose.PoseLandmark.LEFT_KNEE], lm[mp_pose.PoseLandmark.LEFT_ANKLE])]
        avg    = sum(hip_angs)/2
        phase  = 2 if avg < ANGLE_THRESH['squat']['down'] else (0 if avg > ANGLE_THRESH['squat']['up'] else 1)
    elif exe_key=='deadlift' or active_exercise=='deadlift':
        key   = 'deadlift'; exe_key='deadlift'
        hinge = [calc_angle(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER], lm[mp_pose.PoseLandmark.RIGHT_HIP], lm[mp_pose.PoseLandmark.RIGHT_KNEE]),
                 calc_angle(lm[mp_pose.PoseLandmark.LEFT_SHOULDER], lm[mp_pose.PoseLandmark.LEFT_HIP], lm[mp_pose.PoseLandmark.LEFT_KNEE])]
        avg   = sum(hinge)/2
        phase = 2 if avg < ANGLE_THRESH['deadlift']['down'] else (0 if avg > ANGLE_THRESH['deadlift']['up'] else 1)

    if key and phase is not None:
        if phase==2 and not down_flag[key]:
            down_flag[key] = True
            active_exercise = key

        if phase==0 and down_flag.get(active_exercise, False):
            cond_names = english_cond_map[exe_key]
            ok_flags   = [scores[i] > CONDITION_THRESHOLD for i in range(len(cond_names))]
            ok_ratio = sum(ok_flags) / len(ok_flags)

            if exe_key == 'squat':
                condition_passed = ok_ratio >= 0.75
            else:
                condition_passed = all(ok_flags)

            # === 변경된 부분: 잘못된 운동도 total에 포함되도록 항상 total_reps 증가 ===
            total_reps[key] += 1

            if condition_passed:
                correct_reps[key] += 1
                combo_streak += 1
                combo_start_time = time.time()  # 콤보 유지 시마다 갱신
                if combo_streak % 5 == 0:
                    bonus_score += 10
                    print(f"🔥 Combo {combo_streak}! +10 보너스 점수 획득!")
            else:
                incorrect_reps[key] += 1
                combo_streak = 0
                combo_start_time = None

            down_flag[active_exercise] = False
            active_exercise = None

    # 9.4 시각화
    mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    for cond, score in zip(english_cond_map[exe_key], scores):
        color = (0,255,0) if score>CONDITION_THRESHOLD else (0,0,255)
        for lm_idx in delta_map[cond]:
            x, y = int(lm[lm_idx].x*w), int(lm[lm_idx].y*h)
            cv2.circle(frame, (x,y), 8, color, -1)

    # 9.5 텍스트
    cv2.putText(frame, f"Squat T={total_reps['squat']} C={correct_reps['squat']} W={incorrect_reps['squat']}", (10,30),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    cv2.putText(frame, f"Deadlift T={total_reps['deadlift']} C={correct_reps['deadlift']} W={incorrect_reps['deadlift']}", (10,60),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    cv2.putText(frame, f"Exercise: {exe_key}",    (10,90),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    cv2.putText(frame, f"Phase: {phase}",         (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    if combo_streak >= 1 and combo_start_time:
        elapsed = int(time.time() - combo_start_time)
        cv2.putText(frame, f"🔥 Combo {combo_streak} ({elapsed}s)", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.imshow( 'Feedback', frame)
    out.write(frame)   # 영상 저장
    if cv2.waitKey(1)==ord('q'): break

# 종료 처리
cap.release()
out.release()
cv2.destroyAllWindows()
save_result_csv()
