import os
import json
import numpy as np

# 1️⃣ 관절 사이 각도 계산 함수
def calculate_angle(a, b, c):
    a = np.array([a['x'], a['y']])
    b = np.array([b['x'], b['y']])
    c = np.array([c['x'], c['y']])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# 2️⃣ 스쿼트 상태 감지 기준
def detect_phase_from_angle(knee_angle):
    if knee_angle > 160:
        return 'top'
    elif knee_angle < 100:
        return 'bottom'
    else:
        return 'middle'

# 3️⃣ 경로 설정
json_dir = "output_json_squat"
sequence_length = 30

# 4️⃣ 데이터 준비
X_all, y_phases_all, y_counts_all = [], [], []
phase_map = {'top': 0, 'middle': 1, 'bottom': 2}
joint_keys = ['RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE', 'LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE']

prev_phase = 'top'
current_count = 0

json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])

for file in json_files:
    file_path = os.path.join(json_dir, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 5️⃣ 무릎 각도 계산
    right_angle = calculate_angle(data['RIGHT_HIP'], data['RIGHT_KNEE'], data['RIGHT_ANKLE'])
    left_angle = calculate_angle(data['LEFT_HIP'], data['LEFT_KNEE'], data['LEFT_ANKLE'])
    knee_angle = (right_angle + left_angle) / 2

    # 6️⃣ 상태 판단 및 JSON에 label 추가
    phase = detect_phase_from_angle(knee_angle)
    data['label'] = phase

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    # 7️⃣ 카운트 로직: bottom → top 전환 시 count 증가
    if prev_phase == 'bottom' and phase == 'top':
        current_count += 1
    prev_phase = phase

    # 8️⃣ 프레임 벡터 구성
    frame_vector = []
    for key in joint_keys:
        frame_vector.extend([data[key]['x'], data[key]['y']])

    X_all.append(frame_vector)
    y_phases_all.append(phase_map[phase])
    y_counts_all.append(current_count)

# 9️⃣ 카운트 변화 감지 (프레임별)
y_count_change = [1 if i > 0 and y_counts_all[i] > y_counts_all[i - 1] else 0 for i in range(len(y_counts_all))]

# 🔟 시퀀스 생성 함수
def create_sequences(data, labels1, labels2, seq_len):
    X_seq, Y1, Y2 = [], [], []
    for i in range(len(data) - seq_len + 1):
        X_seq.append(data[i:i + seq_len])
        Y1.append(labels1[i + seq_len - 1])  # 마지막 프레임의 상태
        Y2.append(labels2[i + seq_len - 1])  # 마지막 프레임의 카운트 변화
    return np.array(X_seq), np.array(Y1), np.array(Y2)

# ⓫ 시퀀스 생성
X_seq, y_phase_seq, y_count_seq = create_sequences(X_all, y_phases_all, y_count_change, sequence_length)

# ⓬ .npy 파일로 저장
np.save("X_squat_sequences.npy", X_seq)
np.save("y_squat_phases.npy", y_phase_seq)
np.save("y_squat_counts.npy", y_count_seq)

# ⓭ 정보 출력
print(f"시퀀스 shape: {X_seq.shape}")
print(f"상태 라벨 분포: {np.bincount(y_phase_seq)}")
print(f"카운트 라벨 분포: {np.bincount(y_count_seq)}")
print("JSON 파일에 label 추가 및 시퀀스 데이터 저장 완료!")
