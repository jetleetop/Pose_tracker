import os
import json
import numpy as np
from collections import deque

# 경로 설정
json_dir = "output_json"
sequence_length = 30  # 30프레임(1초) 시퀀스
feature_keys = ['RIGHT_ELBOW', 'RIGHT_KNEE', 'LEFT_ELBOW', 'LEFT_KNEE']

# 상태 매핑 딕셔너리 (라벨 파싱용)
phase_map = {
    'idle': 0,
    'crunching': 1,
    'top': 2
}

# 데이터 저장용
X_sequences = []  # 관절 좌표 시퀀스
y_phases = []  # 상태 라벨 (다중 출력)
y_counts = []  # 카운트 라벨 (이진 분류)

# JSON 파일 순회
json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])
for json_file in json_files:
    with open(os.path.join(json_dir, json_file), 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 관절 좌표 추출
    frame_data = []
    for key in feature_keys:
        joint = data[key]
        frame_data.extend([joint['x'], joint['y']])

    # 상태 라벨 파싱 (예: "R:top L:idle" → "top")
    phase_label = data['label'].split(' ')[0].split(':')[1]  # 첫 번째 상태 추출
    phase_encoded = phase_map[phase_label]

    # 카운트 증가 여부 (이전 프레임과 비교)
    count_label = 1 if data['count'] > data.get('prev_count', 0) else 0

    X_sequences.append(frame_data)
    y_phases.append(phase_encoded)
    y_counts.append(count_label)


# 시퀀스 생성 함수
def create_sequences(data, labels, seq_length):
    sequences = []
    seq_labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        seq_labels.append(labels[i + seq_length - 1])  # 시퀀스의 마지막 프레임 라벨 사용
    return np.array(sequences), np.array(seq_labels)


# 시퀀스 생성
X, y_phase = create_sequences(X_sequences, y_phases, sequence_length)
_, y_count = create_sequences(X_sequences, y_counts, sequence_length)

# 저장
np.save("X_sequences.npy", X)
np.save("y_phases.npy", y_phase)
np.save("y_counts.npy", y_count)

print(f"생성된 시퀀스: {X.shape}")
print(f"상태 라벨 분포: {np.bincount(y_phase)}")
print(f"카운트 라벨 분포: {np.bincount(y_count)}")