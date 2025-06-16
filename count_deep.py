import os
import json
import math

def calc_distance(p1, p2):
    return math.hypot(p1['x']-p2['x'], p1['y']-p2['y'])

json_dir = "output_json"
json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])

# 임계값 설정
THRESHOLDS = {
    'right': {'top': 0.08, 'mid': 0.18},
    'left': {'top': 0.08, 'mid': 0.18}
}
tolerance = 0.02  # 상단 인식 완화 값

# 상태 추적 변수
state_history = {
    'right': {'prev': 'idle', 'done': False},
    'left': {'prev': 'idle', 'done': False}
}
count = 0

for json_file in json_files:
    with open(os.path.join(json_dir, json_file), 'r', encoding='utf-8') as f:
        data = json.load(f)

    try:
        joints = {
            'right': {
                'elbow': data['RIGHT_ELBOW'],
                'knee': data['RIGHT_KNEE']
            },
            'left': {
                'elbow': data['LEFT_ELBOW'],
                'knee': data['LEFT_KNEE']
            }
        }
    except KeyError:
        continue

    # 각 관절 상태 판별
    current_states = {}
    for side in ['right', 'left']:
        dist = calc_distance(joints[side]['elbow'], joints[side]['knee'])
        threshold = THRESHOLDS[side]

        if dist < (threshold['top'] + tolerance):
            current_states[side] = 'top'
        elif dist < threshold['mid']:
            current_states[side] = 'crunching'
        else:
            current_states[side] = 'idle'

    # 완료 플래그 체크 (top → crunching/idle 전환 시)
    for side in ['right', 'left']:
        if state_history[side]['prev'] == 'top' and current_states[side] != 'top':
            state_history[side]['done'] = True

    # 양쪽 중 한쪽이라도 완료되면 카운트
    if state_history['right']['done'] and state_history['left']['done']:
        count += 1
        # 플래그 초기화
        state_history['right']['done'] = False
        state_history['left']['done'] = False

    # 이전 상태 업데이트
    for side in ['right', 'left']:
        state_history[side]['prev'] = current_states[side]

    # 결과 저장
    data['label'] = f"R:{current_states['right']} L:{current_states['left']}"
    data['count'] = count
    with open(os.path.join(json_dir, json_file), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

print(f"[✅ 완료] 총 카운트: {count}회")