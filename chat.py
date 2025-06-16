import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# 이미지 크기
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

# 사용할 관절 이름
JOINT_NAMES = [
    'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
    'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
    'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle',
    'Neck', 'Left Palm', 'Right Palm', 'Back', 'Waist',
    'Left Foot', 'Right Foot'
]

# JSON에서 조건 맵 만들기
def build_condition_map(json_folder):
    exercise_condition_map = {}
    for file_name in os.listdir(json_folder):
        if not file_name.endswith('.json') or file_name.endswith('-3d.json'):
            continue
        file_path = os.path.join(json_folder, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            exercise = data['type_info']['exercise']
            conditions = data['type_info']['conditions']
            condition_names = [c['condition'] for c in conditions]
            if exercise not in exercise_condition_map:
                exercise_condition_map[exercise] = condition_names
        except Exception as e:
            print(f"[Error] {file_name}: {e}")
            continue
    # 저장
    with open('exercise_condition_map.json', 'w', encoding='utf-8') as f:
        json.dump(exercise_condition_map, f, ensure_ascii=False, indent=2)
    return exercise_condition_map

# 조건맵 불러오기 (없으면 자동 생성)
def load_condition_map(path, json_folder):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return build_condition_map(json_folder)

# 데이터 불러오기
def load_data_with_flexible_conditions(json_folder, condition_map):
    features = []
    condition_vectors = []
    exercise_labels = []
    exercise_to_index = {}
    index_counter = 0
    if not condition_map:
        raise ValueError("Condition map is empty. Check your JSON files and structure.")
    max_condition_count = max(len(v) for v in condition_map.values())

    for file_name in os.listdir(json_folder):
        if not file_name.endswith('.json') or file_name.endswith('-3d.json'):
            continue
        file_path = os.path.join(json_folder, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            exercise = data['type_info']['exercise']
            conditions = data['type_info']['conditions']

            if exercise not in condition_map:
                continue
            expected_cond_names = condition_map[exercise]
            if len(conditions) != len(expected_cond_names):
                continue
            if exercise not in exercise_to_index:
                exercise_to_index[exercise] = index_counter
                index_counter += 1

            cond_vector = [int(c['value']) for c in conditions]
            cond_vector += [0] * (max_condition_count - len(cond_vector))

            for frame in data['frames']:
                view_keypoints = []
                for view in ['view1', 'view2', 'view3', 'view4', 'view5']:
                    pts = frame.get(view, {}).get('pts', {})
                    keypoints = []
                    valid = True
                    for joint in JOINT_NAMES:
                        if joint not in pts:
                            valid = False
                            break
                        x = pts[joint]['x'] / IMG_WIDTH
                        y = pts[joint]['y'] / IMG_HEIGHT
                        keypoints.extend([x, y])
                    if valid:
                        view_keypoints.append(np.array(keypoints))
                if len(view_keypoints) == 5:
                    avg_keypoints = np.mean(view_keypoints, axis=0)
                    features.append(avg_keypoints)
                    condition_vectors.append(cond_vector)
                    exercise_labels.append(exercise_to_index[exercise])
        except Exception as e:
            print(f"[Error] {file_name}: {e}")
            continue

    return (
        np.array(features),
        np.array(condition_vectors),
        to_categorical(np.array(exercise_labels)),
        exercise_to_index,
        max_condition_count
    )

# 경로 설정
json_folder = 'json_files'
condition_map_path = 'exercise_condition_map.json'

# 조건맵 불러오기 or 생성
condition_map = load_condition_map(condition_map_path, json_folder)

# 데이터 로딩
X, y_conditions, y_exercise, exercise_to_index, max_condition_count = load_data_with_flexible_conditions(json_folder, condition_map)
print(f"Loaded {len(X)} samples with shape {X.shape}")

# 학습/테스트 데이터 분할
X_train, X_test, y_cond_train, y_cond_test, y_exe_train, y_exe_test = train_test_split(
    X, y_conditions, y_exercise, test_size=0.2, random_state=42)

# 모델 구성
input_layer = Input(shape=(len(JOINT_NAMES) * 2,))
x = Dense(256, activation='relu')(input_layer)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)

condition_output = Dense(max_condition_count, activation='sigmoid', name='conditions')(x)
exercise_output = Dense(len(exercise_to_index), activation='softmax', name='exercise')(x)

model = Model(inputs=input_layer, outputs=[condition_output, exercise_output])
model.compile(
    optimizer=Adam(0.001),
    loss={'conditions': 'binary_crossentropy', 'exercise': 'categorical_crossentropy'},
    metrics={'conditions': 'accuracy', 'exercise': 'accuracy'}
)

# 학습
model.fit(
    X_train,
    {'conditions': y_cond_train, 'exercise': y_exe_train},
    validation_data=(X_test, {'conditions': y_cond_test, 'exercise': y_exe_test}),
    epochs=50,
    batch_size=32
)

# 저장
model.save('pose_condition_exercise_model.keras')
with open('exercise_index.json', 'w', encoding='utf-8') as f:
    json.dump(exercise_to_index, f, ensure_ascii=False, indent=2)
