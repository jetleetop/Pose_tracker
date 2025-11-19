import os
import json
import numpy as np
import re
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

# 281번부터 328번까지 학습 대상 파일 필터
def is_target_file(fn):
    m = re.match(r"D\d{2}-\d-(\d{3})\.json$", fn)
    if not m:
        return False
    num = int(m.group(1))
    return 281 <= num <= 328

# condition map 생성
def build_condition_map(json_folder):
    cmap = {}
    for fn in os.listdir(json_folder):
        if not fn.endswith('.json') or fn.endswith('-3d.json') or not is_target_file(fn):
            continue
        data = json.load(open(os.path.join(json_folder, fn), encoding='utf-8'))
        ex = data['type_info']['exercise']
        conds = [c['condition'] for c in data['type_info']['conditions']]
        if ex not in cmap:
            cmap[ex] = conds
    with open('exercise_condition_babel_map.json','w',encoding='utf-8') as f:
        json.dump(cmap,f,ensure_ascii=False,indent=2)
    return cmap

# 데이터 로딩 (각 view를 개별 샘플로)
def load_data(json_folder, condition_map):
    X, C, Y, idx = [], [], [], {}
    cnt = 0
    maxc = max(len(v) for v in condition_map.values())
    for fn in os.listdir(json_folder):
        if not fn.endswith('.json') or fn.endswith('-3d.json') or not is_target_file(fn):
            continue
        data = json.load(open(os.path.join(json_folder, fn), encoding='utf-8'))
        ex = data['type_info']['exercise']
        conds = data['type_info']['conditions']
        if ex not in condition_map or len(conds)!=len(condition_map[ex]):
            continue
        if ex not in idx:
            idx[ex] = cnt; cnt+=1
        vec = [int(c['value']) for c in conds] + [0]*(maxc-len(conds))
        for frame in data['frames']:
            for view in ['view1','view2','view3','view4','view5']:
                pts = frame.get(view,{}).get('pts',{})
                kp=[]; ok=True
                for j in JOINT_NAMES:
                    if j not in pts:
                        ok=False; break
                    kp += [pts[j]['x']/IMG_WIDTH, pts[j]['y']/IMG_HEIGHT]
                if ok:
                    X.append(kp); C.append(vec); Y.append(idx[ex])
    return (np.array(X),
            np.array(C),
            to_categorical(np.array(Y)),
            idx,
            maxc)

# 경로 설정
json_folder = 'json_files_babel'
condition_map = build_condition_map(json_folder)

# 데이터 로딩
X, y_cond, y_exe, exe_idx, max_cond = load_data(json_folder, condition_map)
print(f"✅ Loaded {len(X)} samples, input shape {X.shape}")

# 학습/검증 분할
X_tr, X_te, yc_tr, yc_te, ye_tr, ye_te = train_test_split(
    X, y_cond, y_exe, test_size=0.2, random_state=42)

# 모델 구성
inp = Input(shape=(len(JOINT_NAMES)*2,))
x = Dense(256,activation='relu')(inp)
x = Dense(128,activation='relu')(x)
x = Dense(64,activation='relu')(x)

out_cond = Dense(max_cond,activation='sigmoid',name='conditions')(x)
out_exe  = Dense(len(exe_idx),activation='softmax',name='exercise')(x)

model = Model(inputs=inp, outputs=[out_cond,out_exe])
model.compile(
    optimizer=Adam(1e-3),
    loss={'conditions':'binary_crossentropy','exercise':'categorical_crossentropy'},
    metrics={'conditions':'accuracy','exercise':'accuracy'}
)

# 학습
model.fit(
    X_tr,
    {'conditions':yc_tr,'exercise':ye_tr},
    validation_data=(X_te,{'conditions':yc_te,'exercise':ye_te}),
    epochs=50,
    batch_size=32
)

# 저장
model.save('pose_model.h5')
with open('exercise_index_babel.json','w',encoding='utf-8') as f:
    json.dump(exe_idx,f,ensure_ascii=False,indent=2)

print("✅ Training complete and model saved as pose_condition_exercise_babel_model.keras")