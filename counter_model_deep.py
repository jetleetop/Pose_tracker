import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# 1. 데이터 로드
X = np.load("X_sequences.npy")  # (시퀀스 수, 30, 8)
y_phase = np.load("y_phases.npy")  # (시퀀스 수,)
y_count = np.load("y_counts.npy")  # (시퀀스 수,)

# 2. 데이터 분할
X_train, X_test, y_phase_train, y_phase_test, y_count_train, y_count_test = train_test_split(
    X, y_phase, y_count, test_size=0.2, random_state=42
)

# 3. 모델 아키텍처
input_layer = Input(shape=(30, 8))  # 30프레임 × 8특징(4관절 x,y)
x = LSTM(64, return_sequences=False)(input_layer)
x = Dropout(0.3)(x)

# 출력 1: 상태 분류 (3클래스)
phase_output = Dense(3, activation='softmax', name='phase')(x)

# 출력 2: 카운트 예측 (이진 분류)
count_output = Dense(1, activation='sigmoid', name='count')(x)

# 4. 모델 컴파일
model = Model(inputs=input_layer, outputs=[phase_output, count_output])
model.compile(
    optimizer='adam',
    loss={
        'phase': 'sparse_categorical_crossentropy',  # 상태 분류
        'count': 'binary_crossentropy'               # 카운트 예측
    },
    metrics={
        'phase': 'accuracy',
        'count': 'accuracy'
    },
    loss_weights=[0.6, 0.4]  # 상태 분류에 더 높은 가중치
)

# 5. 학습
history = model.fit(
    X_train,
    {
        'phase': y_phase_train,
        'count': y_count_train
    },
    validation_data=(
        X_test,
        {'phase': y_phase_test, 'count': y_count_test}
    ),
    epochs=50,
    batch_size=32,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)

# 6. 모델 저장
model.save("exercise_phase_count_model.keras")