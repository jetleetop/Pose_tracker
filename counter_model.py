import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# 데이터 로드
X = np.load("X_sequences.npy")  # 관절 좌표 시퀀스
y_phase = np.load("y_phases.npy")  # 상태 레이블
y_count = np.load("y_counts.npy")  # 카운트 레이블

# 데이터 분할 (훈련 세트, 테스트 세트)
X_train, X_test, y_train_phase, y_test_phase, y_train_count, y_test_count = train_test_split(
    X, y_phase, y_count, test_size=0.2, random_state=42
)

# 모델 구성
input_layer = Input(shape=(30, 8))  # 30프레임, 각 프레임은 8개의 좌표

# LSTM 층: 시퀀스 데이터를 처리할 수 있는 모델
lstm_layer = LSTM(64, return_sequences=False)(input_layer)
lstm_layer = Dropout(0.2)(lstm_layer)  # Dropout을 추가하여 과적합 방지

# 운동 상태 예측
phase_output = Dense(3, activation='softmax', name='phase_output')(lstm_layer)  # 3개 상태 (idle, crunching, top)

# 카운트 예측
count_output = Dense(1, activation='sigmoid', name='count_output')(lstm_layer)  # 이진 분류 (카운트 증가 여부)

# 모델 정의
model = Model(inputs=input_layer, outputs=[phase_output, count_output])

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.001),
              loss={'phase_output': 'sparse_categorical_crossentropy', 'count_output': 'binary_crossentropy'},
              metrics={'phase_output': 'accuracy', 'count_output': 'accuracy'})

# 모델 요약
model.summary()

# 모델 훈련
history = model.fit(X_train, [y_train_phase, y_train_count],
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_test, [y_test_phase, y_test_count]),
                    verbose=1)

# 모델 평가
test_loss, test_phase_loss, test_count_loss, test_phase_acc, test_count_acc = model.evaluate(
    X_test, [y_test_phase, y_test_count]
)

print(f"테스트 정확도 (운동 상태): {test_phase_acc}")
print(f"테스트 정확도 (카운트 예측): {test_count_acc}")

# 모델 저장
model.save("exercise_model.h5")
