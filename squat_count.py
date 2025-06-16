import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 데이터 로드
X = np.load("X_squat_sequences.npy")  # 시퀀스 데이터
y_count = np.load("y_squat_counts.npy")  # 카운트 라벨

# 데이터 정규화 (옵션)
X = X.astype('float32') / 255.0  # x, y 좌표 정규화 (범위 0~1)

# 모델 설계
model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))  # LSTM 레이어
model.add(Dropout(0.2))  # 과적합 방지
model.add(Dense(32, activation='relu'))  # Dense 레이어
model.add(Dense(1))  # 출력 레이어: 카운트 값 예측 (연속 값)

# 모델 컴파일
model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])

# 모델 요약
model.summary()

# 모델 학습
model.fit(X, y_count, epochs=20, batch_size=32, validation_split=0.2)

# 모델 저장
model.save('squat_count_model.h5')

# 모델 평가 (선택사항)
# test_loss, test_mae = model.evaluate(X_test, y_test)
