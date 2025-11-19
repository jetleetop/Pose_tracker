import tensorflow as tf

# 1. 모델 로드 (.h5)
model = tf.keras.models.load_model("pose_model.h5")

# 2. SavedModel 형식으로 내보내기
model.export("saved_pose_model")  # Keras 3에서 SavedModel 저장용

# 3. TFLite 변환
converter = tf.lite.TFLiteConverter.from_saved_model("saved_pose_model")
tflite_model = converter.convert()

# 4. .tflite 파일로 저장
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
