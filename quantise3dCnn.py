import tensorflow as tf
from tensorflow import keras

MODEL_PATH = 'models/3dCNN10Signs.keras'
model = keras.models.load_model(MODEL_PATH)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# ADD THIS LINE - it's critical for 3D CNN models
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

tflite_model = converter.convert()

tflite_path = 'models/3dCNN10signs.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"✓ Model saved: {tflite_path}")