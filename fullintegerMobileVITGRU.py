import tensorflow as tf
from tensorflow.keras.models import Model
from keras_vision.MobileViT_v1 import build_MobileViT_v1
import numpy as np
import os

# ===== CONFIG =====
GESTURES = ['FistHalt', 'Swipe', 'ThumbsUp', 'Wave', 'ZoomIn']
frames = 16
img_size = 112
num_classes = len(GESTURES)

# ===== Build the same architecture =====
base = build_MobileViT_v1(
    model_type="XXS",
    pretrained=True,
    include_top=False,
    input_shape=(img_size, img_size, 3),
    num_classes=0
)

video_input = tf.keras.Input((frames, img_size, img_size, 3))
x = tf.keras.layers.TimeDistributed(base)(video_input)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)
x = tf.keras.layers.GRU(32, return_sequences=False)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = Model(video_input, output)

# ===== Load pretrained weights =====
model.load_weights("weights_only_mobilevit_gru.weights.h5")
print("✅ Original model loaded successfully.")

# ===== Optional: save model before conversion =====
# model.save("mobilevit_gru_full.h5")

# ===== Representative dataset for full integer quantization =====
def representative_data_gen():
    # Ideally, use real or preprocessed sample clips
    for _ in range(100):
        dummy_clip = np.random.rand(1, frames, img_size, img_size, 3).astype(np.float32)
        yield [dummy_clip]

# ===== Convert to TensorFlow Lite (Full Integer Quantization) =====
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Quantization setup
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# Allow fallback for unsupported ops (GRU / LayerNorm)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,  # Prefer full INT8 ops
    tf.lite.OpsSet.SELECT_TF_OPS          # Fallback for GRU if needed
]

# Force int8 input/output tensors
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Required for RNNs using TensorList
converter._experimental_lower_tensor_list_ops = False

# === Perform conversion ===
tflite_model = converter.convert()
print("✅ Model converted successfully with FULL integer quantization!")

# === Save the quantized model ===
output_path = "full_integer_quant_mobilevitgru_2.tflite"
with open(output_path, "wb") as f:
    f.write(tflite_model)
print(f"💾 Saved as '{output_path}'")

# === File size info ===
size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"📦 Quantized model size: {size_mb:.2f} MB")
