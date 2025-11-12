import tensorflow as tf
from tensorflow.keras.models import Model
from keras_vision.MobileViT_v1 import build_MobileViT_v1

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
model.load_weights("models/new_best_112_mobilevit_gru.h5")
print("✅ Original model loaded successfully.")

# === Convert to TensorFlow Lite (compatible mode) ===
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Only allow built-in TFLite ops (no SELECT_TF_OPS)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

# Force model lowering to older op versions (<= 11)
converter._experimental_lower_to_numbered_tflite_version = 11

# Disable TensorList lowering (helps GRU)
converter._experimental_lower_tensor_list_ops = False

# Enable lightweight dynamic quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Optional: ensure reproducibility and simpler kernels
converter.experimental_new_converter = True

# === Convert ===
tflite_model = converter.convert()
print("✅ Model converted successfully for older runtime (opset <= 11)")

# === Save the quantized model ===
with open("model_compatible_v11.tflite", "wb") as f:
    f.write(tflite_model)

print("💾 Saved as 'model_compatible_v11.tflite'")
