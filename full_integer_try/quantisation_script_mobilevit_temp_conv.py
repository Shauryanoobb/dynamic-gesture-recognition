import tensorflow as tf
from tensorflow.keras.models import Model
from keras_vision.MobileViT_v1 import build_MobileViT_v1

# ===== CONFIG =====
GESTURES = ['FistHalt', 'Swipe', 'ThumbsUp', 'Wave', 'ZoomIn']
frames = 16
img_size = 256
num_classes = len(GESTURES)

# === Backbone ===
base = build_MobileViT_v1(
    model_type="XXS",
    pretrained=False,
    include_top=False,
    num_classes=0
    # input_shape=(img_size, img_size, 3)
)

# === Video model ===
video_input = tf.keras.Input((frames, img_size, img_size, 3))

# extract per-frame features
x = tf.keras.layers.TimeDistributed(base)(video_input)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)
# shape: (batch, frames, features)

# === Replace GRU with Quantization-Friendly Temporal Block ===
# Temporal convolution simulates sequence modeling but is quantizable
x = tf.keras.layers.Conv1D(
    filters=64,
    kernel_size=3,
    activation="relu",
    padding="same"
)(x)
x = tf.keras.layers.Conv1D(
    filters=32,
    kernel_size=3,
    activation="relu",
    padding="same"
)(x)

# Global average pooling over time
x = tf.keras.layers.GlobalAveragePooling1D()(x)

# === Classification head ===
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(video_input, output)
dummy_input = tf.random.uniform((1, frames, img_size, img_size, 3))
model(dummy_input) 
# ===== Load pretrained weights =====
model.load_weights("test_hailmary.keras")

model.trainable = False

print("✅ Original model loaded successfully.")

# # ===== Save full model before conversion =====
# model.save("mobilevit_gru_full.h5")
# print("💾 Saved Keras model as 'mobilevit_gru_full.h5'")

# === Convert to TensorFlow Lite with dynamic range quantization ===
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# # Allow TensorFlow ops that TFLite doesn't support natively (e.g., TensorListReserve)
# converter.target_spec.supported_ops = [
#     tf.lite.OpsSet.TFLITE_BUILTINS,
#     tf.lite.OpsSet.SELECT_TF_OPS
# ]

# # Disable TensorList lowering — required for RNNs / GRUs
# converter._experimental_lower_tensor_list_ops = False

# Enable dynamic range quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# (Optional) reduce logging verbosity
# converter.experimental_new_converter = True

# === Convert ===
tflite_quant_model = converter.convert()
print("✅ Model converted successfully with dynamic range quantization!")

# === Save the quantized model ===
with open("try_dynamic_quant_mobilevit_temp_conv.tflite", "wb") as f:
    f.write(tflite_quant_model)
print("💾 Saved as 'NO_WEIGHTS_try_dynamic_quant_mobilevit_temp_conv.tflite'")