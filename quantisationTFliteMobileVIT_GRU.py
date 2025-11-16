import tensorflow as tf
from tensorflow.keras.models import Model
from keras_vision.MobileViT_v1 import build_MobileViT_v1

# ===== CONFIG =====
# GESTURES = ['FistHalt', 'Swipe', 'ThumbsUp', 'Wave', 'ZoomIn']
GESTURES = ['Again', 'FistHalt', 'Shoot', 'Sign', 'Swipe', 'Talk', 'Teacher', 'ThumbsUp', 'Wave', 'ZoomIn']
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
model.load_weights("10signs_5people_16frames_best_mobilevit_gru_unquantisable.h5")

print("✅ Original model loaded successfully.")

# # ===== Save full model before conversion =====
# model.save("mobilevit_gru_full.h5")
# print("💾 Saved Keras model as 'mobilevit_gru_full.h5'")

# === Convert to TensorFlow Lite with dynamic range quantization ===
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Allow TensorFlow ops that TFLite doesn't support natively (e.g., TensorListReserve)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

# Disable TensorList lowering — required for RNNs / GRUs
converter._experimental_lower_tensor_list_ops = False

# Enable dynamic range quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# (Optional) reduce logging verbosity
converter.experimental_new_converter = True

# === Convert ===
tflite_quant_model = converter.convert()
print("✅ Model converted successfully with dynamic range quantization!")

# === Save the quantized model ===
with open("10_signs_unquantisable_model_dynamic_quant.tflite", "wb") as f:
    f.write(tflite_quant_model)
print("💾 Saved as 'model_dynamic_quant.tflite'")