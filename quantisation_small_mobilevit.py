import tensorflow as tf
from tensorflow.keras.models import Model
from keras_vision.MobileViT_v1 import build_MobileViT_v1

# ===== CONFIG =====
GESTURES = ['FistHalt', 'Swipe', 'ThumbsUp', 'Wave', 'ZoomIn']
frames = 16
img_size = 112
num_classes = len(GESTURES)

# ===== Build the same architecture =====
# === Backbone ===
updates = {
        # Reduce channel widths across all blocks
        "block_1_1_dims": 8,
        "block_1_2_dims": 16,
        "block_2_1_dims": 24,
        "block_2_2_dims": 24,
        "block_2_3_dims": 32,
        "block_3_1_dims": 40,
        "block_3_2_dims": 48,
        "block_4_1_dims": 64,
        "block_4_2_dims": 64,
        "block_5_1_dims": 80,
        "block_5_2_dims": 96,
        "final_conv_dims": 128,

        # Reduce transformer embedding dims and repeats
        "tf_block_3_dims": 64,
        "tf_block_4_dims": 80,
        "tf_block_5_dims": 96,
        "tf_block_3_repeats": 1,
        "tf_block_4_repeats": 1,
        "tf_block_5_repeats": 1,

        # Reduce expansion factor slightly
        "depthwise_expansion_factor": 2,
    }
# base = build_MobileViT_v1(
#     model_type="XXS",
#     pretrained=True,
#     include_top=False,
#     input_shape=(img_size, img_size,3),
#     num_classes=0
# )
base = build_MobileViT_v1(
        model_type="XXS",            # start from smallest base
        num_classes=0,               # feature extractor only
        input_shape=(img_size,img_size,3),
        include_top=False,
        pretrained=False,
        updates=updates,
        linear_drop=0.0,
        attention_drop=0.0,
        dropout=0.0,
    )


# base = build_MobileViT_v1(
#     model_type="XXS",
#     pretrained=True,
#     include_top=False,
#     input_shape=(img_size, img_size, 3),
#     num_classes=0
# )

video_input = tf.keras.Input((frames, img_size, img_size, 3))
x = tf.keras.layers.TimeDistributed(base)(video_input)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)
x = tf.keras.layers.GRU(32, return_sequences=False)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = Model(video_input, output)

# ===== Load pretrained weights =====
model.load_weights("models/new_best_small_mobilevit_gru (3).h5")

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
with open("true_small_model_dynamic_quant.tflite", "wb") as f:
    f.write(tflite_quant_model)
print("💾 Saved as 'true_small_model_dynamic_quant.tflite'")