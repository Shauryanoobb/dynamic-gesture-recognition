import tensorflow as tf
from tensorflow.keras import layers, Model
from keras_vision.MobileViT_v1 import build_MobileViT_v1

# === CONFIG ===
GESTURES = ['FistHalt', 'Swipe', 'ThumbsUp', 'Wave', 'ZoomIn']
frames = 16
img_size = 112
num_classes = len(GESTURES)

# === Backbone ===
base = build_MobileViT_v1(
    model_type="XXS",
    pretrained=False,
    include_top=False,
    num_classes=0,
    input_shape=(img_size, img_size, 3)
)

# === Video input ===
video_input = tf.keras.Input((frames, img_size, img_size, 3))

# --- Step 1: Merge frames into channels ---
# Reshape (B, T, H, W, C) → (B, H, W, T*C)
x = layers.Reshape((img_size, img_size, frames * 3))(video_input)

# --- Step 2: 1x1 Conv to compress temporal channels ---
x = layers.Conv2D(3, (1, 1), activation="relu", padding="same")(x)

# --- Step 3: Pass through MobileViT backbone ---
x = base(x)

# --- Step 4: Global pooling (spatial) ---
x = layers.GlobalAveragePooling2D()(x)

# --- Step 5: Temporal feature learning (Conv1D over synthetic time axis) ---
# Create synthetic temporal dimension
x = layers.RepeatVector(frames)(x)  # shape: (B, T, features)

x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)

# --- Step 6: Temporal pooling ---
x = layers.GlobalAveragePooling1D()(x)

# --- Step 7: Classification head ---
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(num_classes, activation="softmax")(x)

model = Model(video_input, output)

dummy_input = tf.random.uniform((1, frames,img_size,img_size, 3))
model(dummy_input)

MODEL_PATH = "hailmary_part3.h5"
model.load_weights(MODEL_PATH)

model.trainable = False
num_classes = model.output_shape[-1]

print("Model loaded")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("3rd_hailmaryfullofgrace.tflite","wb") as f:
    f.write(tflite_model)
    
print("goat nano")


