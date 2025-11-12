import tensorflow as tf
from tensorflow.keras import layers, Model
from keras_vision.MobileViT_v1 import build_MobileViT_v1
import cv2 
from collections import deque
import numpy as np

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



# ===== Frame Buffer =====
frame_buffer = deque(maxlen=frames)  # stores last N frames

# ===== Preprocessing function =====
def preprocess_frame(frame):
    frame = cv2.resize(frame, (img_size, img_size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype("float32") / 255.0
    return frame

# ===== Webcam =====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open webcam!")

print("🎥 Starting real-time gesture recognition... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not captured.")
        break

    # Flip horizontally for natural mirror view
    frame = cv2.flip(frame, 1)
    
    # Add to buffer
    processed = preprocess_frame(frame)
    frame_buffer.append(processed)

    # Predict when we have enough frames
    if len(frame_buffer) == frames:
        input_clip = np.expand_dims(np.array(frame_buffer), axis=0)  # shape: (1, frames, H, W, C)
        preds = model.predict(input_clip, verbose=0)
        pred_label = GESTURES[np.argmax(preds)]
        conf = np.max(preds)

        # Show prediction
        cv2.putText(frame, f"{pred_label} ({conf:.2f})", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Gesture Recognition", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🟢 Webcam closed.")



