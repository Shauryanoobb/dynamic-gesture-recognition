import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque
from keras_vision.MobileViT_v1 import build_MobileViT_v1
import tensorflow as tf
from tensorflow.keras.models import Model

# ===== CONFIG =====

GESTURES = ['FistHalt', 'Swipe', 'ThumbsUp', 'Wave', 'ZoomIn']  # <-- update to your actual class list

# ===== Load Model =====
frames = 16
img_size = 112
num_classes = len(GESTURES)  # update to len(GESTURES)

# === Backbone ===
base = build_MobileViT_v1(
    model_type="XXS",
    pretrained=True,
    include_top=False,
    input_shape=(img_size, img_size,3),
    num_classes=0
)

# === Video model ===
video_input = tf.keras.Input((frames, img_size, img_size, 3))
x = tf.keras.layers.TimeDistributed(base)(video_input)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)
x = tf.keras.layers.GRU(32, return_sequences=False)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = Model(video_input, output)
# model.summary()

# Load weights only
model.load_weights("models/new_best_112_mobilevit_gru.h5")  # <-- change this to the .h5 weights you saved

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
