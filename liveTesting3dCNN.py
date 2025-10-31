import tensorflow as tf
import cv2
import numpy as np
from collections import deque

# ----------------------------
# Load trained model
# ----------------------------
model = tf.keras.models.load_model("gesture_3dcnn_mirrored.keras")
print("✅ Model loaded successfully!")

# Gesture classes
gesture_classes = ['FistHalt', 'Swipe', 'ThumbsUp', 'Wave', 'ZoomIn']

# ----------------------------
# Parameters
# ----------------------------
FRAMES = 16
IMG_SIZE = 112
PREDICT_EVERY = 4   # make prediction every 4 new frames (tunable)
CONF_THRESHOLD = 0.6  # confidence threshold to display

# ----------------------------
# Frame buffer (sliding window)
# ----------------------------
frame_buffer = deque(maxlen=FRAMES)

# ----------------------------
# Open webcam
# ----------------------------
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("❌ Cannot access webcam")
    exit()

print("🎥 Starting live gesture recognition... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess current frame
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_norm = frame_rgb / 255.0

    # Add to buffer
    frame_buffer.append(frame_norm)

    # Only predict if we have enough frames and at defined intervals
    if len(frame_buffer) == FRAMES and (cap.get(cv2.CAP_PROP_POS_FRAMES) % PREDICT_EVERY == 0):
        x = np.expand_dims(np.array(frame_buffer), axis=0)  # (1, 16, 112, 112, 3)
        pred = model.predict(x, verbose=0)[0]
        pred_class = np.argmax(pred)
        conf = pred[pred_class]

        # Display prediction if confident enough
        if conf > CONF_THRESHOLD:
            text = f"{gesture_classes[pred_class]} ({conf*100:.1f}%)"
            color = (0, 255, 0)
        else:
            text = "..."
            color = (0, 0, 255)
    else:
        text = "Collecting frames..."
        color = (255, 255, 0)

    # Overlay prediction
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imshow("Live Gesture Recognition", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
