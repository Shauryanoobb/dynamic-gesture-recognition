import tensorflow as tf
import cv2
import numpy as np
import os

# ----------------------------
# Load trained model
# ----------------------------
model = tf.keras.models.load_model("gesture_3dcnn.keras")
print("✅ Model loaded successfully!")

# Gesture classes (same order as training)
gesture_classes = ['FistHalt', 'Swipe', 'ThumbsUp', 'Wave', 'ZoomIn']

# ----------------------------
# Video preprocessing function (evenly sampled frames)
# ----------------------------
FRAMES = 16
IMG_SIZE = 112

def preprocess_video(video_path, num_frames=FRAMES, size=IMG_SIZE):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total_frames-1, 0), num_frames, dtype=int)

    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (size, size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    frames = np.array(frames)

    # Pad with zeros if fewer frames
    if len(frames) < num_frames:
        pad = np.zeros((num_frames - len(frames), size, size, 3))
        frames = np.concatenate([frames, pad])

    # Normalize like training
    frames = frames / 255.0

    # Add batch dimension
    return np.expand_dims(frames, axis=0)  # shape (1, T, H, W, C)

# ----------------------------
# Pick any test video
# ----------------------------
test_video_path = "waveTest.mp4"  # <-- replace this with your local path

# ----------------------------
# Run prediction
# ----------------------------
x = preprocess_video(test_video_path)
pred = model.predict(x)[0]

print("\n--- Prediction Results ---")
for i, cls in enumerate(gesture_classes):
    print(f"{cls:<10}: {pred[i]*100:.2f}%")
print("---------------------------")
print("🎯 Predicted class:", gesture_classes[np.argmax(pred)])
