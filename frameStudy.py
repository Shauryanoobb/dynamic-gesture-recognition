import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- SAME SETTINGS AS BEFORE ---
video_path = "waveTest.mp4"
model_path = "gesture_3dcnn.keras"
FRAMES = 16
IMG_SIZE = (112, 112)

def preprocess_video(path, frames=FRAMES, size=IMG_SIZE):
    cap = cv2.VideoCapture(path)
    frames_list = []
    while len(frames_list) < frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        frame = frame.astype("float32") / 255.0   # ✅ normalize again
        frames_list.append(frame)
    cap.release()

    frames_list = np.array(frames_list)
    if len(frames_list) < frames:
        pad = np.repeat(frames_list[-1][None, ...], frames - len(frames_list), axis=0)
        frames_list = np.concatenate([frames_list, pad])
    return np.expand_dims(frames_list, axis=0), frames_list  # return both (model input, raw frames)

# Load and preprocess
x, frames_list = preprocess_video(video_path)
print("Input shape:", x.shape)
print("Frame mean/std:", x.mean(), x.std())

# --- SHOW A FEW SAMPLE FRAMES ---
num_to_show = 5
indices = np.linspace(0, len(frames_list)-1, num_to_show, dtype=int)

plt.figure(figsize=(15, 3))
for i, idx in enumerate(indices):
    plt.subplot(1, num_to_show, i+1)
    plt.imshow(cv2.cvtColor((frames_list[idx]*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title(f"Frame {idx+1}")
    plt.axis("off")
plt.suptitle("Sample Frames Passed to Model", fontsize=14)
plt.show()
