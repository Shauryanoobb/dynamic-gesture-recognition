import torch
import torch.nn as nn
import cv2
import numpy as np
from collections import deque
from torchvision.models.video import swin3d_t

# ----------------------------
# CONFIG
# ----------------------------
FRAMES = 16
IMG_SIZE = 112
PREDICT_EVERY = 4
CONF_THRESHOLD = 0.6
device = "cuda" if torch.cuda.is_available() else "cpu"

# Gesture classes (must match training)
GESTURES = ['Again', 'FistHalt', 'Shoot', 'Sign', 'Swipe', 'Talk', 'Teacher', 'ThumbsUp', 'Wave', 'ZoomIn']

# ----------------------------
# LOAD MODEL
# ----------------------------
model = swin3d_t(pretrained=False)
num_features = model.head.in_features
model.head = nn.Linear(num_features, len(GESTURES))
model.load_state_dict(torch.load("models/pretrainedVideoVits/best_swin3d_model (1).pth", map_location=device))
model = model.to(device)
model.eval()

print("✅ Model loaded successfully!")

# ----------------------------
# SLIDING FRAME BUFFER
# ----------------------------
frame_buffer = deque(maxlen=FRAMES)

# ----------------------------
# START WEBCAM
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access webcam")
    exit()

print("🎥 Starting live gesture recognition... Press 'q' to quit.")

frame_counter = 0
text, color = "Collecting frames...", (255, 255, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0

    # Append to buffer
    frame_buffer.append(frame_tensor)
    frame_counter += 1

    # Make prediction every few frames
    if len(frame_buffer) == FRAMES and frame_counter % PREDICT_EVERY == 0:
        with torch.no_grad():
            video = torch.stack(list(frame_buffer), dim=1).unsqueeze(0).to(device)  # (1, 3, T, H, W)
            preds = model(video)
            probs = torch.softmax(preds, dim=1)[0]
            conf, pred_class = torch.max(probs, dim=0)
            conf = conf.item()
            pred_class = pred_class.item()

        if conf > CONF_THRESHOLD:
            text = f"{GESTURES[pred_class]} ({conf*100:.1f}%)"
            color = (0, 255, 0)
        else:
            text = "..."
            color = (0, 0, 255)
    elif len(frame_buffer) < FRAMES:
        text, color = "Collecting frames...", (255, 255, 0)

    # Display
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imshow("Live Gesture Recognition", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
