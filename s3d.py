import torch
import torch.nn as nn
import cv2
import numpy as np
from collections import deque
from torchvision.models.video import s3d, S3D_Weights

# ----------------------------
# CONFIG
# ----------------------------
FRAMES = 16               # number of frames to stack for prediction
IMG_SIZE = 224            # match training resolution
PREDICT_EVERY = 4         # how often to predict (in frames)
CONF_THRESHOLD = 0.6
device = "cuda" if torch.cuda.is_available() else "cpu"

# gesture labels (must match your training dataset order)
GESTURES = ['FistHalt', 'Swipe', 'ThumbsUp', 'Wave', 'ZoomIn']

# ----------------------------
# LOAD MODEL
# ----------------------------
print("🧠 Loading S3D model...")
weights = S3D_Weights.KINETICS400_V1  # or S3D_Weights.KINETICS400_V1 for pretrained backbone
model = s3d(weights=weights)

# Replace classifier head (exactly as during training)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Conv3d(1024, len(GESTURES), kernel_size=1, stride=1, bias=True)
)

# Load fine-tuned weights
model.load_state_dict(torch.load("s3d_gesture.pth", map_location=device))
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

    # Preprocess each frame
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0

    # Append to buffer
    frame_buffer.append(frame_tensor)
    frame_counter += 1

    # Make prediction every few frames
    if len(frame_buffer) == FRAMES and frame_counter % PREDICT_EVERY == 0:
        with torch.no_grad():
            # Convert frame buffer to model input: (1, 3, T, H, W)
            video = torch.stack(list(frame_buffer), dim=1).unsqueeze(0).to(device)
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

    # Display result
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imshow("Live Gesture Recognition (S3D)", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
