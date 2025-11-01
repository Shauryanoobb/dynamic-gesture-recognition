import torch
import torch.nn as nn
import cv2
import numpy as np
from collections import deque
from torchvision.models.video import s3d

# ----------------------------
# CONFIG
# ----------------------------
FRAMES = 16               # number of frames to stack for prediction
IMG_SIZE = 224            # match training resolution
PREDICT_EVERY = 4         # how often to predict (in frames)
CONF_THRESHOLD = 0.6
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# LOAD QUANTIZED MODEL
# ----------------------------
print("🧠 Loading quantized S3D model...")

# Load quantized checkpoint
checkpoint = torch.load("s3d_gesture_int8.pth", map_location="cpu")
quantized_state_dict = checkpoint['state_dict']
scales = checkpoint['scales']
GESTURES = checkpoint['gestures']

print(f"📝 Gestures: {GESTURES}")

# Dequantize weights back to float32 for inference
dequantized_state_dict = {}
for name, param in quantized_state_dict.items():
    if name in scales:
        # Dequantize: convert int8 back to float32
        dequantized = param.to(torch.float32) * scales[name]
        dequantized_state_dict[name] = dequantized
    else:
        dequantized_state_dict[name] = param

# Rebuild model architecture
model = s3d(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Conv3d(1024, len(GESTURES), kernel_size=1, stride=1, bias=True)
)

# Load dequantized weights
model.load_state_dict(dequantized_state_dict)
model = model.to(device)
model.eval()

print("✅ Quantized model loaded successfully!")

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