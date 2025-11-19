import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from keras_vision.MobileViT_v1 import build_MobileViT_v1
from collections import deque

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 112 #change this according to the model trained
SEQ_LEN = 16 #last 16 frames to be considered for a gesture
GESTURES= ['Again', 'Shoot', 'Sign', 'Swipe', 'Talk', 'Teacher', 'ThumbsUp', 'Wave', 'ZoomIn']
MODEL_PATH = "models/LstmVit/mobilevit_lstm_9.keras"
CONF_THRESHOLD = 0.6
frame_skip = 2   # play with this
frame_count = 0
# -----------------------------
# REBUILD MODEL ARCHITECTURE
# -----------------------------
backbone = build_MobileViT_v1(
    model_type="XXS",
    pretrained=True,
    include_top=False,
    num_classes=0,
    input_shape=(IMG_SIZE, IMG_SIZE, 3) #uncomment if 112 
)
backbone.trainable = False

video_in = layers.Input((SEQ_LEN, IMG_SIZE, IMG_SIZE, 3))
x = layers.TimeDistributed(backbone)(video_in)
x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
x = layers.LSTM(128, dropout=0.3, return_sequences=False)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(len(GESTURES), activation='softmax')(x)

model = Model(video_in, out, name="MobileViT_LSTM_Gesture")

# Build with dummy input to initialize weights
dummy = tf.random.uniform((1, SEQ_LEN, IMG_SIZE, IMG_SIZE, 3))
_ = model(dummy)

# Load trained weights
model.load_weights(MODEL_PATH)
print("✅ Model loaded successfully!")

# -----------------------------
# LIVE TESTING (WEBCAM)
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access webcam")
    exit()

frame_buffer = deque(maxlen=SEQ_LEN)


print("🎥 Starting live gesture recognition... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # skip frames to simulate training frame rate

    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE)) #we can also use a bounding box here
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_norm = frame_rgb / 255.0
    frame_buffer.append(frame_norm)

    if len(frame_buffer) == SEQ_LEN: #also do you need prediction every frame?
        x = np.expand_dims(np.array(frame_buffer), axis=0)
        pred = model.predict(x, verbose=0)[0]
        cls = np.argmax(pred)
        conf = pred[cls]

        if conf > CONF_THRESHOLD:
            text = f"{GESTURES[cls]} ({conf*100:.1f}%)"
            color = (0, 255, 0)
        else:
            text = "..."
            color = (0, 0, 255)
    else:
        text = f"Collecting {SEQ_LEN-len(frame_buffer)} frames..."
        color = (255, 255, 0)

    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imshow("Live Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
