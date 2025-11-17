import time
import numpy as np
import cv2
import tensorflow as tf
from collections import deque

# ------------------------------
# CONFIG
# ------------------------------
FRAMES = 16
IMG_SIZE = 112
MODEL_PATH = "models/cnn/3dCNN10signs_quantisedf16.tflite"

GESTURES = [
    'Again', 'FistHalt', 'Shoot', 'Sign', 'Swipe',
    'Talk', 'Teacher', 'ThumbsUp', 'Wave', 'ZoomIn'
]

# ------------------------------
# LOAD TFLITE MODEL
# ------------------------------
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ------------------------------
# BUFFER + TIMING STORAGE
# ------------------------------
frame_buffer = deque(maxlen=FRAMES)
inference_times = []

# ------------------------------
# PREPROCESS FUNC (same as before)
# ------------------------------
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return img

# ------------------------------
# INFERENCE FUNCTION
# ------------------------------
def run_inference():
    arr = np.expand_dims(np.array(frame_buffer, dtype=np.float32), axis=0)

    # ---- TIME START ----
    t1 = time.time()

    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()

    # ---- TIME END ----
    t2 = time.time()

    # Timing in ms
    infer_ms = (t2 - t1) * 1000
    inference_times.append(infer_ms)

    preds = interpreter.get_tensor(output_details[0]['index'])
    idx = np.argmax(preds)

    return GESTURES[idx], float(preds[0][idx]), infer_ms

# ------------------------------
# DUMMY LOOP (simulate receiving frames)
# Replace this with your actual camera/UDP receive
# ------------------------------
print("🔬 Starting inference timing test...")

for i in range(200):
    # Simulate receiving a frame (replace this)
    dummy_frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

    processed = preprocess(dummy_frame)
    frame_buffer.append(processed)

    if len(frame_buffer) == FRAMES:
        gesture, conf, infer_ms = run_inference()
        print(f"[Inference] {gesture} ({conf:.2f})  |  Time = {infer_ms:.2f} ms")

print("\n---------------------------")
print("Inference Stats")
print("---------------------------")

if inference_times:
    print(f"Total inferences: {len(inference_times)}")
    print(f"Avg inference time: {np.mean(inference_times):.2f} ms")
    print(f"Min: {np.min(inference_times):.2f} ms")
    print(f"Max: {np.max(inference_times):.2f} ms")
else:
    print("No inferences were made.")

print("\nDone.")
