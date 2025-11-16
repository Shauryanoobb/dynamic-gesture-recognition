import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# ===== CONFIG =====
GESTURES = ['FistHalt', 'Swipe', 'ThumbsUp', 'Wave', 'ZoomIn']
frames = 16
img_size = 112
tflite_model_path = "models/quant_mobilevit_gru_112.tflite"
video_path = "test_clips/waveTest.mp4"   # <-- put your video file here

# ===== Load TFLite model =====
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ TFLite model loaded successfully.")
print(f"Input shape: {input_details[0]['shape']}")
print(f"Output shape: {output_details[0]['shape']}")

# ===== Frame buffer =====
frame_buffer = deque(maxlen=frames)

# ===== Preprocessing =====
def preprocess_frame(frame):
    frame = cv2.resize(frame, (img_size, img_size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype("float32") / 255.0
    return frame

# ===== Prediction =====
def predict_clip(frames_seq):
    input_data = np.expand_dims(np.array(frames_seq, dtype=np.float32), axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])
    return preds

# ===== Video Input =====
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open test video!")

print("🎥 Running gesture recognition on test video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    processed = preprocess_frame(frame)
    frame_buffer.append(processed)

    if len(frame_buffer) == frames:
        preds = predict_clip(frame_buffer)
        pred_label = GESTURES[np.argmax(preds)]
        conf = np.max(preds)
        cv2.putText(frame, f"{pred_label} ({conf:.2f})", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition (Test Video)", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Inference complete.")
