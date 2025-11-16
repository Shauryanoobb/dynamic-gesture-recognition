import cv2
import numpy as np
import tensorflow as tf
from collections import deque
print(
    tf.__version__
)
# ===== CONFIG =====
# GESTURES = ['FistHalt', 'Swipe', 'ThumbsUp', 'Wave', 'ZoomIn']
GESTURES = ['Again', 'FistHalt', 'Shoot', 'Sign', 'Swipe', 'Talk', 'Teacher', 'ThumbsUp', 'Wave', 'ZoomIn']

frames = 16
img_size = 112
# tflite_model_path = "quantised_GRU_QT.tflite"
# tflite_model_path = "models/quant_mobilevit_gru_112.tflite"
tflite_model_path = "quantised_GRU_QT.tflite"

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

# ===== Preprocessing function =====
def preprocess_frame(frame):
    frame = cv2.resize(frame, (img_size, img_size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype("float32") / 255.0
    return frame

# ===== Prediction function =====
def predict_clip(frames_seq):
    # Expand dims -> shape: (1, frames, H, W, C)
    input_data = np.expand_dims(np.array(frames_seq, dtype=np.float32), axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])
    return preds

# ===== Webcam =====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open webcam!")

print("🎥 Starting real-time gesture recognition (Quantized Model)... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not captured.")
        break

    # Flip horizontally for mirror view
    frame = cv2.flip(frame, 1)

    # Preprocess and add to buffer
    processed = preprocess_frame(frame)
    frame_buffer.append(processed)

    # Predict when buffer is full
    if len(frame_buffer) == frames:
        preds = predict_clip(frame_buffer)
        pred_label = GESTURES[np.argmax(preds)]
        conf = np.max(preds)

        # Display prediction
        cv2.putText(frame, f"{pred_label} ({conf:.2f})", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Quantized Gesture Recognition", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🟢 Webcam closed.")
