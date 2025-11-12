#check this, wont work on mac
import cv2
import numpy as np
from collections import deque
import tensorflow as tf 

# ----------------------------
# Load quantized TFLite model
# ----------------------------
TFLITE_MODEL_PATH = "models/checking3dCnnquantisation.tflite"
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ TFLite model loaded successfully!")
print(f"Input shape: {input_details[0]['shape']}")
print(f"Output shape: {output_details[0]['shape']}")

# Gesture classes
GESTURES = ['FistHalt', 'Swipe', 'ThumbsUp', 'Wave', 'ZoomIn']

# ----------------------------
# Parameters
# ----------------------------
FRAMES = 16
IMG_SIZE = 112
PREDICT_EVERY = 4
CONF_THRESHOLD = 0.6

# ----------------------------
# Frame buffer
# ----------------------------
frame_buffer = deque(maxlen=FRAMES)

# ----------------------------
# Webcam Input
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access webcam")
    exit()

print("🎥 Running gesture recognition... Press 'q' to quit.")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_norm = frame_rgb.astype(np.float32) / 255.0
    frame_buffer.append(frame_norm)

    frame_count += 1
    if len(frame_buffer) == FRAMES and (frame_count % PREDICT_EVERY == 0):
        input_data = np.expand_dims(np.array(frame_buffer, dtype=np.float32), axis=0)

        # Set tensor and run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        pred_idx = np.argmax(output_data)
        conf = output_data[pred_idx]

        if conf > CONF_THRESHOLD:
            text = f"{GESTURES[pred_idx]} ({conf*100:.1f}%)"
            color = (0, 255, 0)
        else:
            text = "..."
            color = (0, 0, 255)
    else:
        text = "Collecting frames..."
        color = (255, 255, 0)

    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imshow("3D CNN Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Inference complete.")
