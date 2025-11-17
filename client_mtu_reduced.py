import socket
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import struct

VIDEO_PORT = 5005      # Pi receives video here (UDP)
PRED_PORT = 6005       # Pi sends predictions back to Mac

GESTURES = ['Again', 'FistHalt', 'Shoot', 'Sign', 'Swipe', 'Talk', 'Teacher', 'ThumbsUp', 'Wave', 'ZoomIn']
FRAMES = 16
IMG_SIZE = 112

# Chunking parameters
CHUNK_SIZE = 1472  # MTU-safe chunk size (1500 - 20 IP - 8 UDP)

# ------ Load TFLite model ------
print("📦 Loading model...")
interpreter = tf.lite.Interpreter(model_path="models/GruVitQTD/10_people_16_frames_quantised_GRU_QT.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✅ Model loaded")

frame_buffer = deque(maxlen=FRAMES)

def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return img

def predict():
    arr = np.expand_dims(np.array(frame_buffer, dtype=np.float32), axis=0)
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])
    idx = np.argmax(preds)
    return GESTURES[idx], float(preds[0][idx])

# ------------- UDP sockets -------------
sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_video.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 200000)  # Increase buffer
sock_video.bind(("0.0.0.0", VIDEO_PORT))

sock_pred = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"📡 Listening for video on UDP:{VIDEO_PORT}")

# Frame reconstruction buffer
frame_chunks = {}
expected_chunks = {}

frame_count = 0
INFERENCE_EVERY = 4  # Run inference every 4 frames to reduce load
last_prediction = ("Buffering", 0.0)  # Store last prediction to send continuously

while True:
    try:
        data, mac_addr = sock_video.recvfrom(65535)
        
        # Parse header: frame_id (4 bytes) + chunk_num (2 bytes) + total_chunks (2 bytes) + data
        if len(data) < 8:
            continue
            
        frame_id = struct.unpack(">I", data[0:4])[0]
        chunk_num = struct.unpack(">H", data[4:6])[0]
        total_chunks = struct.unpack(">H", data[6:8])[0]
        chunk_data = data[8:]
        
        # Initialize frame buffer if new frame
        if frame_id not in frame_chunks:
            frame_chunks[frame_id] = {}
            expected_chunks[frame_id] = total_chunks
        
        # Store chunk
        frame_chunks[frame_id][chunk_num] = chunk_data
        
        # Check if all chunks received
        if len(frame_chunks[frame_id]) == expected_chunks[frame_id]:
            # Reconstruct frame
            sorted_chunks = [frame_chunks[frame_id][i] for i in range(total_chunks)]
            full_data = b"".join(sorted_chunks)
            
            # Decode frame
            np_data = np.frombuffer(full_data, np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
            
            # Clean up old frame data
            del frame_chunks[frame_id]
            del expected_chunks[frame_id]
            
            # Clean up very old frames (keep only last 3)
            if len(frame_chunks) > 3:
                oldest = min(frame_chunks.keys())
                del frame_chunks[oldest]
                if oldest in expected_chunks:
                    del expected_chunks[oldest]
            
            if frame is not None:
                frame_buffer.append(preprocess(frame))
                frame_count += 1
                
                # Run inference periodically
                if len(frame_buffer) == FRAMES and frame_count % INFERENCE_EVERY == 0:
                    gesture, conf = predict()
                    last_prediction = (gesture, conf)  # Update stored prediction
                    print(f"{gesture} ({conf:.3f})")
                
                # Always send the last prediction (even if we didn't just compute a new one)
                msg = f"{last_prediction[0]}|{last_prediction[1]:.4f}".encode()
                sock_pred.sendto(msg, (mac_addr[0], PRED_PORT))
    
    except Exception as e:
        print(f"❌ Error: {e}")
        continue