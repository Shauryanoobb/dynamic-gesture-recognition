import cv2
import socket
import struct
import threading

# -------------------------
# Configuration
# -------------------------
PI_IP = "10.133.118.191"  # Change to your Pi's IP
PI_PORT = 6001
MAC_PORT = 5006

# -------------------------
# Connect to Pi
# -------------------------
sock_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print(f"🔌 Connecting to Pi at {PI_IP}:{PI_PORT}...")
sock_send.connect((PI_IP, PI_PORT))
print("✅ Connected to Pi")

# -------------------------
# Prediction Receiver (runs in background)
# -------------------------
prediction_text = "Waiting..."
prediction_lock = threading.Lock()

def listen_predictions():
    global prediction_text
    sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_recv.bind(("0.0.0.0", MAC_PORT))
    print(f"👂 Listening for predictions on port {MAC_PORT}...")
    
    while True:
        try:
            data, _ = sock_recv.recvfrom(1024)
            with prediction_lock:
                prediction_text = data.decode()
        except Exception as e:
            print(f"⚠️ Prediction receive error: {e}")
            break

# Start prediction listener thread
threading.Thread(target=listen_predictions, daemon=True).start()

# -------------------------
# ROI Configuration
# -------------------------
FRAME_SKIP = 1  # Send every 3rd frame for optimal latency with temporal model
JPEG_QUALITY = 40  # Balance quality and bandwidth

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open webcam")

frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Square ROI centered on frame
roi_size = int(0.45 * frame_w)  # Larger for better hand visibility
x = (frame_w - roi_size) // 2
y = (frame_h - roi_size) // 2
ROI = (x, y, roi_size, roi_size)

print(f"🎥 Webcam opened: {frame_w}x{frame_h}")
print(f"📦 ROI: {roi_size}x{roi_size} at ({x}, {y})")
print("Press 'q' to quit")

# -------------------------
# Main Loop
# -------------------------
frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame")
            continue
        
        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # Draw ROI bounding box
        x, y, w, h = ROI
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Send ROI to Pi (with frame skipping)
        if frame_count % FRAME_SKIP == 0:
            roi = frame[y:y+h, x:x+w]
            _, encoded = cv2.imencode(".jpg", roi, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            data = encoded.tobytes()
            
            # Send with length prefix
            try:
                sock_send.sendall(struct.pack(">L", len(data)) + data)
            except Exception as e:
                print(f"❌ Send error: {e}")
                break
        
        # -------------------------
        # Display Prediction
        # -------------------------
        with prediction_lock:
            text = prediction_text
        
        font_scale = 1.2
        thickness = 3
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size for centering
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = x + (w - text_w) // 2
        text_y = y - 20
        
        # Background rectangle for readability
        cv2.rectangle(frame, 
                     (text_x - 5, text_y - text_h - 5),
                     (text_x + text_w + 5, text_y + 5), 
                     (0, 0, 0), -1)
        
        # Draw prediction text
        cv2.putText(frame, text, (text_x, text_y),
                   font, font_scale, (0, 255, 0), thickness)
        
        # Show frame info
        info_text = f"FPS: ~{30//FRAME_SKIP} | Frame: {frame_count}"
        cv2.putText(frame, info_text, (10, frame_h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display
        cv2.imshow("Gesture Recognition (Mac)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user")
except Exception as e:
    print(f"❌ Error: {e}")
finally:
    cap.release()
    sock_send.close()
    cv2.destroyAllWindows()
    print("🟢 Client closed")