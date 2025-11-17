import socket
import cv2
import threading
import struct

PI_IP = "10.133.118.191"
VIDEO_PORT = 5005
MAC_PRED_PORT = 6005

prediction_text = "Waiting..."
prediction_lock = threading.Lock()

#see if resizing on mac side leads to lesser latency, also focus on brining good results because latency is already low

# ---------- UDP Prediction Receiver ----------
def listen_predictions():
    global prediction_text
    sock_pred = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_pred.bind(("0.0.0.0", MAC_PRED_PORT))
    print(f"👂 Listening for predictions on UDP:{MAC_PRED_PORT}")

    while True:
        try:
            data, _ = sock_pred.recvfrom(1024)
            with prediction_lock:
                prediction_text = data.decode()
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")

threading.Thread(target=listen_predictions, daemon=True).start()

# ---------- UDP Video Sender ----------
sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open webcam")

FRAME_SKIP = 1  # Send every 3rd frame (reduced from 4 for better temporal model performance)
JPEG_QUALITY = 60  # increase this and see do we get better results? tradeoff with latency, if udp is costly
MAX_PACKET_SIZE = 1472  # MTU 1500 - IP header (20) - UDP header (8) = safe UDP payload

frame_count = 0
frame_id = 0  # Unique ID for each frame sent

w = int(cap.get(3))
h = int(cap.get(4))
roi_size = int(w * 0.5)
x = (w - roi_size) // 2
y = (h - roi_size) // 2

print(f"🎥 Webcam: {w}x{h}, ROI: {roi_size}x{roi_size}")
print("📤 Sending frames with chunking to handle large sizes")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1) #see if needed, i dont think it is right? It is more needed for your intuition when seeing yourself on screen
    frame_count += 1

    # Draw ROI
    cv2.rectangle(frame, (x, y), (x+roi_size, y+roi_size), (0, 255, 0), 2)

    # Send frame with chunking
    if frame_count % FRAME_SKIP == 0:
        roi = frame[y:y+roi_size, x:x+roi_size]
        _, jpeg = cv2.imencode(".jpg", roi, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        frame_bytes = jpeg.tobytes()
        
        # Calculate chunk size (account for 8-byte header)
        data_per_chunk = MAX_PACKET_SIZE - 8
        total_size = len(frame_bytes)
        total_chunks = (total_size + data_per_chunk - 1) // data_per_chunk  # Ceiling division
        
        print(f"🔍 Frame {frame_id}: Total size = {total_size} bytes, will send in {total_chunks} chunks")
        
        # Send each chunk with header: frame_id + chunk_num + total_chunks + data
        for chunk_num in range(total_chunks):
            start = chunk_num * data_per_chunk
            end = min(start + data_per_chunk, total_size)
            chunk_data = frame_bytes[start:end]
            
            # Pack header: frame_id (4 bytes) + chunk_num (2 bytes) + total_chunks (2 bytes)
            header = struct.pack(">IHH", frame_id, chunk_num, total_chunks)
            packet = header + chunk_data
            
            # Print actual chunk size
            print(f"  📦 Chunk {chunk_num}/{total_chunks-1}: header={len(header)} + data={len(chunk_data)} = TOTAL {len(packet)} bytes")
            
            try:
                sock_video.sendto(packet, (PI_IP, VIDEO_PORT))
                print(f"  ✅ Sent successfully")
            except OSError as e:
                print(f"  ❌ Send failed: {e} (packet size was {len(packet)})")
        
        frame_id = (frame_id + 1) % 10000  # Wrap around to prevent overflow

    # Display prediction
    with prediction_lock:
        text = prediction_text
    
    # Parse gesture and confidence
    if "|" in text:
        gesture, conf_str = text.split("|")
        display_text = f"{gesture} ({conf_str})"
    else:
        display_text = text
    
    # Text styling
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    (text_w, text_h), _ = cv2.getTextSize(display_text, font, font_scale, thickness)
    text_x = x + (roi_size - text_w) // 2
    text_y = y - 15
    
    # Background for readability
    cv2.rectangle(frame, 
                 (text_x - 5, text_y - text_h - 5),
                 (text_x + text_w + 5, text_y + 5),
                 (0, 0, 0), -1)
    
    cv2.putText(frame, display_text, (text_x, text_y),
                font, font_scale, (0, 255, 0), thickness)

    # Frame info
    cv2.putText(frame, f"Frame: {frame_count}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("UDP Gesture Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🟢 Client closed")