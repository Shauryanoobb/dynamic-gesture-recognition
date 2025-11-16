import cv2
import socket
import struct

SERVER_IP = "10.200.80.191"
PORT = 5000

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((SERVER_IP, PORT))
print("Connected to Raspberry Pi")

def send_frame(frame):
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 20])
    data = jpeg.tobytes()
    msg = struct.pack(">I", len(data)) + data
    sock.sendall(msg)

def crop_center(frame, crop_size=200):
    h, w, _ = frame.shape
    ch, cw = crop_size, crop_size
    x1 = w//2 - cw//2
    y1 = h//2 - ch//2
    x2 = x1 + cw
    y2 = y1 + ch
    return frame[y1:y2, x1:x2]

def recv_prediction():
    header = sock.recv(4)
    if not header:
        return None, 0
    msg_len = struct.unpack(">I", header)[0]
    body = sock.recv(msg_len).decode()
    gesture, conf = body.split("|")
    return gesture, float(conf)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not working.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Send to Pi
    cropped = crop_center(frame, crop_size=200)
    small = cv2.resize(cropped, (160, 160))
    send_frame(cropped)

    # Receive inference
    gesture, conf = recv_prediction()
    if gesture != "None":
        cv2.putText(frame, f"{gesture} ({conf:.2f})",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)

    cv2.imshow("Remote Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
sock.close()
cv2.destroyAllWindows()
