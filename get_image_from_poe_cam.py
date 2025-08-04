# client_display_stream.py
import socket
import struct
import cv2
import numpy as np
import time

HOST = "192.168.192.2"
PORT = 5000

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((HOST, PORT))
print("Connected to server")

def recv_exact(sock, length):
    data = b''
    while len(data) < length:
        packet = sock.recv(length - len(data))
        if not packet:
            return None
        data += packet
    return data

frame_cnt = 0
fps = 0
start_time = time.time()

try:
    while True:
        # Receive 4-byte header for JPEG length
        header = recv_exact(client, 4)
        if not header:
            break
        (jpeg_size,) = struct.unpack(">I", header)

        # Receive the actual JPEG data
        jpeg_data = recv_exact(client, jpeg_size)
        if not jpeg_data:
            break

        # Decode and display
        img = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        orig_height, orig_width = img.shape[:2]
        new_width = orig_width // 2
        new_height = orig_height // 2
        img = cv2.resize(img, (new_width, new_height))

        frame_cnt += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_cnt / elapsed_time
            frame_cnt = 0
            start_time = time.time()

        cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("TCP Stream", img)
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    client.close()