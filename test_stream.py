import cv2
import threading
import time
from flask import Flask, Response

app    = Flask(__name__)
latest = {'frame': None, 'counter': 0}
lock   = threading.Lock()

def capture():
    cap = cv2.VideoCapture(2, cv2.CAP_V4L2)  # CAM0(USB2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while True:
        ret, frame = cap.read()
        if ret:
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            with lock:
                latest['frame']   = buf.tobytes()
                latest['counter'] += 1

def generate():
    last = -1
    while True:
        with lock:
            c = latest['counter']
            d = latest['frame']
        if d and c != last:
            last = c
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + d + b'\r\n'
        else:
            time.sleep(0.005)

@app.route('/stream')
def stream():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

threading.Thread(target=capture, daemon=True).start()
print("▶ 스트리밍 시작: http://localhost:5002/stream")
app.run(host='0.0.0.0', port=5002, threaded=True)
