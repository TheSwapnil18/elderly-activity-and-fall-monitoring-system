import cv2
import socket
import time
import numpy as np
import platform
from flask import Flask, Response, render_template_string

app = Flask(__name__)

# Try to capture from the default built-in laptop webcam
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def get_local_ip():
    """Finds the local IP address of the laptop on the network."""
    try:
        # Create a temporary socket to check local network interface IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

LOCAL_IP = get_local_ip()
STREAM_URL = f"http://{LOCAL_IP}:8000/video_feed"

def gen_frames():
    global camera
    failed_frames = 0
    while True:
        try:
            if camera is None or not camera.isOpened():
                camera = cv2.VideoCapture(0, cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY)
                time.sleep(1.0)
                
            success, frame = camera.read()
            if not success:
                failed_frames += 1
                if failed_frames > 15:  # 0.5s of failure
                    camera.release()
                    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY)
                    failed_frames = 0
                
                # Draw elegant placeholder frame with neon emergency alert text
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "WEBCAM DISCONNECTED / IN USE", (60, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                ret, buffer = cv2.imencode('.jpg', placeholder)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.1)
                continue
                
            failed_frames = 0
            # Compress the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # Cap streaming rate around 30 FPS to conserve CPU and LAN bandwidth
        except Exception as e:
            print(f"Error in streaming frame: {e}")
            time.sleep(0.5)

@app.route('/video_feed')
def video_feed():
    """Endpoint that streams the raw camera frames as MJPEG."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Simple status page shown on the laptop."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Elderly Monitor - Laptop Camera Streamer</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #0f172a;
                color: #f1f5f9;
                margin: 0;
                padding: 20px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
            }
            .card {
                background: rgba(30, 41, 59, 0.7);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
                padding: 30px;
                max-width: 600px;
                width: 100%;
                text-align: center;
                box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
            }
            h1 {
                background: linear-gradient(135deg, #818cf8, #c084fc);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-top: 0;
            }
            .status {
                display: inline-block;
                padding: 6px 12px;
                background-color: #10b981;
                color: white;
                border-radius: 20px;
                font-size: 14px;
                font-weight: 600;
                margin-bottom: 20px;
            }
            .url-container {
                background: #020617;
                padding: 15px;
                border-radius: 8px;
                font-family: 'Courier New', Courier, monospace;
                color: #38bdf8;
                font-size: 16px;
                font-weight: bold;
                border: 1px solid #1e293b;
                margin: 15px 0;
                word-break: break-all;
                user-select: all;
            }
            .instructions {
                text-align: left;
                color: #94a3b8;
                font-size: 14px;
                line-height: 1.6;
            }
            .instructions ol {
                padding-left: 20px;
            }
            .video-preview {
                width: 100%;
                max-width: 480px;
                height: 270px;
                border-radius: 12px;
                background: #020617;
                margin-top: 20px;
                border: 1px solid rgba(255, 255, 255, 0.05);
                object-fit: cover;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>📷 Laptop Camera Streamer</h1>
            <div class="status">🟢 BROADCASTING LIVE</div>
            <p>Your laptop webcam is now streaming over the local network.</p>
            <p>Enter the following URL into your <strong>Raspberry Pi Settings Panel</strong>:</p>
            <div class="url-container">{{ stream_url }}</div>
            
            <div class="instructions">
                <strong>How to use this for your College Demo:</strong>
                <ol>
                    <li>Ensure both your Laptop and your Raspberry Pi are connected to the <strong>exact same Wi-Fi network</strong>.</li>
                    <li>Start the master program on your Raspberry Pi: <code>python app.py</code>.</li>
                    <li>Open the Pi's dashboard in a browser on any device.</li>
                    <li>Go to Settings, paste the URL above into the <strong>Video Source</strong> box, and click Save.</li>
                </ol>
            </div>
            
            <img class="video-preview" src="/video_feed" alt="Webcam Preview">
        </div>
    </body>
    </html>
    """
    return render_template_string(html, stream_url=STREAM_URL)

if __name__ == '__main__':
    print("==========================================================")
    print("      ELDERLY MONITOR - LAPTOP WEBCAM STREAMING SERVER    ")
    print("==========================================================")
    print(f"Local IP Address Found: {LOCAL_IP}")
    print(f"Streaming webcam feed at: {STREAM_URL}")
    print("==========================================================")
    print("Keep this terminal open during your presentation!")
    print("Press Ctrl+C to stop broadcasting.")
    
    # Automatically open local browser window to display streamer HUD
    import webbrowser
    webbrowser.open("http://127.0.0.1:8000")
    
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
