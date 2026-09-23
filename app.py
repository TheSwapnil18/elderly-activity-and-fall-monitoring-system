import cv2
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
import time
import math
import json
import os
import sys
import socket
import threading
import requests
import urllib.parse
import platform
from datetime import datetime
from flask import Flask, render_template_string, Response, jsonify, request

# --- Flask Server Initialization ---
app = Flask(__name__)

# Threading locks and global state
state_lock = threading.Lock()
frame_lock = threading.Lock()

latest_raw_frame = None
latest_processed_frame = None
system_status = {
    "state": "INITIALIZING",        # INITIALIZING, UPRIGHT, LYING DOWN, INACTIVE, FALL ALERT, NO PERSON, CAMERA OFFLINE
    "person_detected": False,
    "inactivity_timer": 0,
    "torso_angle": 0.0,
    "nose_speed": 0.0,
    "fps": 0.0,
    "last_movement_duration": 0
}

# --- Load and Save Configurations ---
CONFIG_FILE = "config.json"
ALERTS_FILE = "alerts.json"

DEFAULT_CONFIG = {
    "video_source": "0",
    "normal_inactivity_threshold": 15,
    "sleep_inactivity_threshold": 30,
    "movement_threshold_px": 10,
    "fall_velocity_threshold": 0.4,
    "angle_3d_threshold": 55,
    "telegram_bot_token": "8719214387:AAFlHVLbgmpIY7q0UGKqdMFkPGfWM5haX6M",
    "telegram_chat_id": "5776686318",
    "enable_telegram": True,
    "buzzer_enabled": True, 
    "buzzer_pin": 17
}

def load_config():
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        return DEFAULT_CONFIG
    try:
        with open(CONFIG_FILE, 'r') as f:
            data = json.load(f)
            # Ensure all keys from DEFAULT_CONFIG exist (backwards compatibility)
            for k, v in DEFAULT_CONFIG.items():
                if k not in data:
                    data[k] = v
            return data
    except Exception as e:
        print(f"Error reading config: {e}")
        return DEFAULT_CONFIG

def save_config(config):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

# Load active configuration
config = load_config()

# --- Alert Logging System ---
def log_alert(description, state):
    alert_event = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": description,
        "state": state
    }
    try:
        alerts = []
        if os.path.exists(ALERTS_FILE):
            with open(ALERTS_FILE, 'r') as f:
                try:
                    alerts = json.load(f)
                except Exception:
                    alerts = []
        alerts.insert(0, alert_event)  # Prepend new alerts so latest is first
        alerts = alerts[:100]          # Keep last 100 alerts
        with open(ALERTS_FILE, 'w') as f:
            json.dump(alerts, f, indent=4)
    except Exception as e:
        print(f"Error logging alert: {e}")

# --- Platform-Aware Hardware Alarms ---
# Initialize Buzzer if on Raspberry Pi
buzzer = None
if platform.system() != "Windows":
    try:
        from gpiozero import Buzzer as GpioBuzzer
        buzzer = GpioBuzzer(config.get("buzzer_pin", 17))
        print(f"--> Hardware initialized: GPIO Buzzer on pin {config.get('buzzer_pin', 17)} enabled.")
    except Exception as e:
        print(f"--> Raspberry Pi GPIO import failed or pin busy: {e}. Falling back to system bell.")

def play_sound(pattern="single"):
    if not config.get("buzzer_enabled", True):
        return
    
    if buzzer:
        # Raspberry Pi buzzer control
        try:
            if pattern == "single":
                buzzer.beep(on_time=0.8, off_time=0.2, n=1)
            elif pattern == "triple":
                buzzer.beep(on_time=0.25, off_time=0.15, n=3)
        except Exception as e:
            print(f"Buzzer play error: {e}")
    elif platform.system() == "Windows":
        # Windows speaker control
        import winsound
        try:
            if pattern == "single":
                winsound.Beep(1000, 800)
            elif pattern == "triple":
                winsound.Beep(2000, 250)
                winsound.Beep(2000, 250)
                winsound.Beep(2000, 250)
        except Exception as e:
            print(f"Windows Beep error: {e}")
    else:
        # Fallback terminal bell for other Linux/macOS systems
        if pattern == "single":
            print("\a", end="", flush=True)
        elif pattern == "triple":
            print("\a\a\a", end="", flush=True)

# --- Dispatch Telegram Alerts ---
last_telegram_time = 0

def send_telegram(message):
    global last_telegram_time
    if not config.get("enable_telegram", True):
        return
    token = config.get("telegram_bot_token")
    chat_id = config.get("telegram_chat_id")
    if not token or not chat_id or token == "YOUR_BOT_TOKEN" or chat_id == "YOUR_CHAT_ID":
        print("--> Telegram notification skipped: Invalid credentials.")
        return
    
    current_time = time.time()
    # Cooldown of 10 seconds between same alert dispatches to prevent spamming
    if current_time - last_telegram_time < 10:
        return
        
    def worker():
        global last_telegram_time
        try:
            encoded_message = urllib.parse.quote(message)
            url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={encoded_message}"
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                print("--> Telegram Alert Sent successfully!")
                last_telegram_time = time.time()
            else:
                print(f"--> Failed to send Telegram alert. Status code: {res.status_code}")
        except Exception as ex:
            print(f"--> Telegram Network Error: {ex}")
            
    threading.Thread(target=worker, daemon=True).start()

# --- 3D Skeletal Vector Calculation ---
def calculate_3d_angle(p1, p2):
    """
    Calculates the 3D angle of torso vector (p1 -> p2) relative to absolute vertical gravity line (Y axis).
    Returns degrees from vertical: 0 = straight up, 90 = flat horizontal.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    
    magnitude = math.sqrt(dx*dx + dy*dy + dz*dz)
    if magnitude == 0:
        return 0.0
    
    # Dot product with Y gravity unit vector (0, 1, 0)
    # Using absolute value of dy to calculate offset from vertical regardless of facing direction
    angle_rad = math.acos(min(abs(dy) / magnitude, 1.0))
    return math.degrees(angle_rad)

# --- MediaPipe Pose Connections ---
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), 
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), 
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23), 
    (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), 
    (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

def draw_skeleton(frame, landmarks, w, h):
    px_points = {}
    # Draw landmarks
    for i, lm in enumerate(landmarks):
        if lm.visibility > 0.4:
            px_points[i] = (int(lm.x * w), int(lm.y * h))
            cv2.circle(frame, px_points[i], 4, (0, 215, 255), -1)
            
    # Draw joint skeletons
    for connection in POSE_CONNECTIONS:
        start, end = connection
        if start in px_points and end in px_points:
            cv2.line(frame, px_points[start], px_points[end], (0, 255, 0), 2)
            
    return px_points

def draw_synthetic_skeleton(frame, x, y, w_box, h_box):
    """
    Renders an estimated, glowing 2D skeletal stick-figure inside the bounding box.
    Adapts dynamically to whether the person is standing (tall box) or lying down (wide box).
    """
    joints = {}
    is_upright = h_box > w_box
    
    if is_upright:
        # --- Upright Skeletal Estimation ---
        joints['head'] = (x + w_box // 2, y + int(h_box * 0.12))
        joints['neck'] = (x + w_box // 2, y + int(h_box * 0.22))
        
        shldr_offset = int(w_box * 0.3)
        joints['l_shoulder'] = (joints['neck'][0] - shldr_offset, joints['neck'][1] + int(h_box * 0.02))
        joints['r_shoulder'] = (joints['neck'][0] + shldr_offset, joints['neck'][1] + int(h_box * 0.02))
        
        joints['pelvis'] = (x + w_box // 2, y + int(h_box * 0.6))
        hip_offset = int(w_box * 0.22)
        joints['l_hip'] = (joints['pelvis'][0] - hip_offset, joints['pelvis'][1])
        joints['r_hip'] = (joints['pelvis'][0] + hip_offset, joints['pelvis'][1])
        
        joints['l_elbow'] = (joints['l_shoulder'][0] - int(w_box * 0.08), joints['l_shoulder'][1] + int(h_box * 0.16))
        joints['r_elbow'] = (joints['r_shoulder'][0] + int(w_box * 0.08), joints['r_shoulder'][1] + int(h_box * 0.16))
        
        joints['l_wrist'] = (joints['l_elbow'][0], joints['l_elbow'][1] + int(h_box * 0.15))
        joints['r_wrist'] = (joints['r_elbow'][0], joints['r_elbow'][1] + int(h_box * 0.15))
        
        joints['l_knee'] = (joints['l_hip'][0] - int(w_box * 0.05), joints['l_hip'][1] + int(h_box * 0.18))
        joints['r_knee'] = (joints['r_hip'][0] + int(w_box * 0.05), joints['r_hip'][1] + int(h_box * 0.18))
        
        joints['l_ankle'] = (x + int(w_box * 0.15), y + h_box - int(h_box * 0.05))
        joints['r_ankle'] = (x + w_box - int(w_box * 0.15), y + h_box - int(h_box * 0.05))
    else:
        # --- Horizontal / Lying Down Skeletal Estimation ---
        joints['head'] = (x + int(w_box * 0.12), y + h_box // 2)
        joints['neck'] = (x + int(w_box * 0.22), y + h_box // 2)
        
        shldr_offset = int(h_box * 0.3)
        joints['l_shoulder'] = (joints['neck'][0] + int(w_box * 0.02), joints['neck'][1] - shldr_offset)
        joints['r_shoulder'] = (joints['neck'][0] + int(w_box * 0.02), joints['neck'][1] + shldr_offset)
        
        joints['pelvis'] = (x + int(w_box * 0.6), y + h_box // 2)
        hip_offset = int(h_box * 0.22)
        joints['l_hip'] = (joints['pelvis'][0], joints['pelvis'][1] - hip_offset)
        joints['r_hip'] = (joints['pelvis'][0], joints['pelvis'][1] + hip_offset)
        
        joints['l_elbow'] = (joints['l_shoulder'][0] + int(w_box * 0.15), joints['l_shoulder'][1] - int(h_box * 0.05))
        joints['r_elbow'] = (joints['r_shoulder'][0] + int(w_box * 0.15), joints['r_shoulder'][1] + int(h_box * 0.05))
        
        joints['l_wrist'] = (joints['l_elbow'][0] + int(w_box * 0.12), joints['l_elbow'][1])
        joints['r_wrist'] = (joints['r_elbow'][0] + int(w_box * 0.12), joints['r_elbow'][1])
        
        joints['l_knee'] = (joints['l_hip'][0] + int(w_box * 0.16), joints['l_hip'][1] - int(h_box * 0.05))
        joints['r_knee'] = (joints['r_hip'][0] + int(w_box * 0.16), joints['r_hip'][1] + int(h_box * 0.05))
        
        joints['l_ankle'] = (x + w_box - int(w_box * 0.05), y + int(h_box * 0.2))
        joints['r_ankle'] = (x + w_box - int(w_box * 0.05), y + h_box - int(h_box * 0.2))

    # Draw Bones (thick green lines)
    connections = [
        ('head', 'neck'),
        ('neck', 'l_shoulder'), ('neck', 'r_shoulder'),
        ('neck', 'pelvis'),
        ('pelvis', 'l_hip'), ('pelvis', 'r_hip'),
        ('l_shoulder', 'l_elbow'), ('l_elbow', 'l_wrist'),
        ('r_shoulder', 'r_elbow'), ('r_elbow', 'r_wrist'),
        ('l_hip', 'l_knee'), ('l_knee', 'l_ankle'),
        ('r_hip', 'r_knee'), ('r_knee', 'r_ankle')
    ]
    
    for start, end in connections:
        if start in joints and end in joints:
            cv2.line(frame, joints[start], joints[end], (0, 255, 0), 2)
            
    # Draw Joints (glowing cyan circles)
    for name, pt in joints.items():
        cv2.circle(frame, pt, 4, (0, 215, 255), -1)

# --- Real-Time Non-Blocking Frame Grabber Thread ---
class RealTimeVideoCapture:
    def __init__(self, source, open_func):
        self.source = source
        self.open_func = open_func
        self.cap = self.open_func(self.source)
        self.frame = None
        self.ret = False
        self.running = True
        self.lock = threading.Lock()
        
        # Launch non-blocking background thread to grab frames and flush the driver queue.
        # This prevents buffering delays on both network streams and local loopback/USB cameras.
        print(f"--> [RealTimeVideoCapture] Launching non-blocking background thread for video source: {source}")
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.ret = ret
                        self.frame = frame.copy()
                    # No sleep here! cap.read() blocks naturally in the driver, 
                    # ensuring we grab frames as fast as they arrive without delay.
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.1)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def isOpened(self):
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()

# --- Master AI Background Worker ---
def ai_processing_thread():
    global latest_raw_frame, latest_processed_frame, system_status, config
    
    detector = None
    if HAS_MEDIAPIPE:
        print("--> Initializing MediaPipe Pose Landmarker...")
        try:
            base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False)
            detector = vision.PoseLandmarker.create_from_options(options)
            print("--> MediaPipe Pose Landmarker loaded successfully!")
        except Exception as e:
            print(f"--> MediaPipe creation failed: {e}. Falling back to OpenCV Contour engine.")
            detector = None
    else:
        print("--> MediaPipe library not found. Launching Traditional OpenCV Contour Tracking Engine!")

    # Keep track of active video source to handle live swaps
    active_source = config.get("video_source", "0")
    cap = None
    
    def open_camera(source):
        print(f"--> Connecting to video source: {source}...")
        try:
            # Parse integer for webcam, or string for network stream
            if source.isdigit():
                src = int(source)
                # On Windows, DirectShow helps open cameras quickly
                if platform.system() == "Windows":
                    c = cv2.VideoCapture(src, cv2.CAP_DSHOW)
                else:
                    c = cv2.VideoCapture(src)
            else:
                c = cv2.VideoCapture(source)
                
            if c.isOpened():
                # Only set buffer size for network streams to avoid breaking local/loopback devices
                source_str = str(source).lower().strip()
                is_network = any(source_str.startswith(p) for p in ["http://", "https://", "rtsp://", "rtp://", "udp://"])
                if is_network:
                    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    print(f"--> Set CAP_PROP_BUFFERSIZE to 1 for network stream.")
                print(f"--> Successfully connected to camera source: {source}")
                return c
        except Exception as ex:
            print(f"Error opening camera source {source}: {ex}")
        return None

    cap = RealTimeVideoCapture(active_source, open_camera)

    # Warmup and Ghost Suppression variables
    warmup_until = time.time() + 3.0
    static_check_history = []

    # Inactivity Tracking variables
    last_movement_time = time.time()
    inactivity_reported = False
    fall_alert_reported = False
    fall_detected_time = 0
    
    # Mathematical tracking variables
    last_px_points = {}
    nose_history = []  # List of (timestamp, nose_y)
    
    # OpenCV Traditional CV Tracking variables
    backSub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=True)
    last_centroid = None
    centroid_history = []  # List of (timestamp, centroid_y)

    # State tracking variables for Temporal Smoothing (State Hysteresis)
    state_history = []
    HYSTERESIS_FRAMES = 5 # Must remain in state for 5 consecutive frames to switch

    # FPS counter
    fps_counter = 0
    fps_start_time = time.time()

    while True:
        # 1. Handle live camera configuration changes
        current_config_source = config.get("video_source", "0")
        if current_config_source != active_source:
            print(f"--> Config changed! Re-routing video source {active_source} -> {current_config_source}")
            if cap:
                cap.release()
            active_source = current_config_source
            cap = RealTimeVideoCapture(active_source, open_camera)
            
            # Reset traditional background subtractor to prevent ghost contour alerts
            backSub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=True)
            last_centroid = None
            centroid_history.clear()
            static_check_history.clear()
            warmup_until = time.time() + 3.0

        # 2. Capture frame and handle offline cameras
        if cap is None or not cap.isOpened():
            with state_lock:
                system_status["state"] = "CAMERA OFFLINE"
                system_status["person_detected"] = False
            time.sleep(3.0)
            cap = RealTimeVideoCapture(active_source, open_camera)
            static_check_history.clear()
            warmup_until = time.time() + 3.0
            continue

        ret, frame = cap.read()
        if not ret:
            print("--> Camera failed to grab frame. Attempting recovery...")
            with state_lock:
                system_status["state"] = "CAMERA OFFLINE"
            time.sleep(2.0)
            static_check_history.clear()
            warmup_until = time.time() + 3.0
            continue

        # Keep original frame for streaming in background
        with frame_lock:
            latest_raw_frame = frame.copy()

        # Build processing details
        frame = cv2.flip(frame, 1) # Flip horizontally for natural mirrors
        h, w, _ = frame.shape
        current_time = time.time()
        
        # Calculate Frame Rate
        fps_counter += 1
        elapsed_fps_time = current_time - fps_start_time
        if elapsed_fps_time >= 1.0:
            with state_lock:
                system_status["fps"] = round(fps_counter / elapsed_fps_time, 1)
            fps_counter = 0
            fps_start_time = current_time

        # 3. Process frame based on Dual-Mode availability
        person_detected = False
        is_currently_falling = False
        frame_state = "NO PERSON"
        current_torso_angle = 0.0
        current_nose_speed = 0.0

        if HAS_MEDIAPIPE and detector is not None:
            # --- ENGINE: MEDIAPIPE (3D SKELETAL AI) ---
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = detector.detect(mp_image)

                if len(detection_result.pose_landmarks) > 0:
                    person_detected = True
                    landmarks = detection_result.pose_landmarks[0]
                    
                    # Draw skeletal graphics on overlay frame
                    px_points = draw_skeleton(frame, landmarks, w, h)

                    # --- A. NOISE-FILTERED MICRO-MOVEMENT DETECTION ---
                    moved = False
                    if last_px_points:
                        total_shift = 0.0
                        points_checked = 0
                        for idx in px_points:
                            if idx in last_px_points and landmarks[idx].visibility > 0.5:
                                dx = px_points[idx][0] - last_px_points[idx][0]
                                dy = px_points[idx][1] - last_px_points[idx][1]
                                dist = math.sqrt(dx*dx + dy*dy)
                                
                                # Apply a noise threshold (ignore shifts less than 1.5 pixels)
                                if dist > 1.5:
                                    total_shift += dist
                                    points_checked += 1
                        
                        threshold = config.get("movement_threshold_px", 10)
                        if points_checked > 0 and (total_shift / points_checked) > threshold:
                            moved = True

                    last_px_points = px_points

                    if moved:
                        last_movement_time = current_time
                        inactivity_reported = False

                    # --- B. OMNI-DIRECTIONAL 3D TORSO PITCH ---
                    torso_is_horizontal = False
                    if 11 in px_points and 12 in px_points and 23 in px_points and 24 in px_points:
                        l_shldr, r_shldr = landmarks[11], landmarks[12]
                        l_hip, r_hip = landmarks[23], landmarks[24]
                        
                        mid_shldr_3d = (
                            (l_shldr.x + r_shldr.x) / 2.0,
                            (l_shldr.y + r_shldr.y) / 2.0,
                            (l_shldr.z + r_shldr.z) / 2.0
                        )
                        mid_hip_3d = (
                            (l_hip.x + r_hip.x) / 2.0,
                            (l_hip.y + r_hip.y) / 2.0,
                            (l_hip.z + r_hip.z) / 2.0
                        )
                        
                        current_torso_angle = calculate_3d_angle(mid_hip_3d, mid_shldr_3d)
                        
                        angle_threshold = config.get("angle_3d_threshold", 55)
                        if current_torso_angle > angle_threshold:
                            torso_is_horizontal = True
                            frame_state = "LYING DOWN"
                        else:
                            frame_state = "UPRIGHT"

                    # --- C. STABLE VELOCITY & HEAD-PROXIMITY ALGORITHMS ---
                    if 0 in px_points:
                        nose_y_norm = landmarks[0].y # Y coordinate from 0.0 (top) to 1.0 (bottom)
                        nose_history.append((current_time, nose_y_norm))
                        
                        # Maintain sliding history window of 1.5 seconds
                        nose_history = [(t, y) for (t, y) in nose_history if current_time - t <= 1.5]
                        
                        # Compute stabilized speed over a sliding 0.4s temporal step (resilient to jitter)
                        target_lookback = current_time - 0.4
                        closest_past = None
                        closest_diff = 999.0
                        
                        for t_hist, y_hist in nose_history:
                            diff = abs(t_hist - target_lookback)
                            if diff < closest_diff:
                                closest_diff = diff
                                closest_past = (t_hist, y_hist)
                        
                        if closest_past and closest_diff < 0.15:
                            t_past, y_past = closest_past
                            dt = current_time - t_past
                            if dt > 0.1:
                                current_nose_speed = (nose_y_norm - y_past) / dt

                        # Check proximity constraints: Head lower than hips (Y decreases upwards)
                        head_is_low = False
                        if 23 in px_points and 24 in px_points:
                            avg_hip_y_norm = (landmarks[23].y + landmarks[24].y) / 2.0
                            # Head coordinate (nose) is close to hips, or lower (larger y coordinate)
                            if nose_y_norm >= (avg_hip_y_norm - 0.12):
                                head_is_low = True

                        # Evaluate final fall trigger
                        vel_threshold = config.get("fall_velocity_threshold", 0.4)
                        if torso_is_horizontal and current_nose_speed > vel_threshold and head_is_low:
                            is_currently_falling = True

                    # Draw engine HUD banner
                    cv2.putText(frame, "ENGINE: MEDIAPIPE (3D)", (w - 240, 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    
                    # Draw visual metrics on the feed
                    cv2.putText(frame, f"3D Torso Angle: {int(current_torso_angle)} deg", (20, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if not torso_is_horizontal else (0, 165, 255), 2)
                    cv2.putText(frame, f"Nose Drop Speed: {max(0.0, current_nose_speed):.2f} norm/s", (20, 110), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                else:
                    # Nobody in frame
                    last_movement_time = current_time
                    nose_history.clear()
                    frame_state = "NO PERSON"
            except Exception as e:
                print(f"MediaPipe runtime error: {e}. Attempting OpenCV fallback...")
                detector = None  # Force traditional fallback on next frame
        
        if detector is None:
            # --- ENGINE: TRADITIONAL OPENCV CONTOUR TRACKING (FALLBACK) ---
            is_warming_up = current_time < warmup_until
            
            # 1. Apply background subtraction & morphological cleanups
            # Use high learning rate (0.05) during warmup to quickly build background model
            l_rate = 0.05 if is_warming_up else -1
            fgMask = backSub.apply(frame, learningRate=l_rate)
            _, fgMask = cv2.threshold(fgMask, 250, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
            
            # 2. Extract contours
            contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            large_contour = None
            max_area = 0
            for c in contours:
                area = cv2.contourArea(c)
                # Enforce a larger minimum area (12000 pixels, ~4% of a 640x480 frame)
                # to prevent minor light shifts, camera auto-exposure noise, shadows, or pets from triggering.
                if area > 12000:
                    x_c, y_c, w_c, h_c = cv2.boundingRect(c)
                    aspect_ratio_c = w_c / max(1.0, h_c)
                    extent_c = area / float(w_c * h_c)
                    
                    # --- SMART HUMAN FILTERING ---
                    # 1. Ignore top-of-frame square movements (like ceiling fans)
                    is_ceiling_fan = (y_c < h * 0.25) and (0.75 < aspect_ratio_c < 1.25)
                    # 2. Ignore highly horizontal wide-stretching thin bands (like moving shadows or light reflections)
                    is_shadow_band = (h_c < 30) or (aspect_ratio_c > 4.0)
                    # 3. Enforce solidity: humans are solid bodies, not hollow/fragmented borders (extent > 0.22)
                    is_solid_body = extent_c > 0.22
                    # 4. Enforce height/width ratio: humans are taller than wide (standing) or wider than tall (lying),
                    # not perfectly square unless it is a massive moving shape (area > 22000)
                    is_human_proportion = (aspect_ratio_c < 0.85) or (aspect_ratio_c > 1.2) or (area > 22000)
                    
                    if not is_ceiling_fan and not is_shadow_band and is_solid_body and is_human_proportion:
                        if area > max_area:
                            max_area = area
                            large_contour = c
            
            if large_contour is not None and not is_warming_up:
                x, y, w_box, h_box = cv2.boundingRect(large_contour)
                cx = x + w_box // 2
                cy = y + h_box // 2
                
                # --- STATIC GHOST SUPPRESSION CHECK ---
                static_check_history.append((cx, cy, current_time))
                # Keep last 4 seconds of coordinates
                static_check_history = [pt for pt in static_check_history if current_time - pt[2] <= 4.0]
                
                is_static_ghost = False
                if len(static_check_history) > 30 and (current_time - static_check_history[0][2] >= 3.0):
                    xs = [pt[0] for pt in static_check_history]
                    ys = [pt[1] for pt in static_check_history]
                    range_x = max(xs) - min(xs)
                    range_y = max(ys) - min(ys)
                    
                    # If the object has moved by less than 2.0 pixels in 3.0+ seconds, it is a static ghost!
                    if range_x < 2.0 and range_y < 2.0:
                        is_static_ghost = True
                
                if is_static_ghost:
                    # Suppress detection
                    person_detected = False
                    large_contour = None
                    print("👻 STATIC GHOST DETECTED: Auto-resetting background subtractor to absorb static object.")
                    backSub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=True)
                    static_check_history.clear()
                else:
                    person_detected = True
                cx = x + w_box // 2
                cy = y + h_box // 2
                
                # --- A. NOISE-FILTERED MICRO-MOVEMENT DETECTION ---
                moved = False
                if last_centroid is not None:
                    dx = cx - last_centroid[0]
                    dy = cy - last_centroid[1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    
                    threshold = config.get("movement_threshold_px", 10)
                    if dist > threshold:
                        moved = True
                
                last_centroid = (cx, cy)
                
                if moved:
                    last_movement_time = current_time
                    inactivity_reported = False
                
                # --- B. ESTIMATED TORSO ANGLE ---
                # Aspect ratio of the box (width / height)
                aspect_ratio = w_box / max(1.0, h_box)
                # Map aspect ratio from 0.45 (upright) to 1.40 (flat horizontal) to a 0 - 90 deg range
                current_torso_angle = min(90.0, max(0.0, ((aspect_ratio - 0.45) / 0.95) * 90.0))
                
                torso_is_horizontal = False
                angle_threshold = config.get("angle_3d_threshold", 55)
                if current_torso_angle > angle_threshold:
                    torso_is_horizontal = True
                    frame_state = "LYING DOWN"
                else:
                    frame_state = "UPRIGHT"
                
                # --- C. STABLE VELOCITY CALCULATION ---
                centroid_y_norm = cy / max(1.0, h)
                centroid_history.append((current_time, centroid_y_norm))
                centroid_history = [(t, val) for (t, val) in centroid_history if current_time - t <= 1.5]
                
                # Compute stabilized speed over sliding 0.4s temporal step
                target_lookback = current_time - 0.4
                closest_past = None
                closest_diff = 999.0
                for t_hist, y_hist in centroid_history:
                    diff = abs(t_hist - target_lookback)
                    if diff < closest_diff:
                        closest_diff = diff
                        closest_past = (t_hist, y_hist)
                
                if closest_past and closest_diff < 0.15:
                    t_past, y_past = closest_past
                    dt = current_time - t_past
                    if dt > 0.1:
                        current_nose_speed = (centroid_y_norm - y_past) / dt
                
                # Evaluate final fall trigger: torso flat and centroid velocity high
                vel_threshold = config.get("fall_velocity_threshold", 0.4)
                if torso_is_horizontal and current_nose_speed > vel_threshold:
                    is_currently_falling = True
                
                # --- D. DRAW VISUAL HUD OVERLAYS & SYNTHETIC SKELETON ---
                # Glassmorphic bounding box
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 215, 255), 2)
                # Centroid marker (Center of Gravity)
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                
                # Draw estimated skeletal stick figure overlay
                draw_synthetic_skeleton(frame, x, y, w_box, h_box)
                
                # Label subject
                cv2.putText(frame, "TRACKED HUMAN SUBJECT", (x, y - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1, cv2.LINE_AA)
                
                # Engine HUD banner
                cv2.putText(frame, "ENGINE: OPENCV (SYNTHETIC SKELETON)", (w - 340, 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
                
                # Visual metrics text
                cv2.putText(frame, f"Est Torso Angle: {int(current_torso_angle)} deg", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if not torso_is_horizontal else (0, 165, 255), 2)
                cv2.putText(frame, f"Centroid Drop Speed: {max(0.0, current_nose_speed):.2f} norm/s", (20, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            else:
                # Nobody in frame
                last_movement_time = current_time
                centroid_history.clear()
                frame_state = "NO PERSON"

        # --- D. FRAME-BASED STATE HYSTERESIS (SMOOTHING) ---

        state_history.append(frame_state)
        if len(state_history) > HYSTERESIS_FRAMES:
            state_history.pop(0)
            
        # Determine smoothed consensus state
        consensus_state = system_status["state"]
        if len(state_history) == HYSTERESIS_FRAMES and len(set(state_history)) == 1:
            consensus_state = state_history[0]

        # Inactivity calculations
        time_since_movement = current_time - last_movement_time
        inactivity_limit = config.get("sleep_inactivity_threshold" if consensus_state == "LYING DOWN" else "normal_inactivity_threshold", 15)

        # Handle Alerts
        if person_detected:
            # 1. Emergency Fall alert
            if is_currently_falling:
                if current_time - fall_detected_time > 10:  # Prevent double alert spam
                    consensus_state = "FALL ALERT"
                    fall_detected_time = current_time
                    fall_alert_reported = True
                    
                    print("🚨 MASTER SYSTEM ALERT: Fall Detected!")
                    log_alert("Emergency: Sudden fall detected by AI tracking system", "FALL ALERT")
                    send_telegram("🚨 EMERGENCY: A sudden fall has been detected by the AI monitoring system!")
                    
                    # Fire local buzzer thread
                    threading.Thread(target=play_sound, args=("triple",), daemon=True).start()
            
            # Keep fall message flashing on screen for 5 seconds
            if fall_alert_reported and (current_time - fall_detected_time < 5.0):
                consensus_state = "FALL ALERT"
                cv2.putText(frame, "EMERGENCY: FALL DETECTED!", (20, 220), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
            else:
                fall_alert_reported = False

            # 2. Prolonged Inactivity alerts
            if time_since_movement > inactivity_limit:
                if not inactivity_reported:
                    consensus_state = "INACTIVE"
                    print(f"⚠️ MASTER SYSTEM WARNING: No movement for {int(time_since_movement)} seconds.")
                    log_alert(f"Warning: No movement detected for {inactivity_limit} seconds (State: {consensus_state})", "INACTIVE")
                    send_telegram(f"⚠️ Elderly Monitor: No movement detected for {inactivity_limit} seconds!")
                    
                    threading.Thread(target=play_sound, args=("single",), daemon=True).start()
                    inactivity_reported = True
                
                # If inactivity state is active and reported, override standard state display
                if inactivity_reported:
                    consensus_state = "INACTIVE"

            # Draw inactivity metrics
            cv2.putText(frame, f"Inactivity: {int(time_since_movement)}s / {inactivity_limit}s", (20, 140), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            consensus_state = "NO PERSON"
            cv2.putText(frame, "NO PERSON DETECTED", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Draw HUD State box on stream
        if consensus_state == "UPRIGHT":
            hud_color = (0, 255, 0)
        elif consensus_state == "LYING DOWN":
            hud_color = (0, 165, 255)
        elif consensus_state == "INACTIVE":
            hud_color = (0, 69, 255)
        elif consensus_state == "FALL ALERT":
            hud_color = (0, 0, 255)
        else:
            hud_color = (128, 128, 128)
            
        cv2.putText(frame, f"State: {consensus_state}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2)

        # Write frame to global for server stream
        ret_png, jpeg_buffer = cv2.imencode('.jpg', frame)
        if ret_png:
            with frame_lock:
                latest_processed_frame = jpeg_buffer.tobytes()

        # Update global metrics
        with state_lock:
            system_status["state"] = consensus_state
            system_status["person_detected"] = person_detected
            system_status["inactivity_timer"] = int(time_since_movement)
            system_status["torso_angle"] = round(current_torso_angle, 1)
            system_status["nose_speed"] = round(max(0.0, current_nose_speed), 2)
            system_status["last_movement_duration"] = int(inactivity_limit)

        time.sleep(0.03) # Cap background AI processing loop around 30 FPS to save CPU

    if cap:
        cap.release()

# Start background AI processing thread immediately
threading.Thread(target=ai_processing_thread, daemon=True).start()


# --- Flask Route Handlers ---

@app.route('/')
def dashboard():
    """Beautiful, responsive dark-themed glassmorphic Web Dashboard."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Elderly Activity Monitor - Master Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-color: #0b0f19;
                --card-bg: rgba(22, 29, 49, 0.7);
                --card-border: rgba(255, 255, 255, 0.08);
                --primary: #6366f1;
                --primary-glow: rgba(99, 102, 241, 0.35);
                --success: #10b981;
                --success-glow: rgba(16, 185, 129, 0.25);
                --warning: #f59e0b;
                --warning-glow: rgba(245, 158, 11, 0.25);
                --danger: #ef4444;
                --danger-glow: rgba(239, 68, 68, 0.35);
                --text: #f1f5f9;
                --text-muted: #94a3b8;
            }

            * {
                box-sizing: border-box;
                transition: all 0.2s ease-in-out;
            }

            body {
                font-family: 'Inter', sans-serif;
                background-color: var(--bg-color);
                color: var(--text);
                margin: 0;
                padding: 24px;
                min-height: 100vh;
                background-image: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.05) 0%, transparent 40%),
                                  radial-gradient(circle at 90% 80%, rgba(239, 68, 68, 0.04) 0%, transparent 40%);
            }

            .container {
                max-width: 1400px;
                margin: 0 auto;
            }

            /* Header Section */
            header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 24px;
                background: var(--card-bg);
                backdrop-filter: blur(12px);
                border: 1px solid var(--card-border);
                border-radius: 16px;
                padding: 16px 28px;
                box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
            }

            header h1 {
                margin: 0;
                font-size: 24px;
                font-weight: 700;
                background: linear-gradient(135deg, #a5b4fc, #f87171);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .connection-badge {
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 14px;
                font-weight: 600;
                padding: 6px 14px;
                border-radius: 20px;
                background: rgba(255,255,255,0.05);
            }

            .dot {
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background: #6b7280;
            }

            /* Main Layout Grid */
            .grid {
                display: grid;
                grid-template-columns: 1fr 400px;
                gap: 24px;
            }

            @media(max-width: 1024px) {
                .grid {
                    grid-template-columns: 1fr;
                }
            }

            /* Left: Video and Real-time Cards */
            .main-content {
                display: flex;
                flex-direction: column;
                gap: 24px;
            }

            .video-panel {
                background: var(--card-bg);
                backdrop-filter: blur(12px);
                border: 1px solid var(--card-border);
                border-radius: 20px;
                padding: 16px;
                display: flex;
                flex-direction: column;
                align-items: center;
                box-shadow: 0 20px 40px -15px rgba(0,0,0,0.5);
            }

            .video-container {
                width: 100%;
                aspect-ratio: 16/9;
                border-radius: 12px;
                overflow: hidden;
                background: #020617;
                display: flex;
                align-items: center;
                justify-content: center;
                border: 1px solid rgba(255,255,255,0.05);
            }

            .video-feed {
                width: 100%;
                height: 100%;
                object-fit: contain;
            }

            /* Metric Cards Row */
            .metrics-row {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 16px;
            }

            @media(max-width: 640px) {
                .metrics-row {
                    grid-template-columns: repeat(2, 1fr);
                }
            }

            .metric-card {
                background: var(--card-bg);
                backdrop-filter: blur(12px);
                border: 1px solid var(--card-border);
                border-radius: 16px;
                padding: 16px;
                text-align: center;
            }

            .metric-label {
                font-size: 12px;
                color: var(--text-muted);
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 8px;
                font-weight: 600;
            }

            .metric-value {
                font-size: 26px;
                font-weight: 700;
                color: var(--text);
            }

            /* Right Pane: Settings & Event Logger */
            .side-content {
                display: flex;
                flex-direction: column;
                gap: 24px;
            }

            .panel {
                background: var(--card-bg);
                backdrop-filter: blur(12px);
                border: 1px solid var(--card-border);
                border-radius: 20px;
                padding: 24px;
                box-shadow: 0 15px 35px -10px rgba(0,0,0,0.5);
            }

            .panel h2 {
                margin: 0 0 16px 0;
                font-size: 18px;
                font-weight: 600;
                border-bottom: 1px solid rgba(255,255,255,0.06);
                padding-bottom: 10px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            /* Form Elements */
            .form-group {
                margin-bottom: 16px;
            }

            .form-group label {
                display: block;
                font-size: 13px;
                font-weight: 600;
                color: var(--text-muted);
                margin-bottom: 6px;
            }

            .form-group input, .form-group select {
                width: 100%;
                background: rgba(15, 23, 42, 0.6);
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 8px;
                padding: 10px 14px;
                color: var(--text);
                font-size: 14px;
            }

            .form-group input:focus, .form-group select:focus {
                outline: none;
                border-color: var(--primary);
                box-shadow: 0 0 0 3px var(--primary-glow);
            }

            .checkbox-group {
                display: flex;
                align-items: center;
                gap: 10px;
                margin: 16px 0;
            }

            .checkbox-group input {
                width: 18px;
                height: 18px;
                cursor: pointer;
            }

            .btn {
                background: var(--primary);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 20px;
                font-weight: 600;
                cursor: pointer;
                width: 100%;
                font-size: 14px;
            }

            .btn:hover {
                box-shadow: 0 0 15px var(--primary-glow);
                filter: brightness(1.1);
            }

            .btn-outline {
                background: transparent;
                border: 1px solid rgba(255,255,255,0.15);
                color: var(--text);
                margin-top: 8px;
            }

            .btn-outline:hover {
                background: rgba(255,255,255,0.05);
            }

            /* Alert Logger List */
            .log-list {
                max-height: 280px;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 10px;
                padding-right: 4px;
            }

            .log-list::-webkit-scrollbar {
                width: 6px;
            }
            .log-list::-webkit-scrollbar-thumb {
                background: rgba(255,255,255,0.1);
                border-radius: 10px;
            }

            .log-item {
                background: rgba(15, 23, 42, 0.4);
                border-left: 4px solid #6b7280;
                padding: 10px 14px;
                border-radius: 4px 8px 8px 4px;
                font-size: 13px;
            }

            .log-item.inactive {
                border-left-color: var(--warning);
            }

            .log-item.fall {
                border-left-color: var(--danger);
                background: rgba(239, 68, 68, 0.05);
            }

            .log-time {
                font-size: 11px;
                color: var(--text-muted);
                margin-bottom: 2px;
                font-weight: 600;
            }

            /* Status Theme Colors */
            .state-bg-initializing { background: rgba(107, 114, 128, 0.1); color: #9ca3af; border: 1px solid rgba(107, 114, 128, 0.2); }
            .state-bg-upright { background: rgba(16, 185, 129, 0.1); color: var(--success); border: 1px solid rgba(16, 185, 129, 0.2); box-shadow: 0 0 15px var(--success-glow); }
            .state-bg-lying_down { background: rgba(245, 158, 11, 0.1); color: var(--warning); border: 1px solid rgba(245, 158, 11, 0.2); box-shadow: 0 0 15px var(--warning-glow); }
            .state-bg-inactive { background: rgba(245, 158, 11, 0.15); color: var(--warning); border: 1px solid rgba(245, 158, 11, 0.35); box-shadow: 0 0 20px var(--warning-glow); animation: pulse 2s infinite; }
            .state-bg-fall_alert { background: rgba(239, 68, 68, 0.15); color: var(--danger); border: 1px solid rgba(239, 68, 68, 0.35); box-shadow: 0 0 25px var(--danger-glow); animation: flash 1s infinite; }
            .state-bg-no_person { background: rgba(255,255,255,0.05); color: var(--text-muted); border: 1px solid rgba(255,255,255,0.1); }
            .state-bg-camera_offline { background: rgba(239, 68, 68, 0.1); color: var(--danger); border: 1px solid rgba(239, 68, 68, 0.2); animation: pulse 2.5s infinite; }

            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.6; }
            }

            @keyframes flash {
                0%, 100% { background: rgba(239, 68, 68, 0.15); box-shadow: 0 0 15px var(--danger-glow); }
                50% { background: rgba(239, 68, 68, 0.35); box-shadow: 0 0 35px var(--danger); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Header Nav -->
            <header>
                <div>
                    <h1>Elderly Activity Monitor</h1>
                    <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">Edge AI Skeletal Safety Platform</div>
                </div>
                <div class="connection-badge">
                    <div id="status-dot" class="dot"></div>
                    <span id="header-state-txt">LOADING SYSTEM</span>
                </div>
            </header>

            <!-- Grid Layout -->
            <div class="grid">
                <!-- Left Pane -->
                <div class="main-content">
                    <!-- Video Stream Card -->
                    <div class="video-panel">
                        <div class="video-container">
                            <!-- Stream directly from Flask stream router -->
                            <img id="stream-img" class="video-feed" src="/video_feed" alt="Master Live Camera Feed">
                        </div>
                    </div>

                    <!-- Metrics Row -->
                    <div class="metrics-row">
                        <div class="metric-card">
                            <div class="metric-label">3D Torso Angle</div>
                            <div id="metric-angle" class="metric-value">0°</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Nose Speed</div>
                            <div id="metric-speed" class="metric-value">0.00</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Inactivity Clock</div>
                            <div id="metric-inactivity" class="metric-value">0s</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">System FPS</div>
                            <div id="metric-fps" class="metric-value">0.0</div>
                        </div>
                    </div>
                </div>

                <!-- Right Pane -->
                <div class="side-content">
                    <!-- Dynamic Settings Panel -->
                    <div class="panel">
                        <h2>⚙️ Configuration Settings</h2>
                        <form id="settings-form">
                            <div class="form-group">
                                <label for="video_source">Video Source (Local index or Stream URL)</label>
                                <input type="text" id="video_source" name="video_source">
                            </div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                                <div class="form-group">
                                    <label for="normal_inactivity_threshold">Awake Inactive (s)</label>
                                    <input type="number" id="normal_inactivity_threshold" name="normal_inactivity_threshold">
                                </div>
                                <div class="form-group">
                                    <label for="sleep_inactivity_threshold">Sleep Inactive (s)</label>
                                    <input type="number" id="sleep_inactivity_threshold" name="sleep_inactivity_threshold">
                                </div>
                            </div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                                <div class="form-group">
                                    <label for="fall_velocity_threshold">Fall Speed Sens.</label>
                                    <input type="number" step="0.05" id="fall_velocity_threshold" name="fall_velocity_threshold">
                                </div>
                                <div class="form-group">
                                    <label for="angle_3d_threshold">Torso Fall Angle (°)</label>
                                    <input type="number" id="angle_3d_threshold" name="angle_3d_threshold">
                                </div>
                            </div>

                            <div style="border-top: 1px solid rgba(255,255,255,0.06); margin: 12px 0; padding-top: 10px;"></div>

                            <div class="form-group">
                                <label for="telegram_bot_token">Telegram Bot Token</label>
                                <input type="password" id="telegram_bot_token" name="telegram_bot_token">
                            </div>
                            <div class="form-group">
                                <label for="telegram_chat_id">Telegram Chat ID</label>
                                <input type="text" id="telegram_chat_id" name="telegram_chat_id">
                            </div>

                            <div class="checkbox-group">
                                <input type="checkbox" id="enable_telegram" name="enable_telegram">
                                <label for="enable_telegram" style="color: var(--text); font-size: 13px; font-weight: 600; cursor: pointer;">Enable Telegram Warnings</label>
                            </div>
                            <div class="checkbox-group">
                                <input type="checkbox" id="buzzer_enabled" name="buzzer_enabled">
                                <label for="buzzer_enabled" style="color: var(--text); font-size: 13px; font-weight: 600; cursor: pointer;">Enable Local Alarm Sound</label>
                            </div>

                            <button type="submit" class="btn">💾 Save Configuration</button>
                        </form>
                        <button id="test-alert-btn" class="btn btn-outline">🚨 Trigger Test Alert</button>
                    </div>

                    <!-- Historical Alerts Panel -->
                    <div class="panel">
                        <h2>
                            <span>📋 Historical Log Stream</span>
                            <button id="clear-logs-btn" style="background: transparent; border: none; color: var(--danger); font-size: 12px; cursor: pointer; font-weight: 600;">Clear All</button>
                        </h2>
                        <div id="log-list" class="log-list">
                            <!-- Logs dynamically loaded -->
                            <div style="color: var(--text-muted); text-align: center; padding: 20px; font-size: 13px;">No historical events recorded.</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Sync and update logic -->
        <script>
            // Elements
            const statusDot = document.getElementById("status-dot");
            const headerStateText = document.getElementById("header-state-txt");
            const metricAngle = document.getElementById("metric-angle");
            const metricSpeed = document.getElementById("metric-speed");
            const metricInactivity = document.getElementById("metric-inactivity");
            const metricFps = document.getElementById("metric-fps");
            
            const settingsForm = document.getElementById("settings-form");
            const testAlertBtn = document.getElementById("test-alert-btn");
            const clearLogsBtn = document.getElementById("clear-logs-btn");
            const logListDiv = document.getElementById("log-list");

            // Live status updater
            function updateSystemStatus() {
                fetch("/api/status")
                    .then(res => res.json())
                    .then(data => {
                        // 1. Update general stats HUD
                        const state = data.state;
                        headerStateText.innerText = state.replace("_", " ");
                        
                        // Status badge color swapping classes
                        statusDot.parentElement.className = "connection-badge " + "state-bg-" + state.toLowerCase().replace(" ", "_");
                        if (state === "UPRIGHT" || state === "LYING DOWN") {
                            statusDot.style.background = "#10b981";
                        } else if (state === "INACTIVE") {
                            statusDot.style.background = "#f59e0b";
                        } else if (state === "FALL ALERT" || state === "CAMERA OFFLINE") {
                            statusDot.style.background = "#ef4444";
                        } else {
                            statusDot.style.background = "#6b7280";
                        }

                        // 2. Metrics
                        metricAngle.innerText = data.torso_angle + "°";
                        metricSpeed.innerText = data.nose_speed.toFixed(2);
                        metricInactivity.innerText = data.inactivity_timer + "s / " + data.last_movement_duration + "s";
                        metricFps.innerText = data.fps;

                        // 3. Populate logs list
                        if (data.alerts && data.alerts.length > 0) {
                            logListDiv.innerHTML = "";
                            data.alerts.forEach(item => {
                                const div = document.createElement("div");
                                let typeClass = "";
                                if (item.state === "INACTIVE") typeClass = "inactive";
                                else if (item.state === "FALL ALERT") typeClass = "fall";
                                
                                div.className = `log-item ${typeClass}`;
                                div.innerHTML = `
                                    <div class="log-time">${item.timestamp}</div>
                                    <div><strong>${item.state}:</strong> ${item.description}</div>
                                `;
                                logListDiv.appendChild(div);
                            });
                        } else {
                            logListDiv.innerHTML = '<div style="color: var(--text-muted); text-align: center; padding: 20px; font-size: 13px;">No historical events recorded.</div>';
                        }
                    })
                    .catch(err => {
                        console.error("Status Sync Failure: ", err);
                        headerStateText.innerText = "SERVER UNREACHABLE";
                        statusDot.style.background = "#6b7280";
                        statusDot.parentElement.className = "connection-badge state-bg-initializing";
                    });
            }

            // Sync Settings values from server on page load
            function loadSettings() {
                fetch("/api/settings")
                    .then(res => res.json())
                    .then(data => {
                        document.getElementById("video_source").value = data.video_source;
                        document.getElementById("normal_inactivity_threshold").value = data.normal_inactivity_threshold;
                        document.getElementById("sleep_inactivity_threshold").value = data.sleep_inactivity_threshold;
                        document.getElementById("fall_velocity_threshold").value = data.fall_velocity_threshold;
                        document.getElementById("angle_3d_threshold").value = data.angle_3d_threshold;
                        document.getElementById("telegram_bot_token").value = data.telegram_bot_token;
                        document.getElementById("telegram_chat_id").value = data.telegram_chat_id;
                        document.getElementById("enable_telegram").checked = data.enable_telegram;
                        document.getElementById("buzzer_enabled").checked = data.buzzer_enabled;
                    });
            }

            // Form Submit settings handler
            settingsForm.addEventListener("submit", function(e) {
                e.preventDefault();
                const formData = new FormData(settingsForm);
                const jsonData = {
                    video_source: formData.get("video_source"),
                    normal_inactivity_threshold: parseInt(formData.get("normal_inactivity_threshold")),
                    sleep_inactivity_threshold: parseInt(formData.get("sleep_inactivity_threshold")),
                    fall_velocity_threshold: parseFloat(formData.get("fall_velocity_threshold")),
                    angle_3d_threshold: parseInt(formData.get("angle_3d_threshold")),
                    telegram_bot_token: formData.get("telegram_bot_token"),
                    telegram_chat_id: formData.get("telegram_chat_id"),
                    enable_telegram: document.getElementById("enable_telegram").checked,
                    buzzer_enabled: document.getElementById("buzzer_enabled").checked
                };

                fetch("/api/settings", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(jsonData)
                })
                .then(res => res.json())
                .then(res => {
                    if (res.success) {
                        alert("Settings updated successfully! System re-configured immediately.");
                    } else {
                        alert("Failed to save settings: " + res.error);
                    }
                });
            });

            // Action triggers
            testAlertBtn.addEventListener("click", () => {
                fetch("/api/test_alarm", { method: "POST" })
                    .then(res => res.json())
                    .then(data => alert("Test Alert Event Dispatched!"));
            });

            clearLogsBtn.addEventListener("click", () => {
                if (confirm("Are you sure you want to clear historical event logs?")) {
                    fetch("/api/clear_logs", { method: "POST" })
                        .then(res => res.json())
                        .then(data => {
                            logListDiv.innerHTML = '<div style="color: var(--text-muted); text-align: center; padding: 20px; font-size: 13px;">No historical events recorded.</div>';
                        });
                }
            });

            // Timers
            loadSettings();
            setInterval(updateSystemStatus, 400); // Poll status every 400ms
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content)

@app.route('/video_feed')
def video_feed():
    """Generates MJPEG streaming endpoint with processed skeletal outputs."""
    def gen():
        global latest_processed_frame
        while True:
            with frame_lock:
                frame_bytes = latest_processed_frame
            if frame_bytes is None:
                # If no frame processed yet, stream offline visual card or sleep
                time.sleep(0.1)
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)
            
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    """Returns JSON state metrics and recent logs."""
    alerts = []
    if os.path.exists(ALERTS_FILE):
        try:
            with open(ALERTS_FILE, 'r') as f:
                alerts = json.load(f)
        except Exception:
            alerts = []
            
    with state_lock:
        response_data = {
            "state": system_status["state"],
            "person_detected": system_status["person_detected"],
            "inactivity_timer": system_status["inactivity_timer"],
            "torso_angle": system_status["torso_angle"],
            "nose_speed": system_status["nose_speed"],
            "fps": system_status["fps"],
            "last_movement_duration": system_status["last_movement_duration"],
            "alerts": alerts[:15]  # Keep latest 15 alerts for dashboard
        }
    return jsonify(response_data)

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    global config
    if request.method == 'POST':
        try:
            new_settings = request.json
            
            # Basic validation
            if not new_settings:
                return jsonify({"success": False, "error": "Invalid JSON"}), 400
                
            # Process and write
            for key in DEFAULT_CONFIG:
                if key in new_settings:
                    # Update config safely
                    config[key] = new_settings[key]
                    
            if save_config(config):
                # Trigger live reload of Buzzer configurations if GPIO pins changed
                global buzzer
                if platform.system() != "Windows" and buzzer is not None:
                    try:
                        buzzer.close() # Free old pin
                        from gpiozero import Buzzer as GpioBuzzer
                        buzzer = GpioBuzzer(config.get("buzzer_pin", 17))
                    except Exception:
                        buzzer = None
                return jsonify({"success": True})
            else:
                return jsonify({"success": False, "error": "Failed to write file"}), 500
        except Exception as ex:
            return jsonify({"success": False, "error": str(ex)}), 500
    else:
        # GET request
        return jsonify(config)

@app.route('/api/test_alarm', methods=['POST'])
def trigger_test_alarm():
    """Manually dispatches fake fall alerts on Telegram and local speakers for testing."""
    log_alert("Manual: User triggered a test system alert", "TEST ALERT")
    send_telegram("🚨 ELDERLY MONITOR TEST: This is a manual test alert dispatched from your master dashboard!")
    threading.Thread(target=play_sound, args=("triple",), daemon=True).start()
    return jsonify({"success": True})

@app.route('/api/clear_logs', methods=['POST'])
def clear_alert_logs():
    """Resets alerts database to empty array."""
    try:
        with open(ALERTS_FILE, 'w') as f:
            json.dump([], f, indent=4)
        return jsonify({"success": True})
    except Exception as ex:
        return jsonify({"success": False, "error": str(ex)}), 500


def get_lan_ip():
    """Discovers and returns the network interface IP on local network."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == '__main__':
    # Discover local LAN IP
    lan_ip = get_lan_ip()
    
    print("\n" + "="*60)
    print("        ELDERLY ACTIVITY MONITOR - WEB SERVER APPLICATION")
    print("="*60)
    print(f" * AI pose-landmarking and state engine running in background.")
    print(f" * Active Platform: {platform.system()} ({platform.release()})")
    print(f" * Localhost Dashboard URL: http://localhost:5000")
    print(f" * Network Dashboard URL:   http://{lan_ip}:5000")
    print("="*60)
    print("Press Ctrl+C to terminate system safely.\n")
    
    # Automatically open local browser window to display master dashboard
    import webbrowser
    webbrowser.open(f"http://localhost:5000")
    
    # Run flask web server in primary thread
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
