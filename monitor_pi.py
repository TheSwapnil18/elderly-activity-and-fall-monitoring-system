import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math
import requests
import urllib.parse
import platform

# --- Hardware Setup ---
try:
    from gpiozero import Buzzer
    # Assuming an active buzzer is connected to GPIO pin 17 on the Raspberry Pi
    buzzer = Buzzer(17)
except Exception:
    buzzer = None

def play_alarm(pattern="single"):
    """Replaces Windows-specific winsound with Raspberry Pi GPIO Buzzer"""
    if buzzer:
        if pattern == "single":
            buzzer.beep(on_time=1, off_time=0, n=1)
        elif pattern == "triple":
            buzzer.beep(on_time=0.3, off_time=0.2, n=3)
    else:
        print("\a") # Terminal bell fallback if no buzzer attached

# --- Configuration ---
NORMAL_INACTIVITY_THRESHOLD = 15  
SLEEP_INACTIVITY_THRESHOLD = 30   
MOVEMENT_THRESHOLD_PX = 10        
FALL_VELOCITY_THRESHOLD = 0.5     
ANGLE_3D_THRESHOLD = 55           

# --- Telegram Configuration ---
TELEGRAM_BOT_TOKEN = "8719214387:AAFlHVLbgmpIY7q0UGKqdMFkPGfWM5haX6M"
TELEGRAM_CHAT_ID = "5776686318"     
ENABLE_TELEGRAM = True               

def send_telegram_alert(message):
    if not ENABLE_TELEGRAM or TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN" or not TELEGRAM_BOT_TOKEN:
        return
    try:
        encoded_message = urllib.parse.quote(message)
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&text={encoded_message}"
        response = requests.get(url)
        if response.status_code == 200:
            print("--> Telegram Alert Sent Successfully!")
        else:
            print(f"--> Failed to send Telegram alert. Status: {response.status_code}")
    except Exception as e:
        print(f"--> Telegram Error: {e}")

def calculate_3d_angle_from_vertical(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    magnitude = math.sqrt(dx*dx + dy*dy + dz*dz)
    if magnitude == 0: return 0
    angle_rad = math.acos(abs(dy) / magnitude)
    return math.degrees(angle_rad)

def main():
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)
    
    try:
        detector = vision.PoseLandmarker.create_from_options(options)
    except Exception as e:
        print(f"Error initializing MediaPipe: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    last_movement_time = time.time()
    fall_detected_time = 0
    inactivity_reported = False
    
    current_state = "UNKNOWN"
    last_px_points = {}
    nose_history = []  
    
    print("--- HEADLESS RASPBERRY PI MONITORING SYSTEM STARTED ---")
    print("Running in background without GUI. Press Ctrl+C in terminal to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: 
                print("Failed to grab frame. Retrying...")
                time.sleep(1)
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)

            h, w, _ = frame.shape
            current_time = time.time()
            
            person_detected = False
            is_falling_alert = False

            if len(detection_result.pose_landmarks) > 0:
                person_detected = True
                landmarks = detection_result.pose_landmarks[0]
                
                # Extract px_points for movement calculation
                px_points = {}
                for i, lm in enumerate(landmarks):
                    if lm.visibility > 0.4:  
                        px_points[i] = (int(lm.x * w), int(lm.y * h))
                
                # --- 1. MICRO-MOVEMENT DETECTION ---
                moved = False
                if last_px_points:
                    total_shift, points_checked = 0, 0
                    for i in px_points:
                        if i in last_px_points:
                            dx = px_points[i][0] - last_px_points[i][0]
                            dy = px_points[i][1] - last_px_points[i][1]
                            total_shift += math.sqrt(dx*dx + dy*dy)
                            points_checked += 1
                    if points_checked > 0 and (total_shift / points_checked) > MOVEMENT_THRESHOLD_PX:
                        moved = True
                last_px_points = px_points
                
                if moved:
                    last_movement_time = current_time
                    inactivity_reported = False

                # --- 2. 3D OMNI-DIRECTIONAL FALL COMPUTATION ---
                torso_is_horizontal = False
                
                if 11 in px_points and 12 in px_points and 23 in px_points and 24 in px_points:
                    l_shldr, r_shldr = landmarks[11], landmarks[12]
                    l_hip, r_hip = landmarks[23], landmarks[24]
                    
                    mid_shldr_3d = ((l_shldr.x + r_shldr.x)/2, (l_shldr.y + r_shldr.y)/2, (l_shldr.z + r_shldr.z)/2)
                    mid_hip_3d = ((l_hip.x + r_hip.x)/2, (l_hip.y + r_hip.y)/2, (l_hip.z + r_hip.z)/2)
                    
                    torso_3d_angle = calculate_3d_angle_from_vertical(mid_hip_3d, mid_shldr_3d)
                    
                    if torso_3d_angle > ANGLE_3D_THRESHOLD:
                        torso_is_horizontal = True
                        current_state = "LYING DOWN"
                    else:
                        current_state = "UPRIGHT"

                if 0 in px_points:
                    nose_y_norm = landmarks[0].y 
                    nose_history.append((current_time, nose_y_norm))
                    
                    nose_history = [(t, y) for (t, y) in nose_history if current_time - t <= 1.5]
                    
                    max_nose_velocity = 0
                    if len(nose_history) > 2:
                        for i in range(1, len(nose_history)):
                            t_prev, y_prev = nose_history[i-1]
                            t_curr, y_curr = nose_history[i]
                            dt = t_curr - t_prev
                            if dt > 0:
                                val = (y_curr - y_prev) / dt
                                if val > max_nose_velocity:
                                    max_nose_velocity = val

                    # --- FALL DETECTION LOGIC ---
                    if torso_is_horizontal and max_nose_velocity > FALL_VELOCITY_THRESHOLD:
                        if 23 in px_points and 24 in px_points:
                            l_hip_y, r_hip_y = landmarks[23].y, landmarks[24].y
                            avg_hip_y_norm = (l_hip_y + r_hip_y) / 2
                            
                            if nose_y_norm >= (avg_hip_y_norm - 0.1):
                                is_falling_alert = True

            else:
                last_movement_time = current_time
                nose_history.clear()
                current_state = "NO PERSON"

            time_since_movement = current_time - last_movement_time
            active_threshold = SLEEP_INACTIVITY_THRESHOLD if current_state == "LYING DOWN" else NORMAL_INACTIVITY_THRESHOLD
            
            if person_detected:
                if time_since_movement > active_threshold and not inactivity_reported:
                    print(f"ALERT: No movement detected for {active_threshold} seconds! State: {current_state}")
                    play_alarm("single")
                    send_telegram_alert(f"⚠️ Elderly Monitor: No movement detected for {active_threshold} seconds! Current State: {current_state}")
                    inactivity_reported = True
                    
                if is_falling_alert:
                    if current_time - fall_detected_time > 3: 
                        print("EMERGENCY ALERT: 3D Omni-Directional Fall Detected!")
                        play_alarm("triple")
                        send_telegram_alert("🚨 EMERGENCY: A sudden fall has been detected by the AI monitoring system!")
                        fall_detected_time = current_time
                        nose_history.clear() 

            # Sleep briefly to prevent 100% CPU usage in a tight while loop
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopping Monitor...")
    finally:
        cap.release()

if __name__ == "__main__":
    main()
