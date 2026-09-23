import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math
import winsound
import requests
import urllib.parse

# --- Configuration ---
NORMAL_INACTIVITY_THRESHOLD = 15  
SLEEP_INACTIVITY_THRESHOLD = 30   
MOVEMENT_THRESHOLD_PX = 10        
FALL_VELOCITY_THRESHOLD = 0.5     # Normalized screen units per second (0.5 = half screen height per second)
ANGLE_3D_THRESHOLD = 55           # Degrees from vertical. > 55 means torso is horizontally pitched

# --- Telegram Configuration ---
TELEGRAM_BOT_TOKEN = "8719214387:AAFlHVLbgmpIY7q0UGKqdMFkPGfWM5haX6M" # Get from @BotFather on Telegram
TELEGRAM_CHAT_ID = "5776686318"     # Get from @userinfobot on Telegram
ENABLE_TELEGRAM = True               # Set to True after configuring

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

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), 
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), 
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23), 
    (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), 
    (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

def calculate_3d_angle_from_vertical(p1, p2):
    """
    Calculates the angle in 3D space of the vector between p1 and p2 relative to the Y-axis (Vertical gravity line).
    Returns 0 if perfectly vertical (standing straight). Returns 90 if perfectly horizontal.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    
    magnitude = math.sqrt(dx*dx + dy*dy + dz*dz)
    if magnitude == 0: return 0
    
    # Dot product with Y-axis (0, 1, 0)
    angle_rad = math.acos(abs(dy) / magnitude)
    return math.degrees(angle_rad)

def draw_full_skeleton(frame, landmarks, w, h):
    px_points = {}
    for i, lm in enumerate(landmarks):
        if lm.visibility > 0.4:  
            px_points[i] = (int(lm.x * w), int(lm.y * h))
            cv2.circle(frame, px_points[i], 4, (0, 215, 255), -1)
            
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx in px_points and end_idx in px_points:
            cv2.line(frame, px_points[start_idx], px_points[end_idx], (0, 255, 0), 2)
            
    return px_points

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
    
    print("--- 3D-AWARE Activity Monitoring System Started ---")
    print("Wait for the camera window to open, then press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

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
            px_points = draw_full_skeleton(frame, landmarks, w, h)
            
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
            
            torso_3d_angle = 0
            torso_is_horizontal = False
            
            if 11 in px_points and 12 in px_points and 23 in px_points and 24 in px_points:
                # Extract 3D normalized coordinates directly from landmarks object
                l_shldr, r_shldr = landmarks[11], landmarks[12]
                l_hip, r_hip = landmarks[23], landmarks[24]
                
                mid_shldr_3d = ((l_shldr.x + r_shldr.x)/2, (l_shldr.y + r_shldr.y)/2, (l_shldr.z + r_shldr.z)/2)
                mid_hip_3d = ((l_hip.x + r_hip.x)/2, (l_hip.y + r_hip.y)/2, (l_hip.z + r_hip.z)/2)
                
                torso_3d_angle = calculate_3d_angle_from_vertical(mid_hip_3d, mid_shldr_3d)
                
                cv2.putText(frame, f"3D Torso Angle: {int(torso_3d_angle)} deg", (20, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                if torso_3d_angle > ANGLE_3D_THRESHOLD:
                    torso_is_horizontal = True
                    current_state = "LYING DOWN"
                else:
                    current_state = "UPRIGHT"

            if 0 in px_points:
                nose_y_norm = landmarks[0].y # Normalized coordinate (0.0 to 1.0)
                nose_history.append((current_time, nose_y_norm))
                
                # 1.5 second history
                nose_history = [(t, y) for (t, y) in nose_history if current_time - t <= 1.5]
                
                max_nose_velocity = 0
                if len(nose_history) > 2:
                    for i in range(1, len(nose_history)):
                        t_prev, y_prev = nose_history[i-1]
                        t_curr, y_curr = nose_history[i]
                        dt = t_curr - t_prev
                        if dt > 0:
                            # Velocity in normalized screen units per second
                            val = (y_curr - y_prev) / dt
                            if val > max_nose_velocity:
                                max_nose_velocity = val
                
                cv2.putText(frame, f"Nose Drop Peak Speed: {max_nose_velocity:.2f} norm/s", (20, 115), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

                # --- FALL DETECTION LOGIC ---
                if torso_is_horizontal and max_nose_velocity > FALL_VELOCITY_THRESHOLD:
                    # Final safety check: Is the nose actually below or near the hips?
                    if 23 in px_points and 24 in px_points:
                        l_hip_y, r_hip_y = landmarks[23].y, landmarks[24].y
                        avg_hip_y_norm = (l_hip_y + r_hip_y) / 2
                        
                        # In images, Y increases downwards. So if nose_y >= (hip_y - small_margin), 
                        # the head is lower than the waist, or level with it on the floor.
                        if nose_y_norm >= (avg_hip_y_norm - 0.1):
                            is_falling_alert = True
                            
            cv2.putText(frame, f"State: {current_state}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if current_state=="UPRIGHT" else (0, 165, 255), 2)

        else:
            last_movement_time = current_time
            nose_history.clear()
            current_state = "NO PERSON"
            cv2.putText(frame, "NO PERSON DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        time_since_movement = current_time - last_movement_time
        active_threshold = SLEEP_INACTIVITY_THRESHOLD if current_state == "LYING DOWN" else NORMAL_INACTIVITY_THRESHOLD
        
        if person_detected:
            cv2.putText(frame, f"Inactivity: {int(time_since_movement)}s / {active_threshold}s", (20, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if time_since_movement > active_threshold and not inactivity_reported:
                print(f"ALERT: No movement detected for {active_threshold} seconds! State: {current_state}")
                winsound.Beep(1000, 1000)
                send_telegram_alert(f"⚠️ Elderly Monitor: No movement detected for {active_threshold} seconds! Current State: {current_state}")
                inactivity_reported = True
                
            if is_falling_alert:
                if current_time - fall_detected_time > 3: 
                    print("EMERGENCY ALERT: 3D Omni-Directional Fall Detected!")
                    winsound.Beep(2000, 300)
                    winsound.Beep(2000, 300)
                    winsound.Beep(2000, 300)
                    send_telegram_alert("🚨 EMERGENCY: A sudden fall has been detected by the AI monitoring system!")
                    fall_detected_time = current_time
                    nose_history.clear() 

        if is_falling_alert or (current_time - fall_detected_time < 3):
            cv2.putText(frame, "EMERGENCY: FALL DETECTED!", (20, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

        cv2.imshow('3D Activity Monitor', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
