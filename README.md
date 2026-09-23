# elderly-activity-monitoring-system

A robust, real-time computer vision monitoring system designed to track human activity and detect falls using edge AI. Originally prototyped for Windows PC webcams with a planned deployment to a Raspberry Pi 4 Model B.

This project uses **MediaPipe's Vision Tasks API** to run a lightweight, high-performance skeletal tracking model without the need for cloud processing, ensuring absolute privacy for the user.

---

## ✨ Key Features

1. **Context-Aware State Tracking:**
   - Detects if a person is `STANDING`, `UPRIGHT` (Sitting), or `LYING DOWN`.
2. **Multi-Variable Fall Detection Algorithm:**
   - Evaluates a matrix of conditions to eliminate false positives (like sitting down fast or leaning towards the camera). A fall is only triggered if:
     - The body bounding box aspect ratio becomes horizontally dominant.
     - The Torso Angle becomes perfectly horizontal.
     - The Nose experiences a massive downward velocity spike (measured over a 1.5s sliding window).
     - The Head drops to the physical level of the Ankles (or Hips, if Ankles are out of frame).
3. **Smart Inactivity Monitoring:**
   - Tracks how long a person is completely inactive.
   - Dynamic Thresholds: Normal inactivity timer is set to 15 seconds (for testing), while sleeping/resting allows for 30 seconds (for testing). 
   - **Micro-movement tracking**: Even the smallest shift of a hand or foot resets the inactivity timer while sleeping.
4. **Full-Body Skeletal Rendering:**
   - Visually maps 33 facial and body landmarks directly onto the live camera feed.

---

## 🛠️ Hardware & Software Requirements

### Hardware
* **Testing:** Windows PC / Laptop with a built-in or USB webcam.
* **Production:** Raspberry Pi 4 Model B with a Raspberry Pi Camera Module or USB Webcam.

### Software Stack
* **Python 3.x**
* **OpenCV:** For video stream capture and image rendering.
* **MediaPipe:** Google's framework for on-device machine learning (specifically the `PoseLandmarker` task).
* **NumPy:** Mathematical calculations for angles and bounding boxes.

---

## 🚀 Setup & Installation (Windows)

**1. Open your terminal in the project directory:**
```cmd
cd path/to/DIY Project
```

**2. Create and Activate a Python Virtual Environment:**
```cmd
python -m venv venv
.\venv\Scripts\activate
```

**3. Install Required Libraries:**
Ensure you have `requirements.txt` in the folder containing `opencv-python`, `mediapipe`, and `numpy`.
```cmd
pip install -r requirements.txt
```

**4. Download the AI Model:**
The new MediaPipe Tasks API requires the `.task` model file to be downloaded locally. You can download `pose_landmarker_lite.task` directly using this python command:
```cmd
python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task', 'pose_landmarker_lite.task')"
```

---

## 🏃 Usage Instructions

Once the environment is activated, simply run the Python script:

```cmd
python monitor.py
```

1. A window will open showing your webcam feed with the AI skeleton overlaid.
2. **Testing Inactivity:** Stand completely still. After 15 seconds, the console will print an alert and play an audible beep.
3. **Testing Falling:** While standing in frame, drop horizontally as fast as possible. You should see the `EMERGENCY: FALL DETECTED!` text flash red on the screen accompanied by a loud triple-beep.
4. **To Exit:** Click on the camera window and press the **`q`** key on your keyboard.

---

## 🧠 Future Enhancements Roadmap
* Replace native Windows audio beeps with **WhatsApp/Telegram** API messaging integrations.
* Adapt camera framerate drops for optimal heat management on the Raspberry Pi.
* Implement a database to generate a Daily Activity Report graph.
