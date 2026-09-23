# 🛡️ Elderly Activity & Fall Monitoring System

[![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python&logoColor=white)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-Video_Capture-green?style=flat-square&logo=opencv&logoColor=white)]()
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Edge_AI-orange?style=flat-square&logo=google&logoColor=white)]()
[![Status](https://img.shields.io/badge/Status-Prototype-brightgreen?style=flat-square)]()

> A robust, real-time computer vision monitoring system designed to track human activity and detect falls using edge AI. Originally prototyped for Windows PC webcams with a planned deployment to a Raspberry Pi 4 Model B.

This project uses **MediaPipe's Vision Tasks API** to run a lightweight, high-performance skeletal tracking model without the need for cloud processing, ensuring absolute privacy for the user.

---

## ✨ Key Features

| Feature | Description | Highlights |
| :--- | :--- | :--- |
| 🕵️ **State Tracking** | Context-aware detection of posture. | Tracks if a person is `STANDING`, `UPRIGHT` (Sitting), or `LYING DOWN`. |
| 🚨 **Fall Detection** | Multi-variable algorithm to eliminate false positives. | Uses aspect ratio, torso angle, downward velocity, and relative head position. |
| ⏱️ **Smart Inactivity** | Monitors prolonged periods of complete stillness. | Dynamic thresholds (15s normal / 30s sleeping) & **micro-movement tracking**. |
| 🦴 **Skeletal Rendering**| Full-body skeletal mapping directly on the feed. | Visually maps 33 facial and body landmarks in real-time. |

### 🔍 Deep Dive: Fall Detection Algorithm
A fall is **only** triggered if all the following conditions are met simultaneously (preventing false alarms from sitting fast or leaning):
* 📐 **Aspect Ratio:** The body bounding box becomes horizontally dominant.
* 🤸 **Torso Angle:** Becomes perfectly horizontal.
* 📉 **Velocity Spike:** The nose experiences a massive downward velocity spike (over a 1.5s sliding window).
* ⬇️ **Head Position:** The head drops to the physical level of the ankles (or hips, if ankles are hidden).

---

## 🛠️ Hardware & Software Requirements

### 🧰 Hardware
* **Testing Phase:** Windows PC / Laptop with a built-in or USB webcam.
* **Production Phase:** Raspberry Pi 4 Model B with a Raspberry Pi Camera Module or USB Webcam.

### 💻 Software Stack
* **Python 3.x**
* **OpenCV:** For video stream capture and image rendering.
* **MediaPipe:** Google's framework for on-device machine learning (specifically the `PoseLandmarker` task).
* **NumPy:** Mathematical calculations for angles, bounding boxes, and velocity.

---

## 🚀 Setup & Installation (Windows)

**1. Clone the repository and navigate into it:**
```cmd
git clone https://github.com/TheSwapnil18/elderly-activity-monitoring-system.git
cd elderly-activity-monitoring-system
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

**5. Configure the Application:**
Copy the template configuration file to create your own local `config.json`.
```cmd
copy config.example.json config.json
```
Open `config.json` and replace `"YOUR_TELEGRAM_BOT_TOKEN_HERE"` and `"YOUR_TELEGRAM_CHAT_ID_HERE"` with your actual Telegram bot credentials.

---

## 🏃 Usage Instructions

Once the environment is activated, simply run the Python script:

```cmd
python monitor.py
```

### 🧪 Testing the System

* 👁️ **Visuals:** A window will open showing your webcam feed with the AI skeleton overlaid.
* 🧍 **Testing Inactivity:** Stand completely still. After 15 seconds, the console will print an alert and play an audible beep.
* 💥 **Testing Falling:** While standing in frame, drop horizontally as fast as possible. 
  > 🚨 You should see the `EMERGENCY: FALL DETECTED!` text flash red on the screen accompanied by a loud triple-beep.
* 🛑 **To Exit:** Click on the camera window and press the **`q`** key on your keyboard.

---

## 🧠 Future Enhancements Roadmap

- [ ] 📱 Replace native Windows audio beeps with **WhatsApp/Telegram** API messaging integrations.
- [ ] 🌡️ Adapt camera framerate drops for optimal heat management on the Raspberry Pi.
- [ ] 📊 Implement a database to generate a Daily Activity Report graph.
