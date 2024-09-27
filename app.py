import cv2
import dlib
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from collections import deque
import os

# --- PATH CONFIGURATION ---
# For Deployment: Ensure the .dat file is in your GitHub root folder
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# --- MODEL LOADING ---
@st.cache_resource
def load_models():
    detector = dlib.get_frontal_face_detector()
    if not os.path.exists(SHAPE_PREDICTOR_PATH):
        st.error(f"Model file not found! Please upload {SHAPE_PREDICTOR_PATH} to your GitHub.")
        return None, None
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    return detector, predictor

face_detector, landmark_predictor = load_models()

# --- LOGIC FUNCTIONS ---
def calculate_manhattan(eye):
    # Optimized Manhattan Distance (L1) based on your original logic
    A = abs(eye[1][0] - eye[5][0]) + abs(eye[1][1] - eye[5][1])
    B = abs(eye[2][0] - eye[4][0]) + abs(eye[2][1] - eye[4][1])
    C = abs(eye[0][0] - eye[3][0]) + abs(eye[0][1] - eye[3][1])
    return (A + B) / (2.0 * C)

class DrowsinessTransformer(VideoTransformerBase):
    def __init__(self):
        self.window_size = 20
        self.eye_closed_threshold = 0.25
        self.perclos_threshold = 0.7 # Slightly lowered for webcams
        self.ear_window = deque(maxlen=self.window_size)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        status = "Active"
        color = (0, 255, 0)
        current_perclos = 0

        for face in faces:
            landmarks = landmark_predictor(gray, face)
            
            # Extract eye coordinates
            left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
            right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]

            # Calculate EAR
            l_ear = calculate_manhattan(left_eye)
            r_ear = calculate_manhattan(right_eye)
            avg_ear = (l_ear + r_ear) / 2.0

            # PERCLOS Update
            self.ear_window.append(avg_ear)
            if len(self.ear_window) == self.window_size:
                closed_count = sum(1 for e in self.ear_window if e < self.eye_closed_threshold)
                current_perclos = closed_count / self.window_size

            # Check Drowsiness
            if current_perclos >= self.perclos_threshold:
                status = "DROWSY ALERT!"
                color = (0, 0, 255)
            
            # Draw eye landmarks
            for (x, y) in left_eye + right_eye:
                cv2.circle(img, (x, y), 1, (255, 255, 255), -1)

        # UI Overlays
        cv2.putText(img, f"Status: {status}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, f"PERCLOS: {current_perclos:.2%}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return img

# --- STREAMLIT UI ---
st.set_page_config(page_title="Drowsiness Detector", layout="wide")
st.title("🚗 Real-Time Drowsiness Detection")
st.write("Using Manhattan Distance (L1) and PERCLOS logic.")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_streamer(
    key="drowsiness-det",
    video_transformer_factory=DrowsinessTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

st.sidebar.info("This system analyzes eye closure duration (PERCLOS) using 68-point facial landmarks.")