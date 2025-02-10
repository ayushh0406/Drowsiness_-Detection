import cv2
import dlib
import pyttsx3
import os
import time
import streamlit as st
from collections import deque


# Manhattan Distance Calculation (L1 Distance)
def calculate_manhattan(eye):
    A = abs(eye[1][0] - eye[5][0]) + abs(eye[1][1] - eye[5][1])
    B = abs(eye[2][0] - eye[4][0]) + abs(eye[2][1] - eye[4][1])
    C = abs(eye[0][0] - eye[3][0]) + abs(eye[0][1] - eye[3][1])
    ear = (A + B) / (2.0 * C)
    return ear

# Class to handle PERCLOS calculation
class PERCLOS_Detector:
    def __init__(self, window_size=30, eye_closed_threshold=0.25, perclos_threshold=0.8):
        self.window_size = window_size
        self.eye_closed_threshold = eye_closed_threshold
        self.perclos_threshold = perclos_threshold
        self.ear_window = deque(maxlen=window_size)
        self.alert_triggered = False  # Flag to track if alert has been triggered

    def update_ear(self, ear):
        self.ear_window.append(ear)
        if len(self.ear_window) == self.window_size:
            perclos = self.calculate_perclos()
            print(f'PERCLOS: {perclos * 100:.2f}%')
            if perclos >= self.perclos_threshold:   
                self.trigger_alert()
            else:
                self.alert_triggered = False  # Reset alert trigger when eyes are open

    def calculate_perclos(self):
        closed_eyes_count = sum(1 for ear in self.ear_window if ear < self.eye_closed_threshold)
        return closed_eyes_count / len(self.ear_window)

    def trigger_alert(self):
        if not self.alert_triggered:  # Trigger alert only if it hasn't been triggered yet
            print("Drowsiness Detected! Alert!")
            engine.say("Alert! Wake up! You are feeling drowsy!")
            engine.runAndWait()
            self.alert_triggered = True  # Set alert as triggered

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Path to the shape predictor file
shape_predictor_path = r"C:\Users\bhomi\OneDrive\Desktop\Projects\Drowsines Detection\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat"

# Check if the shape predictor file exists
if not os.path.isfile(shape_predictor_path):
    raise FileNotFoundError(f"Shape predictor file not found at {shape_predictor_path}. Please check the path.")

# Load Dlib's face detector and landmark predictor
try:
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(shape_predictor_path)
except RuntimeError as e:
    print(f"Failed to load the shape predictor. Error: {e}")
    exit(1)

# Set up camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set height

# Initialize PERCLOS detector
perclos_detector = PERCLOS_Detector(window_size=30, eye_closed_threshold=0.25, perclos_threshold=0.8)

# Create a full-screen window
cv2.namedWindow("PERCLOS Drowsiness Detector", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("PERCLOS Drowsiness Detector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    start_time = time.time()  # Record the start time

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_frame)

    for face in faces:
        landmarks = landmark_predictor(gray_frame, face)

        # Get coordinates of left and right eyes (36-41 for left, 42-47 for right)
        left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
        right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]

        # Calculate EAR using Manhattan distance for both eyes and take the average
        left_ear = calculate_manhattan(left_eye)
        right_ear = calculate_manhattan(right_eye)
        average_ear = (left_ear + right_ear) / 2.0

        # Update the PERCLOS detector with the new EAR value
        perclos_detector.update_ear(average_ear)

        # Draw the eye regions on the frame
        for (x, y) in left_eye + right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Show alert text when drowsiness is detected
        if perclos_detector.alert_triggered:
            cv2.putText(frame, "Alert! Drowsiness Detected!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)

    # Calculate time taken for processing and display timestamp
    processing_time = time.time() - start_time
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    cv2.putText(frame, f"Time: {timestamp}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Processing Time: {processing_time:.2f}s", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("PERCLOS Drowsiness Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
