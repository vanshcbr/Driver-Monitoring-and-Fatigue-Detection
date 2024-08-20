import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from tkinter import Tk, Label, Button
from playsound import playsound
import threading

# Constants for fatigue detection
EYE_AR_THRESH = 0.2  # Threshold for Eye Aspect Ratio (EAR)
EYE_AR_CONSEC_FRAMES = 20  # Number of consecutive frames the eyes must be below the threshold to trigger an alert

# Initialize MediaPipe for face detection and landmark identification
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Extract eye coordinates from MediaPipe landmarks
def get_eye_coordinates(landmarks):
    left_eye = [landmarks[362], landmarks[385], landmarks[387], landmarks[263], landmarks[373], landmarks[380]]
    right_eye = [landmarks[33], landmarks[160], landmarks[158], landmarks[133], landmarks[153], landmarks[144]]
    left_eye = [(int(point.x * width), int(point.y * height)) for point in left_eye]
    right_eye = [(int(point.x * width), int(point.y * height)) for point in right_eye]
    return np.array(left_eye), np.array(right_eye)

# Initialize GUI
class DriverMonitorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Monitoring System")
        self.label = Label(root, text="Monitoring Driver...", font=("Helvetica", 16))
        self.label.pack(pady=20)
        self.alert_button = Button(root, text="Stop Alert", command=self.stop_alert, state="disabled")
        self.alert_button.pack(pady=20)

    def display_alert(self):
        self.label.config(text="Fatigue Detected! Wake Up!", fg="red")
        self.alert_button.config(state="normal")
        self.play_alert_sound()

    def stop_alert(self):
        self.label.config(text="Monitoring Driver...", fg="black")
        self.alert_button.config(state="disabled")

    def play_alert_sound(self):
        threading.Thread(target=lambda: playsound("alert.wav")).start()

# Monitor driver fatigue and raise alert
def monitor_driver():
    global width, height
    cap = cv2.VideoCapture(0)
    counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye, right_eye = get_eye_coordinates(face_landmarks.landmark)
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0

                # Draw eye contours
                for (x, y) in np.concatenate((left_eye, right_eye), axis=0):
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # Check if EAR is below the threshold
                if ear < EYE_AR_THRESH:
                    counter += 1
                    if counter >= EYE_AR_CONSEC_FRAMES:
                        gui.display_alert()  # Trigger alert
                else:
                    counter = 0

        cv2.imshow("Driver Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Initialize the application
if __name__ == "__main__":
    root = Tk()
    gui = DriverMonitorGUI(root)
    threading.Thread(target=monitor_driver).start()
    root.mainloop()
