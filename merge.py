import streamlit as st
import cv2
from keras.models import model_from_json
import numpy as np
from ultralytics import YOLO
import time
from PIL import Image
import os
import subprocess

# Function to load emotion detection model
def load_emotion_model():
    try:
        # Load the model architecture from the JSON file
        with open("C:/Users/Vikas/Desktop/Face_emotion_detection/emotiondetector.json", "r") as json_file:
            model_json = json_file.read()
        emotion_model = model_from_json(model_json)
        emotion_model.load_weights("C:/Users/Vikas/Desktop/Face_emotion_detection/emotiondetector.h5")
        return emotion_model
    except Exception as e:
        st.error(f"Error loading emotion detection model: {e}")
        return None

# Load emotion detection model
emotion_model = load_emotion_model()

# Load pre-trained age and gender models
age_net = cv2.dnn.readNetFromCaffe('C:/Users/Vikas/Desktop/Face_emotion_detection/deploy_age.prototxt', 
                                   'C:/Users/Vikas/Desktop/Face_emotion_detection/age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('C:/Users/Vikas/Desktop/Face_emotion_detection/deploy_gender.prototxt', 
                                      'C:/Users/Vikas/Desktop/Face_emotion_detection/gender_net.caffemodel')

# Load YOLO model
object_model = YOLO("yolov8x.pt")

# Labels for predictions
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
gender_labels = ['Male', 'Female']
age_labels = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Function to predict age and gender
def predict_age_gender(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_labels[gender_preds[0].argmax()]
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_labels[age_preds[0].argmax()]
    return gender, age

# Streamlit Sidebar Configuration
st.sidebar.title("Configuration")
object_detection = st.sidebar.checkbox("Enable Object Detection", value=True)
grayscale_view = st.sidebar.checkbox("Enable Grayscale View", value=False)
st.sidebar.write("Adjust Settings for Real-Time Detection")

# Main Streamlit Page
st.title("Real-Time Emotion, Age, Gender, and Object Detection")
st.markdown("""This application uses advanced deep learning models for real-time detection. You can analyze emotions, predict age and gender, and detect objects in the frame.""")

# Placeholder for webcam feed
frame_placeholder = st.empty()

# Start/Stop Webcam
if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False

if st.button("Start Webcam") and not st.session_state.webcam_running:
    st.session_state.webcam_running = True

if st.button("Stop Webcam") and st.session_state.webcam_running:
    st.session_state.webcam_running = False

# Process Webcam
def process_webcam():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    fps = 0
    prev_time = time.time()
    frame_rgb = None  # Initialize the variable

    while cap.isOpened() and st.session_state.webcam_running:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if grayscale_view else frame
        faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.3, 5)
        face_count = len(faces)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Emotion prediction
            if emotion_model:
                gray_face = cv2.resize(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY), (48, 48)).reshape(1, 48, 48, 1) / 255.0
                emotion_pred = emotion_model.predict(gray_face, verbose=0)
                emotion_label = emotion_labels[np.argmax(emotion_pred)]
            else:
                emotion_label = "N/A"

            # Age and gender prediction
            gender, age = predict_age_gender(cv2.resize(face_img, (227, 227)))
            label = f"{emotion_label}, {gender}, {age}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if object_detection:
            results = object_model.predict(frame, stream=True)
            for result in results:
                if hasattr(result, 'boxes') and result.boxes.xyxy is not None:
                    for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf.numpy(), result.boxes.cls.numpy().astype(int)):
                        x1, y1, x2, y2 = map(int, box)
                        cls_name = object_model.names[cls] if cls < len(object_model.names) else f"Class {cls}"
                        conf_value = float(conf)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{cls_name} ({conf_value:.2f})"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Display FPS and Face Count
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Faces Detected: {face_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

    cap.release()
    return frame_rgb

# Snipping tool function
def open_snipping_tool():
    try:
        # Open Snipping Tool for Windows (only if running on Windows)
        subprocess.Popen(["snippingtool"])
    except Exception as e:
        st.error(f"Error opening Snipping Tool: {e}")

# Button to invoke Snipping Tool
if st.sidebar.button("Open Snipping Tool"):
    open_snipping_tool()

if st.session_state.webcam_running:
    process_webcam()
else:
    st.write("Webcam is stopped. Click 'Start Webcam' to begin.")
