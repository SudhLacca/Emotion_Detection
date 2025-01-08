# Real-Time Emotion, Age, Gender, and Object Detection

This project is a Streamlit-based web application that uses advanced deep learning models for real-time detection of emotions, age, gender, and objects in a video feed. The application processes webcam input and leverages state-of-the-art AI models for comprehensive analysis.


Features

- Emotion Detection: Identifies human emotions using a trained CNN model.
- Age & Gender Prediction: Uses pre-trained DNN models to predict age group and gender.
- Object Detection: Utilizes the YOLOv8 model for detecting and labeling objects in the video feed.
- Real-Time Analysis: Processes webcam feed and displays results with high accuracy and speed.
- Snipping Tool Integration: Allows users to take screenshots for analysis.
- Adjustable Settings: Toggle object detection, grayscale mode, and other configurations.


Technologies Used

- Frontend: Streamlit for interactive UI
- Backend Models:
  - Emotion Detection: Keras (TensorFlow-based)
  - Age and Gender Prediction: OpenCV's DNN module with Caffe models
  - Object Detection: YOLOv8 from the `ultralytics` library
- Additional Libraries: OpenCV, NumPy, PIL, and subprocess

Requirements

- Python 3.8+
- Libraries:
  - `streamlit`
  - `opencv-python`
  - `keras`
  - `numpy`
  - `Pillow`
  - `ultralytics`
- YOLOv8 Model: Pre-trained `yolov8x.pt` weights
- Pre-trained Caffe models for age and gender prediction:
  - `deploy_age.prototxt` and `age_net.caffemodel`
  - `deploy_gender.prototxt` and `gender_net.caffemodel`
