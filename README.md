# Hand Sign Language Recognition using OpenCV and Teachable Machine

This project is a real-time hand sign language recognition system built using OpenCV, Cvzone, and a model trained with Google's Teachable Machine.

## Video Demo

https://github.com/user-attachments/assets/9b467274-c408-4a43-b123-f700fec39290

## Features
- Real-time hand tracking using OpenCV and Cvzone.
- Predicts hand signs using a neural network model trained with Teachable Machine.
- Displays the prediction label and confidence percentage on the video feed.

## How to Train the Model with Google Teachable Machine
1. Go to [Teachable Machine](https://teachablemachine.withgoogle.com/).
2. Create a new Image Project.
3. Capture images for various hand gestures and label them accordingly.
4. Train the model using Teachable Machine’s training interface.
5. Export the model as a **Keras** model.
6. Download the `.h5` model and the `labels.txt` file.
7. Replace the `keras_model.h5` and `labels.txt` in the `model/` directory with your trained files.

## Getting Started

### Prerequisites
- Python 3.6+
- OpenCV
- Cvzone
- TensorFlow/Keras

# Running the Project

1.Run the Python script: python hand_sign_recognition.py

Demo
The script will open your webcam and start recognizing hand gestures in real-time.

Press 'q' to quit the application.


-- Harsh Kumar Sharma 

