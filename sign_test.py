import cv2 
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

# Initialize camera, hand detector, and classifier
cap = cv2.VideoCapture(0)  # 0-Device Camera
detector = HandDetector()
classifier = Classifier("E:\Sign Language\Sign_Lang_Data\keras_model.h5", "Sign_Lang_Data/labels.txt")  # Make sure the path uses forward slashes or double backslashes
offset = 20
imgSize = 300

# Define the labels for classification
labels = ["Hello", "Help", "Me", "No", "Ok", "ThankYou", "Yes"]

while True:
    success, img = cap.read()
    if not success:
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand region
        imgCrop = img[max(0, y - offset):min(y + h + offset, img.shape[0]), 
                      max(0, x - offset):min(x + w + offset, img.shape[1])]

        # Ensure that imgCrop is valid (non-zero size)
        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            aspectRatio = h / w

            # Resize the cropped hand image to fit the white background
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Get prediction from the classifier
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            confidence = max(prediction) * 100  # Convert to percentage

            # Display prediction and bounding box
            cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 350, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, f'{labels[index]} {confidence:.2f}%', (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)
            
            # Display intermediate images for debugging
            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)
    
    # Display the main output image
    cv2.imshow("Image", imgOutput)

    # Break the loop on key press
    key = cv2.waitKey(1)
    if key == ord('q'):  # Press 'q' to quit
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
