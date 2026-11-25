# -*- coding: utf-8 -*-
# Experiment 14 - Using OpenCV to detect faces
# =============================================

!pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless

!pip install opencv-contrib-python==5.1.0.95

import cv2
from google.colab.patches import cv2_imshow

print(cv2.__version__)

face_cascade = cv2.CascadeClassifier(
    "/content/haarcascade_frontalface_default.xml"
)

img = cv2.imread("/content/images.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2_imshow(gray_img)

#apply the face cascade to the gray image
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
print(faces)

for x, y, w, h in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

cv2_imshow(img)
