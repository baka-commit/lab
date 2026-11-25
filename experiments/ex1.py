# -*- coding: utf-8 -*-
# Experiment 1 - Basic OpenCV operations
# =======================================

##exp 1.1 - reading images using OpenCV

import cv2
img=cv2.imread("/content/drive/MyDrive/images/images.png",1)
print(img.shape)
print(img.size)
print(img)

import cv2
img=cv2.imread("/content/drive/MyDrive/images/images.png",0)
print(img.shape)
print(img.size)
print(img)

import cv2
img=cv2.imread("/content/drive/MyDrive/images/images.png",-1)
print(img.shape)
print(img.size)
print(img)

##exp 1.2 - displaying images using OpenCV

import cv2
from google.colab.patches import cv2_imshow
img=cv2.imread("/content/drive/MyDrive/images/images.png",1)
print(img.shape)
print(img.size)
print(img)
cv2_imshow(img)

import cv2
from google.colab.patches import cv2_imshow
img=cv2.imread("/content/drive/MyDrive/images/images.png",0)
print(img.shape)
print(img.size)
print(img)
cv2_imshow(img)

##exp 1.3 - resizing images using OpenCV

import cv2
from google.colab.patches import cv2_imshow
img=cv2.imread("/content/drive/MyDrive/images/images.png",1)
resize=cv2.resize(img,(200,200))
print(resize.shape)
print(resize.size)
cv2_imshow(resize)

import cv2
from google.colab.patches import cv2_imshow
img=cv2.imread("/content/drive/MyDrive/images/images.png",0)
resize=cv2.resize(img,(200,200))
cv2_imshow(resize)
