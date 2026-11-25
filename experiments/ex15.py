# -*- coding: utf-8 -*-
# Experiment 15 - Object Detection using YOLO algorithm on COCO dataset
# ======================================================================

!pip uninstall -y opencv-python opencv-python-headless
!pip install opencv-python==4.10.0.84

!wget -q --show-progress https://data.pjreddie.com/files/yolov3.weights -O yolov3.weights

import os

print("Download complete!")
print("File size:", round(os.path.getsize("yolov3.weights") / (1024**2), 2), "MB")

import cv2
import numpy as np

#load the yolo model files
net = cv2.dnn.readNet("/content/yolov3.weights", "/content/yolov3.cfg")
classes = []
with open("/content/coco.names", "r") as f:
    classes.append(f.read().strip())
print(classes)

#load img
img = cv2.imread("/content/horse.jpg")
from google.colab.patches import cv2_imshow
cv2_imshow(img)
img.shape

img = cv2.resize(img,(340, 226))
img.shape

height, width, channels = img.shape
print(height, width, channels)

blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), (0,0,0), swapRB=True, crop=False)
# Set input for the network
net.setInput(blob)

# Get layer names and output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
print(output_layers)

# Run the forward pass
outs = net.forward(output_layers)
print(outs)

# We also need to extract individual class names from the classes list
# (The coco.names file was read as a single multi-line string in cell H3iCS-XTFFXS)
if len(classes) == 1 and '\n' in classes[0]:
    class_list = classes[0].strip().split('\n')
else:
    class_list = classes

# Initialize lists for detected bounding boxes, confidences, and class IDs
class_ids = []
confidences = []
boxes = []

# Scan through all output detections
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Filter out weak predictions by ensuring confidence is over 50%
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maximum Suppression (NMS) to eliminate redundant overlapping boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

# Draw the bounding boxes and labels on the image
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(class_list[class_ids[i]])
        confidence = confidences[i]

        # Draw bounding box rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Put label text near the bounding box
        cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the final output image with detections
cv2_imshow(img)
