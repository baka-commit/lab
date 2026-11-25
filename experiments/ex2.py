# -*- coding: utf-8 -*-
# Experiment 2 - Basic image processing operations
# =================================================

##exp 2.1 - Histogram equalization

import cv2
import matplotlib.pyplot as plt

img=cv2.imread("/content/drive/MyDrive/images/images.png",0)

hist1=cv2.calcHist([img],[0],None,[256],[0,256])
img2=cv2.equalizeHist(img)
hist2=cv2.calcHist([img2],[0],None,[256],[0,256])

plt.subplot(221),plt.imshow(img)
plt.subplot(222),plt.plot(hist1)
plt.subplot(223),plt.imshow(img2)
plt.subplot(224),plt.plot(hist2)

##exp 2.2 - Thresholding

import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread("/content/drive/MyDrive/images/images.png",0)
thresholdervalue=125
max_value=255

print(img)
print(img.shape)
print()

result=cv2.threshold(img,thresholdervalue,max_value,cv2.THRESH_BINARY)
print(type(result))
print(len(result))
print("retrn value")
print(result)
print()

t1=cv2.threshold(img,thresholdervalue,max_value,cv2.THRESH_BINARY)
t2=cv2.threshold(img,thresholdervalue,max_value,cv2.THRESH_BINARY_INV)
t3=cv2.threshold(img,thresholdervalue,max_value,cv2.THRESH_TRUNC)
t4=cv2.threshold(img,thresholdervalue,max_value,cv2.THRESH_TOZERO)
t5=cv2.threshold(img,thresholdervalue,max_value,cv2.THRESH_TOZERO_INV)
pixclevalue=img[100,100]

print(pixclevalue)
print(t1)
print(t2)
print(t3)
print(t4)
print(t5)

plt.subplot(2,3,1),plt.imshow(img)
plt.subplot(2,3,2),plt.imshow(t1[1])
plt.subplot(2,3,3),plt.imshow(t2[1])
plt.subplot(2,3,4),plt.imshow(t3[1])
plt.subplot(2,3,5),plt.imshow(t4[1])
plt.subplot(2,3,6),plt.imshow(t5[1])

##exp 2.3 - Edge detection

import cv2
import matplotlib.pyplot as plt

img = cv2.imread("/content/image.jpg", 0)

edges = cv2.Canny(img, 100, 200)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Edges")
plt.imshow(edges, cmap="gray")
plt.axis("off")

plt.show()

##exp 2.4 - Morphological operations

import numpy as np
import cv2
import matplotlib.pyplot as plt

img=cv2.imread("/content/drive/MyDrive/images/images.png",0)
kernel=np.ones((4,4,),np.uint8)

dilated=cv2.dilate(img,kernel,iterations=1)
eroded=cv2.erode(img,kernel,iterations=1)

plt.subplot(1,3,1)
plt.imshow(img,cmap="gray")
plt.title("original image")

plt.subplot(1,3,2)
plt.imshow(dilated,cmap="gray")
plt.title("dilated img")

plt.subplot(1,3,3)
plt.imshow(eroded,cmap="gray")
plt.title("Eroded img")

plt.show()

##exp 2.5 - Data augmentation

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array
)

# Step 1: Load sample image
image_path = "/content/image.jpg"

# Step 2: Resize the image
img = load_img(image_path, target_size=(224, 224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Step 3: Define augmentation techniques
augmentations = {
    "Rotation": ImageDataGenerator(rotation_range=60),
    "Flip": ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True
    ),
    "Shearing": ImageDataGenerator(shear_range=0.3),
    "Zooming": ImageDataGenerator(zoom_range=0.4)
}

# Step 4: Display original and augmented images
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

# Step 5: Generate each augmented image
for i, (name, generator) in enumerate(augmentations.items()):
    augmented = next(generator.flow(
        img_array,
        batch_size=1,
        shuffle=False
    ))

    plt.subplot(2, 3, i + 2)
    plt.imshow(augmented[0].astype("uint8"))
    plt.title(name)
    plt.axis("off")

plt.tight_layout()
plt.show()
