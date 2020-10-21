#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import the required libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
from shutil import copyfile
import random

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import tensorflow as tf
print("Tensorflow version " + tf.__version__)


# In[ ]:


import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg

model = tf.keras.models.load_model('mymodel_mobiletnet.h5')

labels_dict={0:'No Mask ! Please wear your mask!',1:'Wrong mask! It protects you from COVID-19',2:'Yes Mask! Perfect!'}
color_dict={0:(255,10,10),1:(0,0,255),2:(0,255,0)}

size = 4
webcam = cv2.VideoCapture(0) #Use camera 0

# We load the xml file
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    (rval, im) = webcam.read()
    im=cv2.flip(im,1,1) #Flip to act as a mirror

    # Resize the image to speed up detection
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    # detect MultiScale / faces 
    faces = classifier.detectMultiScale(mini)

    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
        #Save just the rectangle faces in SubRecFaces
        face_img = im[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(224,224))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,224,224,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        
        label=np.argmax(result,axis=1)[0]
        (nomask,wrongmask,yesmask) = model.predict(reshaped)[0]
        #print(label)
      
    
        label2 = "{}: {:.2f}%".format(labels_dict[label], max(nomask,wrongmask,yesmask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(im, label2, (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, color_dict[label], 2)

        
    # Show the image
    cv2.imshow('LIVE',   im)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop 
    if key == 27: #The Esc key
        break
# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()

