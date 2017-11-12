from picamera.array import PiRGBArray
from picamera import PiCamera
from theano.ifelse import ifelse
import numpy as np
import cv2
import time
from processingimage import *

from keras.models import load_model
model = load_model('KerasCNN.h5')

camera = PiCamera()
camera.resolution = (704, 400)
camera.framerate = 7
#camera.brightness = 60
#camera.contrast = 50
rawCapture = PiRGBArray(camera, camera.resolution)

time.sleep(0.1)

for frame in camera.capture_continuous(rawCapture, format = 'bgr'):
    image = frame.array
    proc_img = image_processing(image)
    nmb = model.predict_classes(proc_img)
    
    cv2.putText(image, str(nmb), (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    cv2.rectangle(image, (202, 50), (502, 350), (255, 0, 0), 1)
    
    cv2.imshow('Lense', image)
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    if key == ord('q'):
        break
