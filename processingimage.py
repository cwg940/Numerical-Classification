import numpy as np
import cv2

def image_processing(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image[50:350, 202:502]
    image = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
    image = image.astype(np.int16)

    maxcol = np.max(image)
    image = np.absolute(image - maxcol)

    threshold = (2*np.std(image))
    image[image < threshold] = 0
    image[image >= threshold] = maxcol

    proc_img = (image / maxcol).reshape(1,1,28,28)

    return proc_img
