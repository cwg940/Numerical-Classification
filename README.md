# Numerical Classification with Raspberry Pi 3, Keras, and OpenCV
## About
My idea for integer recognition always involved the Raspberry Pi to classify numbers via live feed with the PyCamera accessory, so I developed the code, and the choose the tools with that in mind. For the neural network, [Keras](https://keras.io/) is a terrific API for development, and allows the user to build models layer by layer for fast implementation. There are several other APIs available, but I found Keras to be both intuitive, but still gives the user much of the control. For Keras, it requires [TensorFlow](https://www.tensorflow.org/), [CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/), or [Theano](http://deeplearning.net/software/theano/) for the backend. I used Theano as it does not required 64-bit python, and can be implemented on the Pi's python version.

## Details
The program is broken into just two files: 'stream.py' and 'processingimage.py', and the model: 'KerasCNN.h5'
'stream.py' takes the image from the PiCamera, feeds the array of pixels to 'processingimage.py' which then converts the image to mimic the data that was used for training. Due to details such as light levels, and marker thinkness, the effectiveness of cleansing the image could differ environment to environment. 

![alt text](https://github.com/cwg940/Numerical-Classification/blob/master/image112017.jpg?raw=true)
