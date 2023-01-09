import numpy as np
import cv2 as cv
from sys import *
import time
from keras.models import load_model

cap = cv.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
model = load_model('Hand_Gestures_Recognition_final_model.h5')
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    img = cv.resize(frame,(224,224),fx=0,fy=0, interpolation = cv.INTER_CUBIC)
    # print(img.shape)
    img = img.reshape(1, 224, 224, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    result = model.predict(img, verbose = 0)
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    if round(result[0][0]) == 0:
        label = "1"
    else:
        label = "2"
    text = "activity: {}".format(label)
    cv.putText(frame, text, (35, 50), cv.FONT_HERSHEY_SIMPLEX,
      1.25, (0, 255, 0), 5)
    cv.imshow('frame', frame)
    
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()