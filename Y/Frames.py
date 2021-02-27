import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import time
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from skimage.io import imread, imshow
from skimage.transform import resize
from tkinter import *

cap = cv2.VideoCapture(0)
# classifier = Sequential()
# sz = 128
# # First convolution layer and pooling
# classifier.add(Convolution2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# # Second convolution layer and pooling
# classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# # input_shape is going to be the pooled feature maps from the previous convolution layer
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
#
# # Flattening the layers
# classifier.add(Flatten())
#
# # Adding a fully connected layer
# classifier.add(Dense(units=128, activation='relu'))
# classifier.add(Dropout(0.40))
# classifier.add(Dense(units=96, activation='relu'))
# classifier.add(Dropout(0.40))
# classifier.add(Dense(units=64, activation='relu'))
# classifier.add(Dense(units=27, activation='softmax')) # softmax for more than 2
#
# classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# classifier.load_weights('model-bw.h5')
# print(classifier.summary())
backSub = cv2.createBackgroundSubtractorKNN()
# list = []
i=0

while(True):
    ret, frame = cap.read()
    image = cv2.flip(frame,1)
    start_point = (250, 150)
    end_point = (500, 400)
    color = (0, 255, 0)
    thickness = 2
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
    y2 = image[150:401, 250:501]
    y = resize(y2, (128, 128))
    #y2 = resize(y, (250, 250))
    fgMask = backSub.apply(y2)
    cv2.imwrite('img' + str(i) + '.jpg', fgMask)
    i += 1
    #y2 = y2.astype('uint8')
    #y = cv2.cvtColor(y,cv2.COLOR_BGR2GRAY)
    img1 = "Original"
    cv2.namedWindow(img1)
    cv2.moveWindow(img1, 140,130)

    img2 = "Cropped and Sized"
    cv2.namedWindow(img2)
    cv2.moveWindow(img2, 1040,130)

    # newImg = np.reshape(y,[3,128,128,1])
    # prediction = classifier.predict(newImg)
    # print(prediction)
    # mappings = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: 'J', 10: "K", 11: "L",
    #             12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T", 20: "U", 21: "V", 22: "W",
    #             23: "X", 24: "Y", 25: 'Z', 26: ' DEL ', 27: '', 28: '  '}
    # index = 0
    # for i in range(len(prediction[0])):
    #      if prediction[0][i]==1:
    #         index = i
    #         break
    # list.append(mappings[index])
    #
    # def listToString(s):
    #
    #     # initialize an empty string
    #     str = ""
    #
    #     # traverse in the string
    #     for ele in s:
    #         str += ele
    #
    #         # return string
    #     return str
    #
    # string = listToString(list)
    #
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # org = (50, 50)
    # fontScale = 2
    # color = (255, 0, 0)
    # thickness = 2
    # image = cv2.putText(image, string, org, font,
    #                 fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow(img1, image)
    cv2.imshow(img2, fgMask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
# import cv2
# from skimage.transform import resize
# import numpy as np
# backSub = cv2.createBackgroundSubtractorKNN()
# import matplotlib.pyplot as plt
# from skimage.io import imread, imshow
#
#
# capture = cv2.VideoCapture(0)
# i = 0
# while (capture.isOpened()):
#     ret, frame = capture.read()
#     if ret == False:
#         break
#
#     cv2.imwrite('img' + str(i) + '.jpg', fgMask)
#     i += 1
#
# capture.release()
# cv2.destroyAllWindows()