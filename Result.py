from keras.models import load_model
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
import numpy as np
import os
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from keras.layers import Dense , Dropout
from keras_preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize

classifier = Sequential()
sz = 128
# First convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Second convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# input_shape is going to be the pooled feature maps from the previous convolution layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(Flatten())

# Adding a fully connected layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=96, activation='relu'))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=3, activation='softmax')) # softmax for more than 2

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

classifier.load_weights('model-bw2.h5')
test_image = imread('test2/B/img260.jpg')

test_image = resize(test_image, (128,128,1))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('test2',
                                            target_size=(sz , sz),
                                            batch_size=10,
                                            color_mode='grayscale',
                                            class_mode='categorical')
test_set.class_indices
{'0': 0,
 'A': 1,
 'B': 2,
 'C': 3}
 # 'D': 4,
 # 'E': 5,
 # 'F': 6,
 # 'G': 7,
 # 'H': 8,
 # 'I': 9,
 # 'J': 10,
 # 'K': 11,
 # 'L': 12,
 # 'M': 13,
 # 'N': 14,
 # 'O': 15,
 # 'P': 16,
 # 'Q': 17,
 # 'R': 18,
 # 'S': 19,
 # 'T': 20,
 # 'U': 21,
 # 'V': 22,
 # 'W': 23,
 # 'X': 24,
 # 'Y': 25,
 #'Z': 26 }

test_results = classifier.predict_on_batch(test_image)
for category, value in test_set.class_indices.items():
            if value == test_results.argmax():
                print(category)




