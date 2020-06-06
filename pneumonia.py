# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 14:42:57 2019

@author: Saif Mathur
"""

from keras.layers import Convolution2D 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Flatten

'''from keras import backend as K
K.set_image_dim_ordering('th')
'''


classifier = Sequential()
classifier.add(Convolution2D(32,3,3, border_mode='same' , input_shape=(64,64,1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2) ,border_mode='same'))


classifier.add(Convolution2D(32,3,3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2) ,border_mode='same'))

classifier.add(Flatten())



#no. of nodes 
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#image augmentation
from keras.preprocessing.image import ImageDataGenerator
training = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)
        

test = ImageDataGenerator(rescale = 1./255)

train_images = training.flow_from_directory(
         'chest_xray/train',
         target_size = (64,64),
         batch_size=32,
         class_mode = 'binary')

import numpy as np
train_images = np.expand_dims(train_images,axis=0)
test_images = np.expand_dims(train_images,axis=0)
         
test_images =  test.flow_from_directory(
        'chest_xray/test',
        target_size = (64,64),
        batch_size = 32,
        class_mode = 'binary')    
        
classifier.fit_generator(train_images,
                         samples_per_epoch = 5218, #images in training set
                         nb_epoch = 20,
                         validation_data = test_images,
                         nb_val_samples = 624) #images in test set

classifier.summary()