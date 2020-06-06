# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 22:05:33 2019

@author: Saif Mathur
"""

import tensorflow as tf
classifier = tf.keras.models.Sequential()
classifier.add(tf.keras.layers.Conv2D(32,3,3, padding='same' ,data_format="channels_last",
                                      input_shape=(64,64,3),
                                      activation = 'relu'))
classifier.add(tf.keras.layers.AveragePooling2D(pool_size = (2,2)))


#second layer
classifier.add(tf.keras.layers.Conv2D(32,3,3, padding='same',
                                      activation = 'relu'))
classifier.add(tf.keras.layers.AveragePooling2D(pool_size = (2,2)))


classifier.add(tf.keras.layers.Flatten())



#no. of nodes 
classifier.add(tf.keras.layers.Dense(units = 128 , activation = 'relu'))
classifier.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



#image augmentation
training = tf.keras.preprocessing.ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)
        
import tensorflow as tf 

test = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)


train_images = training.flow_from_directory(
         'chest_xray/train',
         target_size = (64,64),
         batch_size=32,
         class_mode = 'binary')
      



test_images =  test.flow_from_directory(
        'chest_xray/test',
        target_size = (64,64),
        batch_size = 32,
        class_mode = 'binary')    
        
classifier.fit_generator(train_images,
                         steps_per_epoch = 5218, #images in training set
                         epochs = 10,
                         validation_data = test_images,
                         validation_steps = 624) #images in test set


classifier.summary()

classifier.save('pneumonia.h5')
del classifier





import tensorflow as tf 
test = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
test_images =  test.flow_from_directory(
        'chest_xray/test',
        target_size = (64,64),
        batch_size = 32,
        class_mode = 'binary')    

new_model = tf.keras.models.load_model('pneumonia.h5')
predictions = new_model.predict(test_images)

test_results = []

def predict():
    for i in predictions:
        if i > 0.5:
            test_results.append('Person has pneumonia')
        else:
            test_results.append('Person does not have pneumonia')
            
predict()
        
        
        
        



    

