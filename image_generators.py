#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import tensorflow as tf
import numpy as np


# folder paths - X_rays 
train_dir = 'COVID-19 Dataset/X-ray'

#image size 

img_height = 224
img_width = 224
batch_size = 32

# rescaling and augmentation
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, fill_mode='nearest',validation_split=0.2)

xr_train_dir = 'COVID-19 Dataset/X-Ray'

# creating generators
XR_train_generator = train_datagen.flow_from_directory(xr_train_dir, target_size=(224, 224),color_mode='rgb', 
                                                    batch_size=32, class_mode='categorical')

xr_validation_generator = train_datagen.flow_from_directory(
    xr_train_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

# folder paths - CT scans
ct_train_dir = 'COVID-19 Dataset/CT'

# rescaling and augmentation
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, fill_mode='nearest',validation_split=0.2)



# creating generators
ct_train_generator = train_datagen.flow_from_directory(ct_train_dir, target_size=(224, 224), color_mode='rgb',
                                                    batch_size=32, class_mode='categorical')


ct_validation_generator = train_datagen.flow_from_directory(
    ct_train_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')






