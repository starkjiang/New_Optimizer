# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 20:11:27 2018

@author: Tony Jiang
"""

import os
from random import randint
import os.path
from keras.models import Model
from keras.layers import Flatten, Dense, Lambda
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
from keras.preprocessing import image

datadir = 'dataset/dogscats/'

for j in os.listdir(datadir):
    if ("jpg" in j):
        if ("dog" in j):
            os.rename(datadir + j, 'dataset/train/dogs/' + j)
        if ("cat" in j):
            os.rename(datadir + j, 'dataset/train/cats/' + j)

# randomly select some pics of cats and dogs for validation from the training set
for i in range(1000):
    while True:
        randy = randint(0, 12499)
        fname = 'dataset/train/dogs/dog.' + str(randy) + '.jpg'
        if os.path.isfile(fname):
            os.rename(fname, 'dataset/valid/dogs/dog.' + str(randy) + '.jpg')
            break

    while True:
        randy = randint(0, 12499)
        fname = 'dataset/train/cats/cat.' + str(randy) + '.jpg'
        if os.path.isfile(fname):
            os.rename(fname, 'dataset/valid/cats/cat.' + str(randy) + '.jpg')
            break

# set variables
gen = image.ImageDataGenerator()
batch_size = 64

# import training data
batches = gen.flow_from_directory('dataset/train',
                                  target_size=(224,224),
                                  class_mode='categorical',
                                  shuffle=True,
                                  batch_size=batch_size)

# import validation data
val_batches = gen.flow_from_directory('dataset/valid',
                                      target_size=(224,224),
                                      class_mode='categorical',
                                      shuffle=True,
                                      batch_size=batch_size)

# retrieve the full Keras VGG model including imagenet weights
vgg = VGG16(include_top=True, weights='imagenet',
                               input_tensor=None, input_shape=(224,224,3), pooling=None)

# set all layers to non-trainable
for layer in vgg.layers: layer.trainable=False

# define a new output layer to connect with the last fc layer in vgg
# thanks to joelthchao https://github.com/fchollet/keras/issues/2371
x = vgg.layers[-2].output
output_layer = Dense(2, activation='softmax', name='predictions')(x)

# combine the original VGG model with the new output layer
vgg2 = Model(inputs=vgg.input, outputs=output_layer)

# compile the new model
vgg2.compile(optimizer=Adam(lr=0.001),
                loss='categorical_crossentropy', metrics=['accuracy'])

vgg2.fit_generator(batches,
                   steps_per_epoch = batches.samples // batch_size,
                   validation_data = val_batches, 
                   validation_steps = val_batches.samples // batch_size,
                   epochs = 1)

