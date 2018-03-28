# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 20:11:27 2018

@author: Tony Jiang

Thanks to ezchx https://github.com/ezchx/keras2_vgg_dogs_vs_cats
"""

import os
from random import randint
import os.path
from keras.models import Model
from keras.layers import Flatten, Dense, Lambda
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, Nadam, Adamax, RMSprop
from keras.preprocessing import image
import matplotlib.pyplot as plt

# Define the directory of raw dataset
datadir = 'dataset/dogscats/'

# Split the dataset into two datasets resepectively for dogs and cats for training
for j in os.listdir(datadir):
    if ("jpg" in j):
        if ("dog" in j):
            os.rename(datadir + j, 'dataset/train/dogs/' + j)
        if ("cat" in j):
            os.rename(datadir + j, 'dataset/train/cats/' + j)

# randomly select some pics of cats and dogs for validation from the training set
for i in range(500):
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

# set variables: the batch_size is set 64
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
for layer in vgg.layers:
    layer.trainable=False

# define a new output layer to connect with the last fc layer in vgg
x = vgg.layers[-2].output
output_layer = Dense(2, activation='softmax', name='predictions')(x)

# Define the specific optimizers for the experiments, including SGD, NAG,
# Adam, AMSGrad, Adagrad, Adadelta, Adamax, Nadam, RMSprop
sgd = SGD(lr=0.01)
SGD = SGD(lr=0.01, momentum=0.99, decay=0.0, nesterov=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
ADAM = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
opt = ['sgd', 'SGD', 'adam', 'ADAM', 'Adagrad', 'Adamax', 'Nadam', 'RMSprop']
history_all = []
for i in opt:
    # combine the original VGG model with the new output layer
    vgg2 = Model(inputs=vgg.input, outputs=output_layer)
    # compile the new model
    vgg2.compile(optimizer=Adam(lr=0.001),
                loss='categorical_crossentropy', metrics=['accuracy'])
    history = vgg2.fit_generator(batches,
                   steps_per_epoch = batches.samples // batch_size,
                   validation_data = val_batches, 
                   validation_steps = val_batches.samples // batch_size,
                   epochs = 10)
    history_all.append(history)
    
# plotting the training history
plt.plot(history_all[0].history['loss'], label='train_loss_SGD')
plt.plot(history_all[1].history['loss'], label='train_loss_NAG')
plt.plot(history_all[2].history['loss'], label='train_loss_Adam')
plt.plot(history_all[3].history['loss'], label='train_loss_AMSGrad')
plt.plot(history_all[4].history['loss'], label='train_loss_Adagrad')
plt.plot(history_all[5].history['loss'], label='train_loss_Adamax')
plt.plot(history_all[6].history['loss'], label='train_loss_Nadam')
plt.plot(history_all[7].history['loss'], label='train_loss_RMSprop')
plt.legend()
plt.show()
