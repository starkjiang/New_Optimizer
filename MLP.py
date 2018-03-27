# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 20:11:27 2018

@author: Tony Jiang
"""

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD, Adam, Adagrad, Adadelta, RMSprop, Nadam, Adamax
from keras.layers import Dense
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

# define a simple nn
def image_to_feature_vector(image, size=(32, 32)):
    # resize the image to a fixed size, then flatten the image into 
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset")
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the data matrix and labels list
data = []
labels = []

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
    # load the image and extract the class label (assuming
    # that our path as the format: /path/to/dataset/{class}.{image_num}.jpg)
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    
    # construct a feature vector raw pixel intensities, then update
    # the data matrix and label list
    features = image_to_feature_vector(image)
    data.append(features)
    labels.append(label)
    
    # show an upate every 1000 images
    if i > 0 and i % 1000 == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))
# encode the labels, converting them from string to integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# scale the input image pixels to the range [0,1], then transform the labels
# into vectors in the range [0, num_classes] -- this 
# generates a vector for each label where the index of the label
# is set to '1' and all other entries to '0'
data = np.array(data)/255.0
labels = np_utils.to_categorical(labels, 2)

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
print("[INFO] constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
    data, labels, test_size=0.25, random_state=42)

# Define the specific optimizers for the experiments, including SGD, NAG,
# Adam, AMSGrad, Adagrad, Adadelta, Adamax, Nadam, RMSprop
sgd = SGD(lr=0.01)
SGD = SGD(lr=0.01, momentum=0.99, decay=0.0, nesterov=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
ADAM = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
opt = ['sgd', 'SGD', 'adam', 'ADAM', 'Adagrad', 'Adamax', 'Nadam', 'RMSprop']
history_all = []
for i in opt:
    model = Sequential()
    model.add(Dense(768, input_dim=3072, init="uniform",
              activation="relu"))
    model.add(Dense(384, init="uniform", activation="relu"))
    model.add(Dense(2))
    model.add(Activation("softmax"))
    print("[INFO] compiling model...")
    model.compile(loss="binary_crossentropy", optimizer='{}'.format(i),
                  metrics=["accuracy"])
    history = model.fit(trainData, trainLabels, nb_epoch=50, batch_size=128)
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
# define the architecture of the network
#model = Sequential()
#model.add(Dense(768, input_dim=3072, init="uniform",
#    activation="relu"))
#model.add(Dense(384, init="uniform", activation="relu"))
#model.add(Dense(2))
#model.add(Activation("softmax"))

# train the model using SGD
#print("[INFO] compiling model...")
#model.compile(loss="binary_crossentropy", optimizer=adam,
#    metrics=["accuracy"])
# model.fit(trainData, trainLabels, nb_epoch=50, batch_size=128)

# show the accuracy on the testing set
#print("[INFO] evaluating on testing set...")
#(loss, accuracy) = model.evaluate(testData, testLabels,
#    batch_size=128, verbose=1)
#print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
