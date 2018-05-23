import os
import json
import models
import numpy as np
from keras.utils import np_utils
from keras.datasets import cifar10, cifar100, mnist
from keras.optimizers import Adam, SGD, Adadelta, RMSprop, Adagrad, Nadam, Adamax
from Eve import Eve
from emvsgd import EMVSGD
from adv_emvsgd import AEMVSGD
from NAG_EMVSGD import EMVNAG
from NAG_AEMV import NAGAEMV
from WNAdam import WNAdam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_BACKEND'] = 'tensorflow'
print(os.environ['CUDA_VISIBLE_DEVICES'])
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def train(model_name, **kwargs):
    """
    Train model

    args: model_name (str, keras model name)
          **kwargs (dict) keyword arguments that specify the model hyperparameters
    """

    # Roll out the parameters
    batch_size = kwargs["batch_size"]
    nb_epoch = kwargs["nb_epoch"]
    dataset = kwargs["dataset"]
    optimizer = kwargs["optimizer"]
    experiment_name = kwargs["experiment_name"]

    # Compile model.
    if optimizer == "SGD":
        opt = SGD(lr=1E-2, decay=1E-4, momentum=0.9, nesterov=True)
    if optimizer == "Adam":
        opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    if optimizer == "Eve":
        opt = Eve(lr=1E-4, decay=1E-4, beta_1=0.9, beta_2=0.999, beta_3=0.999, small_k=0.1, big_K=10, epsilon=1e-08)
    if optimizer == "EMVSGD":
        opt = EMVSGD(lr=1.5E-3, decay=1E-4, beta_1=0.1, beta_2=0.999, epsilon=1e-08)
    if optimizer == "AEMVSGD":
        opt = AEMVSGD(lr=1E-3, beta_1=0.1, beta_2=0.999, beta_3=0.999, epsilon=None)
    if optimizer == "EMVNAG":
        opt = EMVNAG(lr=1E-2, beta_1=0.1, beta_2=0.999)
    if optimizer == "NAGAEMV":
        opt = NAGAEMV(lr=1E-3, beta_1=0.1, beta_2=0.999, beta_3=0.999, momentum=0.95, nesterov=True)
    if optimizer == "WNAdam":
        opt = WNAdam(lr=0.25, beta_1=0.9)
    if optimizer == "Adadelta":
        opt = Adadelta(lr=1.0, rho=0.95)
    if optimizer == "Adagrad":
        opt = Adagrad(lr=0.01)
    if optimizer == "RMSprop":
        opt = RMSprop(lr=0.001, rho=0.9)
    if optimizer == "Adamax":
        opt = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999)
    if optimizer == "Nadam":
        opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, schedule_decay=0.004)

    if dataset == "cifar10":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    if dataset == "cifar100":
        (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    if dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape((X_train.shape[0], 1, 28, 28))
        X_test = X_test.reshape((X_test.shape[0], 1, 28, 28))

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.
    X_test /= 255.

    img_dim = X_train.shape[-3:]
    nb_classes = len(np.unique(y_train))

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    # Compile model
    model = models.load(model_name, img_dim, nb_classes)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for e in range(nb_epoch):
        print(experiment_name)
        loss = model.fit(X_train, Y_train,
                         batch_size=batch_size,
                         validation_data=(X_test, Y_test),
                         epochs=1)

        train_losses.append(loss.history["loss"])
        val_losses.append(loss.history["val_loss"])
        train_accs.append(loss.history["acc"])
        val_accs.append(loss.history["val_acc"])
        print("epoch",(e+1),"is completed with following metrics:,loss:",loss.history["loss"],"accuracy:",loss.history["acc"],"val_loss",loss.history["val_loss"],"val_acc",loss.history["val_acc"])

        # Save experimental log
        d_log = {}
        d_log["experiment_name"] = experiment_name
        d_log["img_dim"] = img_dim
        d_log["batch_size"] = batch_size
        d_log["nb_epoch"] = nb_epoch
        d_log["train_losses"] = train_losses
        d_log["val_losses"] = val_losses
        d_log["train_accs"] = train_accs
        d_log["val_accs"] = val_accs
        d_log["optimizer"] = opt.get_config()
        # Add model architecture
        json_string = json.loads(model.to_json())
        for key in json_string.keys():
            d_log[key] = json_string[key]
        json_file = os.path.join("log", '%s_%s_%s.json' % (dataset, model.name, experiment_name))
        with open(json_file, 'w') as fp:
            json.dump(d_log, fp, indent=4, sort_keys=True)
