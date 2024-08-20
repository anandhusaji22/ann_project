from keras.datasets import mnist
from keras.utils import to_categorical
from keras import optimizers
from keras import metrics

import pandas as pd

def photo_add(photo):
    """Add a photo to the database."""
    




def training_data():
    

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((60000, 784))
    x_train = x_train.astype('float32') / 255

    x_test = x_test.reshape((10000, 784))
    x_test = x_test.astype('float32') / 255


    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    #Create Neural Network Model

    from keras import models        #To define type of model - Sequential/Functional
    from keras import layers         #To define type of layers

    model = models.Sequential()

    model.add(layers.Dense(100 , activation='relu' , input_dim=x_train.shape[1]))    #To add hidden layer 1
    model.add(layers.Dense(50 , activation='relu'))
    model.add(layers.Dense(10 , activation='sigmoid'))                                #To add layer 2

    model.summary()


    xtrainC = x_train/x_train.max()
    xtestC = x_test/x_test.max()


    #import tensorflow as tf



#sgd = tf.keras.optimizers.SGD(0.01)

    model.compile(optimizer = 'sgd',loss = 'categorical_crossentropy',metrics=['accuracy'])
    

    model.fit(xtrainC,y_train,
          epochs = 40 , validation_data=(xtestC,y_test))
    
    c=model.evaluate(xtestC,y_test)
    print(c)
