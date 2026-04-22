# -*- coding: utf-8 -*-

# Install once in your terminal:
# python3 -m pip install tensor


## Loading packages
import numpy as np
import tensorflow as tf

# load MNIST data set
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

a = tf.keras.datasets.mnist.load_data()

# tf.keras: Implementation of the Keras API meant to be a high-level API for TensorFlow.
# loads mnist from keras.datasets


import matplotlib.pyplot as plt

def show(i,T=0):
    if (T == 0):
        print('Training image number',i,'is supposed to be:',y_train[i]) # The label of training data point is printed
        plt.imshow(x_train[i],cmap='Greys')
        plt.show(block=False)
    else:
        print('Test image number',i,'is supposed to be:',y_test[i]) # The label of training data point is printed
        plt.imshow(x_test[i],cmap='Greys')
        plt.show(block=False)
        
image_index = np.random.randint(60000)  # You may select anything up to 60,000
show(image_index)


# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) # 28 image columns and rows, 
# the last entry corresponds to the color chanels (1 for grey scale)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the RGB (red green blue) codes by dividing it to the max RGB value.
# For the pixels there are usually 256 color values [0-255], normalization to [0,1]
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape) #shape property giv
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])
y_train.shape

 

## This imports layer types.
from keras.models import Model
from keras.layers import Input, Dense, Flatten


## This defines an input for the model
inputs = Input(shape=input_shape)

## Here computations are applied. Dense() produces the actual trainable part, the other  
## stuff are mostly computations on the given values
## conv2D has some free parameters which will be trained as well
outputs = Flatten()(inputs)
outputs = Dense(1000, activation='relu')(outputs)
outputs = Dense(10,activation='softmax')(outputs)


model = Model(inputs=inputs, outputs=outputs)





## Model summary:
model.summary()



model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Training happens now:
model.fit(x=x_train,y=y_train, epochs=10, batch_size=100)

# Testing happens here:
x = model.evaluate(x_test, y_test)
print('Test result: [loss, accuracy]=',x)


## Searching for some images which have been predicted falsely
img_rows=28
img_cols=28

for i in range(100): # Prediction on 100 test data points
    image_index = np.random.randint(10000)
    pred = model.predict(x_test[image_index].reshape(1, img_rows, img_cols, 1),verbose=0)
    if pred.argmax() != y_test[image_index]: #show if prediction is not correct
        print(image_index)
        plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
        plt.show(block=False)
        print('Prediction',pred.argmax())
        print('Grand Truth',y_test[image_index])


model.count_params()



## Option part:
