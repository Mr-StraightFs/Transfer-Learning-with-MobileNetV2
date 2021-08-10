import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops

# Load the Data and Split the Data into Train/Test Sets
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_happy_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

# Vizualize a random image
index = 10
plt.imshow(X_train_orig[index]) #display sample training image
plt.show()


#Create the Sequential Model
# The TensorFlow Keras Sequential API can be used to build simple models with layer operations that proceed in a sequential order.

def happyModel():
    """
        Implements the forward propagation for the binary classification model:
        ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE

        Note that for simplicity and grading purposes, you'll hard-code all the values
        such as the stride and kernel (filter) sizes.
        Normally, functions should take these values as function parameters.

        Arguments:
        None

        Returns:
        model -- TF Keras model (object containing the information for the entire training process)
        """

    model = tf.keras.Sequential([
        # ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
        tf.keras.layers.ZeroPadding2D(padding=(3, 3), input_shape=(64, 64, 3)),

        ## Conv2D with 32 7x7 filters and stride of 1
        tf.keras.layers.Conv2D(32, (7, 7), strides=(1, 1)),

        ## BatchNormalization for axis 3
        tf.keras.layers.BatchNormalization(axis=3),
        ## ReLU
        tf.keras.layers.ReLU(),
        ## Max Pooling 2D with default parameters
        tf.keras.layers.MaxPooling2D((2, 2), name='max_pool'),
        ## Flatten layer
        tf.keras.layers.Flatten(),
        ## Dense layer with 1 unit for output & 'sigmoid' activation
        tf.keras.layers.Dense(1, activation='sigmoid')

    ])

    return model

# Compile the Model
happy_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

# Model Summary
happy_model.summary()

# Train and Evaluate the Model
happy_model.fit(X_train, Y_train, epochs=8, batch_size=16)



