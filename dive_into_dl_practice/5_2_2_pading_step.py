import tensorflow as tf
import numpy as np


def comp_conv2d(conv2d, X):
    X = tf.reshape(X, (1,) + X.shape + (1,))
    Y = conv2d(X)
    #input_shape = (samples, rows, cols, channels)
    return tf.reshape(Y, Y.shape[1:3])


conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same')
X = tf.random.uniform(shape=(8, 8))
print(comp_conv2d(conv2d, X).shape)


conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', strides=2)
print(comp_conv2d(conv2d, X).shape)

conv2d = tf.keras.layers.Conv2D(1, kernel_size=(
    3, 5), padding='valid', strides=(3, 4))
print(comp_conv2d(conv2d, X).shape)
