import tensorflow as tf
import numpy as np

import numpy as np

x = tf.ones(3)
print(x)


# read one column and save to memory which in cpu

y = tf.zeros(4)
np.save('xy.npy', [x, y])
x2, y2 = np.load('xy.npy', allow_pickle=True)
print(x2, y2)


X = tf.random.normal((2, 20))
print(X)

mydict = {'x': x, 'y': y}
np.save('mydict.npy', mydict)
mydict2 = np.load('mydict.npy', allow_pickle=True)
print(mydict2)

X = tf.random.normal((2, 20))
print(X)


class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.dense3(x)
        return output


net = MLP()

out = net(X)
print(out)
