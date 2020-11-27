import tensorflow as tf
import numpy as np
print(tf.__version__)

net = tf.keras.models.Sequential()
net.add(tf.keras.layers.Flatten())
net.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
net.add(tf.keras.layers.Dense(10))

X = tf.random.uniform((2, 20))
Y = net(X)

print(Y)


print(net.weights[0], type(net.weights[0]))


class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(
            units=10,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(
                mean=0, stddev=0.01),
            bias_initializer=tf.zeros_initializer()
        )
        self.d2 = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.ones_initializer(),
            bias_initializer=tf.ones_initializer()
        )

    def call(self, input):
        output = self.d1(input)
        output = self.d2(output)
        return output


net = Linear()
net(X)

# this example which show the weights in the model how to init
print(net.get_weights())


def my_init():
    return tf.keras.initializers.Ones()


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, kernel_initializer=my_init()))

Y = model(X)
print(model.weights[0])
