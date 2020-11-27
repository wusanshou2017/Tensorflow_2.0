import tensorflow as tf
import numpy as np
print(tf.__version__)
assert(tf.__version__.startswith("2."))

class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()    # Flatten层将除第一维（batch_size）以外的维度展平
        self.dense1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        print(x.shape)
        x = self.dense1(x)
        output = self.dense2(x)
        return output


X = tf.random.uniform((2, 20))
net = MLP()

print(net(X))


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])


model.build([2, 20])
print(model.summary())
print(model(X))


class FancyMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.flatten = tf.keras.layers.Flatten()
        self.rand_weight = tf.constant(
            tf.random.uniform((20, 20)))
        self.dense = tf.keras.layers.Dense(units=20, activation=tf.nn.relu)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = tf.nn.relu(tf.matmul(x, self.rand_weight) + 1)
        x = self.dense(x)
        while tf.norm(x) > 1:
            x /= 2
        if tf.norm(x) < 0.8:
            x *= 10
        return tf.reduce_sum(x)


net = FancyMLP()
net(X)


class NestMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential()
        self.net.add(tf.keras.layers.Flatten())
        self.net.add(tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu))
        self.net.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
        self.dense = tf.keras.layers.Dense(units=16, activation=tf.nn.relu)

    def call(self, inputs):
        return self.dense(self.net(inputs))


net = tf.keras.Sequential()
net.add(NestMLP())
net.add(tf.keras.layers.Dense(20))
net.add(FancyMLP())

net(X)
