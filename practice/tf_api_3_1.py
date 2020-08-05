import tensorflow as tf
assert (tf.__version__.startswith("2."))
from tensorflow.keras import losses, layers, metrics, optimizers, models

n = 800

X = tf.random.uniform([n, 2], minval=-10, maxval=10)
w0 = tf.constant([[2.0], [-1.0]])
b0 = tf.constant(3.0)

# @表示矩阵乘法,增加正态扰动
Y = X @ w0 + b0 + tf.random.normal([n, 1], mean=0.0, stddev=2.0)

tf.keras.backend.clear_session()

linear = models.Sequential()
linear.add(layers.Dense(1, input_shape=(2,)))
linear.summary()

# 使用fit方法进行训练

linear.compile(optimizer="adam", loss="mse", metrics=["mse"])
linear.fit(X, Y, batch_size=20, epochs=200)

tf.print("w = ", linear.layers[0].kernel)
tf.print("b = ", linear.layers[0].bias)
