import tensorflow as tf
import numpy as np

a = tf.constant([[1.0, 2], [-3, 4.0]])
b = tf.constant([[5.0, 6], [7.0, 8.0]])
tf.print(a + b)  # 运算符重载

tf.print(a - b)

a = tf.range(1, 10)
tf.print(tf.reduce_sum(a))
tf.print(tf.reduce_mean(a))
tf.print(tf.reduce_max(a))
tf.print(tf.reduce_min(a))
tf.print(tf.reduce_prod(a))

b = tf.reshape(a, (3, 3))
tf.print(tf.reduce_sum(b, axis=1, keepdims=True))
tf.print(tf.reduce_sum(b, axis=0, keepdims=True))

p = tf.constant([True, False, False])
q = tf.constant([False, False, True])
tf.print(tf.reduce_all(p))
tf.print(tf.reduce_any(q))

s = tf.foldr(lambda a, b: a + b, tf.range(10))
tf.print(s)


a = tf.range(1, 10)
tf.print(tf.math.cumsum(a))
tf.print(tf.math.cumprod(a))

tf.print(tf.math.cumsum(a))
tf.print(tf.math.cumsum(a))