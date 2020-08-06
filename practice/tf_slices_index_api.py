import tensorflow as tf
tf.random.set_seed(3)

t = tf.random.uniform([5, 5], minval=0, maxval=10, dtype=tf.int32)

tf.print(t)

tf.print(t[0])

tf.print(t[-1])

tf.print(t[1, 3])

tf.print(t[1][3])

tf.print(t[1:4, :])

tf.print(tf.slice(t, [1, 0], [3, 5]))

tf.print(t[1:4, :4:2])

x = tf.Variable([[1, 2], [3, 4]], dtype=tf.float32)

x[1, :].assign(tf.constant([0.0, 0.0]))

tf.print(x)

a = tf.random.uniform([3, 3, 3], minval=0, maxval=10, dtype=tf.int32)

tf.print(a)
