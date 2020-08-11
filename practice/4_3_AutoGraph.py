# 1，被@tf.function修饰的函数应尽可能使用TensorFlow中的函数而不是Python中的其他函数。例如使用tf.print而不是print，使用tf.range而不是range，使用tf.constant(True)而不是True.

# 2，避免在@tf.function修饰的函数内部定义tf.Variable.

# 3，被@tf.function修饰的函数不可修改该函数外部的Python列表或字典等数据结构变量。
#
#
import numpy as np
import tensorflow as tf


# 1，被@tf.function修饰的函数应尽可能使用TensorFlow中的函数而不是Python中的其他函数。例如使用tf.print而不是print，使用tf.range而不是range，使用tf.constant(True)而不是True.
@tf.function
def np_random():
    a = np.random.randn(3, 3)
    tf.print(a)


@tf.function
def tf_random():
    a = tf.random.normal((3, 3))
    tf.print(a)


np_random()
np_random()

tf_random()
tf_random()


# 2，避免在@tf.function修饰的函数内部定义tf.Variable.

x = tf.Variable(1.0, dtype=tf.float32)


@tf.function
def outer_var():
    x.assign_add(1.0)
    tf.print(x)
    return(x)


outer_var()
outer_var()


@tf.function
def inner_var():
    x = tf.Variable(1.0, dtype=tf.float32)
    x.assign_add(1.0)
    tf.print(x)
    return(x)


# 执行将报错
# inner_var()
# inner_var()


# 3，被@tf.function修饰的函数不可修改该函数外部的Python列表或字典等数据结构变量。

tensor_list = []

#@tf.function #加上这一行切换成Autograph结果将不符合预期！！！


# @tf.function
def append_tensor(x):
    tensor_list.append(x)
    return tensor_list


append_tensor(tf.constant(5.0))
append_tensor(tf.constant(6.0))
print(tensor_list)
