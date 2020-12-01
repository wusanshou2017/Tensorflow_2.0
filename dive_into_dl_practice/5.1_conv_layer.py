import tensorflow as tf
import numpy as np


def corr2d(X, K):
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j].assign(tf.cast(tf.reduce_sum(
                X[i:i + h, j:j + w] * K), dtype=tf.float32))
    return Y


X = tf.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = tf.constant([[0, 1], [2, 3]])
print(corr2d(X, K))


# 二维卷积层将输入和卷积核做互相关运算，并加上一个标量偏差来得到输出。卷积层的模型参数包括了卷积核和标量偏差。在训练模型的时候，通常我们先对卷积核随机初始化，然后不断迭代卷积核和偏差。
# 下面基于corr2d函数来实现一个自定义的二维卷积层。在构造函数__init__里我们声明weight和bias这两个模型参数。前向计算函数forward则是直接调用corr2d函数再加上偏差。
class Conv2D(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, kernel_size):
        self.w = self.add_weight(name='w',
                                 shape=kernel_size,
                                 initializer=tf.random_normal_initializer())
        self.b = self.add_weight(name='b',
                                 shape=(1,),
                                 initializer=tf.random_normal_initializer())

    def call(self, inputs):
        return corr2d(inputs, self.w) + self.b


X = tf.Variable(tf.ones((6, 8)))
X[:, 2:6].assign(tf.zeros(X[:, 2:6].shape))

print(X)

K = tf.constant([[1, -1]], dtype=tf.float32)
Y = corr2d(X, K)
print(Y)


# 二维卷积层使用4维输入输出，格式为(样本, 高, 宽, 通道)，这里批量大小（批量中的样本数）和通道数均为1
X = tf.reshape(X, (1, 6, 8, 1))
Y = tf.reshape(Y, (1, 6, 7, 1))

# 构造一个输出通道数为1（将在“多输入通道和多输出通道”一节介绍通道），核数组形状是(1, 2)的二维卷积层
conv2d = tf.keras.layers.Conv2D(1, (1, 2))
#input_shape = (samples, rows, cols, channels)
# Y = conv2d(X)

print(Y.shape)


# 以下 为卷积核 训练代码
Y_hat = conv2d(X)
for i in range(10):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(conv2d.weights[0])
        Y_hat = conv2d(X)
        l = (abs(Y_hat - Y)) ** 2
        dl = g.gradient(l, conv2d.weights[0])
        lr = 3e-2
        update = tf.multiply(lr, dl)
        updated_weights = conv2d.get_weights()
        updated_weights[0] = conv2d.weights[0] - update
        conv2d.set_weights(updated_weights)

        if (i + 1) % 2 == 0:
            print('batch %d, loss %.3f' % (i + 1, tf.reduce_sum(l)))


# check the kernel tensor

print(tf.reshape(conv2d.get_weights()[0], (1, 2)))
