# 维度变换相关函数主要有 tf.reshape, tf.squeeze, tf.expand_dims, tf.transpose.

# tf.reshape 可以改变张量的形状。

# tf.squeeze 可以减少维度。

# tf.expand_dims 可以增加维度。

# tf.transpose 可以交换维度。

# tf.reshape可以改变张量的形状，但是其本质上不会改变张量元素的存储顺序，所以，该操作实际上非常迅速，并且是可逆的。

import tensorflow as tf
assert (tf.__version__.startswith("2."))


a = tf.random.uniform(shape=[1, 3, 3, 2],
                      minval=0, maxval=255, dtype=tf.int32)
tf.print(a.shape)
tf.print(a)

# 改成 （3,6）形状的张量
b = tf.reshape(a, [3, 6])
tf.print(b.shape)
tf.print(b)


# 改回成 [1,3,3,2] 形状的张量
c = tf.reshape(b, [1, 3, 3, 2])
tf.print(c)

# 如果张量在某个维度上只有一个元素，利用tf.squeeze可以消除这个维度。

# 和tf.reshape相似，它本质上不会改变张量元素的存储顺序。

# 张量的各个元素在内存中是线性存储的，其一般规律是，同一层级中的相邻元素的物理地址也相邻。
s = tf.squeeze(a)
tf.print(s.shape)
tf.print(s)

d = tf.expand_dims(s, axis=0)  # 在第0维插入长度为1的一个维度
tf.print(d)


a = tf.random.uniform(shape=[100, 600, 600, 4],
                      minval=0, maxval=255, dtype=tf.int32)
tf.print(a.shape)

# 转换成 Channel,Height,Width,Batch
s = tf.transpose(a, perm=[3, 1, 2, 0])
tf.print(s.shape)


# 和numpy类似，可以用tf.concat和tf.stack方法对多个张量进行合并，可以用tf.split方法把一个张量分割成多个张量。

# tf.concat和tf.stack有略微的区别，tf.concat是连接，不会增加维度，而tf.stack是堆叠，会增加维度。

a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
c = tf.constant([[9.0, 10.0], [11.0, 12.0]])

d = tf.concat([a, b, c], axis=0)

tf.print(d)


# tf.split是tf.concat的逆运算，可以指定分割份数平均分割，也可以通过指定每份的记录数量进行分割。
e, d, f = tf.split(d, 3, axis=0)  # 指定分割份数，平均分割
tf.print(e)
