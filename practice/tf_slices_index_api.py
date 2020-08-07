import tensorflow as tf
import numpy as np
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


x = tf.Variable([[1, 2], [3, 4]], dtype=tf.float32)
x[1, :].assign(tf.constant([0.0, 0.0]))
tf.print(x)

a = tf.random.uniform([3, 3, 3], minval=0, maxval=10, dtype=tf.int32)
tf.print(a)

# 省略号可以表示多个冒号
tf.print(a[..., 1])

scores = tf.random.uniform((4, 10, 7), minval=0, maxval=100, dtype=tf.int32)
tf.print(scores)


# 抽取每个班级第0个学生，第5个学生，第9个学生的全部成绩
p = tf.gather(scores, [0, 5, 9], axis=1)
tf.print(p)


# 抽取每个班级第0个学生，第5个学生，第9个学生的第1门课程，第3门课程，第6门课程成绩
q = tf.gather(tf.gather(scores, [0, 5, 9], axis=1), [1, 3, 6], axis=2)
tf.print(q)

s = tf.gather_nd(scores, indices=[(0, 0), (2, 4), (3, 6)])


# 抽取每个班级第0个学生，第5个学生，第9个学生的全部成绩
p = tf.boolean_mask(scores, [True, False, False, False, False,
                             True, False, False, False, True], axis=1)
tf.print(p)


s = tf.boolean_mask(scores,
                    [[True, False, False, False, False, False, False, False, False, False],
                     [False, False, False, False, False,
                         False, False, False, False, False],
                        [False, False, False, False, True,
                            False, False, False, False, False],
                        [False, False, False, False, False, False, True, False, False, False]])

tf.print(s)

# 利用tf.boolean_mask可以实现布尔索引

# 找到矩阵中小于0的元素
c = tf.constant([[-1, 1, -1], [2, 2, -2], [3, -3, 3]], dtype=tf.float32)
tf.print(c, "\n")

tf.print(tf.boolean_mask(c, c < 0), "\n")
tf.print(c[c < 0])  # 布尔索引，为boolean_mask的语法糖形式

# 找到张量中小于0的元素,将其换成np.nan得到新的张量
# tf.where和np.where作用类似，可以理解为if的张量版本

c = tf.constant([[-1, 1, -1], [2, 2, -2], [3, -3, 3]], dtype=tf.float32)
d = tf.where(c < 0, tf.fill(c.shape, np.nan), c)
tf.print(d)

# 如果where只有一个参数，将返回所有满足条件的位置坐标
indices = tf.where(c < 0)
tf.print(indices)
