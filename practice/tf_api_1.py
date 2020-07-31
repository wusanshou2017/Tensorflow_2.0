import tensorflow as tf


# 样本数量
n = 400

# 生成测试用数据集
X = tf.random.uniform([n, 2], minval=-10, maxval=10)
w0 = tf.constant([[2.0], [-1.0]])
b0 = tf.constant(3.0)
# @表示矩阵乘法,增加正态扰动
Y = X @ w0 + b0 + tf.random.normal([n, 1], mean=0.0, stddev=2.0)


# 使用动态图调试

w = tf.Variable(tf.random.normal(w0.shape))
b = tf.Variable(0.0)


def train(epoches):
    for epoch in tf.range(1, epoches + 1):
        with tf.GradientTape() as tape:
            # 正向传播求损失
            Y_hat = X @ w + b
            loss = tf.squeeze(tf.transpose(Y - Y_hat) @
                              (Y - Y_hat)) / (2.0 * n)

        # 反向传播求梯度
        dloss_dw, dloss_db = tape.gradient(loss, [w, b])
        # 梯度下降法更新参数
        w.assign(w - 0.001 * dloss_dw)
        b.assign(b - 0.001 * dloss_db)
        if epoch % 1000 == 0:

            tf.print("epoch =", epoch, " loss =", loss,)
            tf.print("w =", w)
            tf.print("b =", b)
            tf.print("")


train(5000)


w = tf.Variable(tf.random.normal(w0.shape))
b = tf.Variable(0.0)


@tf.function
def train(epoches):
    for epoch in tf.range(1, epoches + 1):
        with tf.GradientTape() as tape:
            # 正向传播求损失
            Y_hat = X @ w + b
            loss = tf.squeeze(tf.transpose(Y - Y_hat) @
                              (Y - Y_hat)) / (2.0 * n)

        # 反向传播求梯度
        dloss_dw, dloss_db = tape.gradient(loss, [w, b])
        # 梯度下降法更新参数
        w.assign(w - 0.001 * dloss_dw)
        b.assign(b - 0.001 * dloss_db)
        if epoch % 1000 == 0:
            tf.print("epoch =", epoch, " loss =", loss,)
            tf.print("w =", w)
            tf.print("b =", b)
            tf.print("")


train(5000)
