import tensorflow as tf
import numpy as np

x = tf.Variable(0.0, name="x", dtype=tf.float32)

a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)

# 一阶导数  x=0 时的梯度
with tf.GradientTape() as tape:
    y = a * tf.pow(x, 2) + b * x + c

dy_dx = tape.gradient(y, x)
print(dy_dx)

with tf.GradientTape() as tape:
    tape.watch([a, b, c])
    y = a * tf.pow(x, 2) + b * x + c

dy_dx, dy_da, dy_db, dy_dc = tape.gradient(y, [x, a, b, c])

print(dy_dx)
print(dy_da)
print(dy_db)
print(dy_dc)


# 二阶导数

with tf.GradientTape() as tape2:
    with tf.GradientTape() as tape1:
        y = a * tf.pow(x, 2) + b * x + c

    dy_dx = tape1.gradient(y, x)

dy2_dx2 = tape2.gradient(dy_dx, x)

print(dy2_dx2)

# auto graph


@tf.function
def f(x):
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)

    # 自变量转换成tf.float32
    x = tf.cast(x, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = a * tf.pow(x, 2) + b * x + c
    dy_dx = tape.gradient(y, x)

    return((dy_dx, y))


tf.print(f(tf.constant(0.0)))
tf.print(f(tf.constant(1.0)))


# 求f(x) = a*x**2 + b*x + c的最小值
# 使用optimizer.apply_gradients

x = tf.Variable(0.0, name="x", dtype=tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for _ in range(1000):
    with tf.GradientTape() as tape:
        y = a * tf.pow(x, 2) + b * x + c
    dy_dx = tape.gradient(y, x)
    optimizer.apply_gradients(grads_and_vars=[(dy_dx, x)])

tf.print("y =", y, "; x =", x)


# 求f(x) = a*x**2 + b*x + c的最小值
# 使用optimizer.minimize
# optimizer.minimize相当于先用tape求gradient,再apply_gradient

x = tf.Variable(0.0, name="x", dtype=tf.float32)

# 注意f()无参数


def f():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a * tf.pow(x, 2) + b * x + c
    return(y)


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for _ in range(1000):
    optimizer.minimize(f, [x])

tf.print("y =", f(), "; x =", x)


# 在autograph中完成最小值求解
# 使用optimizer.apply_gradients

x = tf.Variable(0.0, name="x", dtype=tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)


@tf.function
def minimizef():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)

    for _ in tf.range(1000):  # 注意autograph时使用tf.range(1000)而不是range(1000)
        with tf.GradientTape() as tape:
            y = a * tf.pow(x, 2) + b * x + c
        dy_dx = tape.gradient(y, x)
        optimizer.apply_gradients(grads_and_vars=[(dy_dx, x)])

    y = a * tf.pow(x, 2) + b * x + c
    return y


tf.print("y=", minimizef(), ";x=", x)


# 在autograph中完成最小值求解
# 使用optimizer.minimize

x = tf.Variable(0.0, name="x", dtype=tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)


@tf.function
def f():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a * tf.pow(x, 2) + b * x + c
    return(y)


@tf.function
def train(epoch):
    for _ in tf.range(epoch):
        optimizer.minimize(f, [x])
    return(f())


tf.print(train(1000))
tf.print(x)
