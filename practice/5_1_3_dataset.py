# -*- coding: utf-8 -*-

# 三，提升管道性能

# 训练深度学习模型常常会非常耗时。

# 模型训练的耗时主要来自于两个部分，一部分来自数据准备，另一部分来自参数迭代。

# 参数迭代过程的耗时通常依赖于GPU来提升。

# 而数据准备过程的耗时则可以通过构建高效的数据管道进行提升。

# 以下是一些构建高效数据管道的建议。

# 1，使用 prefetch 方法让数据准备和参数迭代两个过程相互并行。

# 2，使用 interleave 方法可以让数据读取过程多进程执行,并将不同来源数据夹在一起。

# 3，使用 map 时设置num_parallel_calls 让数据转换过程多进行执行。

# 4，使用 cache 方法让数据在第一个epoch后缓存到内存中，仅限于数据集不大情形。

# 5，使用 map转换时，先batch, 然后采用向量化的转换方法对每个batch进行转换。
#
#
import tensorflow as tf

# 打印时间分割线


@tf.function
def printbar():
    ts = tf.timestamp()
    today_ts = ts % (24 * 60 * 60)

    hour = tf.cast(today_ts // 3600 + 8, tf.int32) % tf.constant(24)
    minite = tf.cast((today_ts % 3600) // 60, tf.int32)
    second = tf.cast(tf.floor(today_ts % 60), tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}", m)) == 1:
            return(tf.strings.format("0{}", m))
        else:
            return(tf.strings.format("{}", m))

    timestring = tf.strings.join([timeformat(hour), timeformat(minite),
                                  timeformat(second)], separator=":")
    tf.print("==========" * 8, end="")
    tf.print(timestring)


import time

# 数据准备和参数迭代两个过程默认情况下是串行的。

# 模拟数据准备


def generator():
    for i in range(10):
        # 假设每次准备数据需要2s
        time.sleep(2)
        yield i


ds = tf.data.Dataset.from_generator(generator, output_types=(tf.int32))

# 模拟参数迭代


def train_step():
    # 假设每一步训练需要1s
    time.sleep(1)


# 训练过程预计耗时 10*2+10*1+ = 30s
printbar()
tf.print(tf.constant("start training..."))
for x in ds:
    train_step()
printbar()
tf.print(tf.constant("end training..."))


# 使用 prefetch 方法让数据准备和参数迭代两个过程相互并行。

# 训练过程预计耗时 max(10*2,10*1) = 20s
printbar()
tf.print(tf.constant("start training with prefetch..."))

# tf.data.experimental.AUTOTUNE 可以让程序自动选择合适的参数
for x in ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE):
    train_step()

printbar()
tf.print(tf.constant("end training..."))

# 2，使用 interleave 方法可以让数据读取过程多进程执行,并将不同来源数据夹在一起。

# ds_files = tf.data.Dataset.list_files("./data/titanic/*.csv")
# ds = ds_files.flat_map(lambda x: tf.data.TextLineDataset(x).skip(1))
# for line in ds.take(4):
#     print(line)

# 3，使用 map 时设置num_parallel_calls 让数据转换过程多进行执行。

ds = tf.data.Dataset.list_files("./data/cifar2/train/*/*.jpg")


def load_image(img_path, size=(32, 32)):
    label = 1 if tf.strings.regex_full_match(
        img_path, ".*/automobile/.*") else 0
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)  # 注意此处为jpeg格式
    img = tf.image.resize(img, size)
    return(img, label)


# 单进程转换
printbar()
tf.print(tf.constant("start transformation..."))

ds_map = ds.map(load_image)
for _ in ds_map:
    pass

printbar()
tf.print(tf.constant("end transformation..."))
# 多进程转换
printbar()
tf.print(tf.constant("start parallel transformation..."))

ds_map_parallel = ds.map(
    load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
for _ in ds_map_parallel:
    pass

printbar()
tf.print(tf.constant("end parallel transformation..."))


# 4，使用 cache 方法让数据在第一个epoch后缓存到内存中，仅限于数据集不大情形。

import time

# 模拟数据准备


def generator():
    for i in range(5):
        # 假设每次准备数据需要2s
        time.sleep(2)
        yield i


ds = tf.data.Dataset.from_generator(generator, output_types=(tf.int32))

# 模拟参数迭代


def train_step():
    # 假设每一步训练需要0s
    pass


# 训练过程预计耗时 (5*2+5*0)*3 = 30s
printbar()
tf.print(tf.constant("start training..."))
for epoch in tf.range(3):
    for x in ds:
        train_step()
    printbar()
    tf.print("epoch =", epoch, " ended")
printbar()
tf.print(tf.constant("end training..."))

import time

# 模拟数据准备


def generator():
    for i in range(5):
        # 假设每次准备数据需要2s
        time.sleep(2)
        yield i


# 使用 cache 方法让数据在第一个epoch后缓存到内存中，仅限于数据集不大情形。
ds = tf.data.Dataset.from_generator(generator, output_types=(tf.int32)).cache()

# 模拟参数迭代


def train_step():
    # 假设每一步训练需要0s
    time.sleep(0)


# 训练过程预计耗时 (5*2+5*0)+(5*0+5*0)*2 = 10s
printbar()
tf.print(tf.constant("start training..."))
for epoch in tf.range(3):
    for x in ds:
        train_step()
    printbar()
    tf.print("epoch =", epoch, " ended")
printbar()
tf.print(tf.constant("end training..."))

# 5，使用 map转换时，先batch, 然后采用向量化的转换方法对每个batch进行转换。

# 先map后batch
ds = tf.data.Dataset.range(100000)
ds_map_batch = ds.map(lambda x: x**2).batch(20)

printbar()
tf.print(tf.constant("start scalar transformation..."))
for x in ds_map_batch:
    pass
printbar()
tf.print(tf.constant("end scalar transformation..."))

# 先batch后map
ds = tf.data.Dataset.range(100000)
ds_batch_map = ds.batch(20).map(lambda x: x**2)

printbar()
tf.print(tf.constant("start vector transformation..."))
for x in ds_batch_map:
    pass
printbar()
tf.print(tf.constant("end vector transformation..."))
