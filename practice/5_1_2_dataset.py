# Dataset数据结构应用非常灵活，因为它本质上是一个Sequece序列，其每个元素可以是各种类型，例如可以是张量，列表，字典，也可以是Dataset。

# Dataset包含了非常丰富的数据转换功能。

# map: 将转换函数映射到数据集每一个元素。

# flat_map: 将转换函数映射到数据集的每一个元素，并将嵌套的Dataset压平。

# interleave: 效果类似flat_map,但可以将不同来源的数据夹在一起。

# filter: 过滤掉某些元素。

# zip: 将两个长度相同的Dataset横向铰合。

# concatenate: 将两个Dataset纵向连接。

# reduce: 执行归并操作。

# batch : 构建批次，每次放一个批次。比原始数据增加一个维度。 其逆操作为unbatch。

# padded_batch: 构建批次，类似batch, 但可以填充到相同的形状。

# window :构建滑动窗口，返回Dataset of Dataset.

# shuffle: 数据顺序洗牌。

# repeat: 重复数据若干次，不带参数时，重复无数次。

# shard: 采样，从某个位置开始隔固定距离采样一个元素。

# take: 采样，从开始位置取前几个元素。

import tensorflow as tf

# map:将转换函数映射到数据集每一个元素
ds = tf.data.Dataset.from_tensor_slices(
    ["hello world", "hello China", "hello Beijing"])

ds_map = ds.map(lambda x: tf.strings.split(x, " "))
for x in ds_map:
    tf.print(x)


# flat_map:将转换函数映射到数据集的每一个元素，并将嵌套的Dataset压平。

ds = tf.data.Dataset.from_tensor_slices(
    ["hello world", "hello China", "hello Beijing"])
ds_flatmap = ds.flat_map(
    lambda x: tf.data.Dataset.from_tensor_slices(tf.strings.split(x, " ")))
for x in ds_flatmap:
    print(x)


# interleave: 效果类似flat_map,但可以将不同来源的数据夹在一起。

ds = tf.data.Dataset.from_tensor_slices(
    ["hello world", "hello China", "hello Beijing"])
ds_interleave = ds.interleave(
    lambda x: tf.data.Dataset.from_tensor_slices(tf.strings.split(x, " ")))
for x in ds_interleave:
    print(x)


# filter:过滤掉某些元素。

ds = tf.data.Dataset.from_tensor_slices(
    ["hello world", "hello China", "hello Beijing"])
# 找出含有字母a或B的元素
ds_filter = ds.filter(lambda x: tf.strings.regex_full_match(x, ".*[a|B].*"))
for x in ds_filter:
    print(x)


# zip:将两个长度相同的Dataset横向铰合。

ds1 = tf.data.Dataset.range(0, 3)
ds2 = tf.data.Dataset.range(3, 6)
ds3 = tf.data.Dataset.range(6, 9)
ds_zip = tf.data.Dataset.zip((ds1, ds2, ds3))
for x, y, z in ds_zip:
    print(x.numpy(), y.numpy(), z.numpy())


ds1 = tf.data.Dataset.range(0, 3)
ds2 = tf.data.Dataset.range(3, 6)
ds_concat = tf.data.Dataset.concatenate(ds1, ds2)
for x in ds_concat:
    print(x)


# reduce:执行归并操作。

ds = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5.0])
result = ds.reduce(0.0, lambda x, y: tf.add(x, y))
print(result)


# batch:构建批次，每次放一个批次。比原始数据增加一个维度。 其逆操作为unbatch。

ds = tf.data.Dataset.range(12)
ds_batch = ds.batch(4)
for x in ds_batch:
    print(x)


# padded_batch:构建批次，类似batch, 但可以填充到相同的形状。

elements = [[1, 2], [3, 4, 5], [6, 7], [8]]
ds = tf.data.Dataset.from_generator(lambda: iter(elements), tf.int32)

ds_padded_batch = ds.padded_batch(2, padded_shapes=[4, ])
for x in ds_padded_batch:
    print(x)


# window:构建滑动窗口，返回Dataset of Dataset.

ds = tf.data.Dataset.range(12)
# window返回的是Dataset of Dataset,可以用flat_map压平
ds_window = ds.window(3, shift=1).flat_map(
    lambda x: x.batch(3, drop_remainder=True))
for x in ds_window:
    print(x)


# shuffle:数据顺序洗牌。

ds = tf.data.Dataset.range(12)
ds_shuffle = ds.shuffle(buffer_size=5)
for x in ds_shuffle:
    print(x)


# shard:采样，从某个位置开始隔固定距离采样一个元素。

ds = tf.data.Dataset.range(12)
ds_shard = ds.shard(3, index=1)

for x in ds_shard:
    print(x)


# take:采样，从开始位置取前几个元素。

ds = tf.data.Dataset.range(12)
ds_take = ds.take(3)

for num in ds_take:
    print(num)
