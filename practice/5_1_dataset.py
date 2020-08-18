import tensorflow as tf
import numpy as np
from sklearn import datasets


# 1.从Numpy array 构建数据管道
iris = datasets.load_iris()

ds1 = tf.data.Dataset.from_tensor_slices((iris["data"], iris["target"]))

for features, label in ds1.take(5):
    print(features, label)


# 2.从Pandas DataFrame 构建数据管道

import pandas as pd

iris = datasets.load_iris()
dfiris = pd.DataFrame(iris["data"], columns=iris.feature_names)
ds2 = tf.data.Dataset.from_tensor_slices(
    (dfiris.to_dict("list"), iris["target"]))

for features, label in ds2.take(3):
    print(features, label)

# 3.从Python generator 构建数据管道

# from matplotlib import pyplot as plt
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # 定义一个从文件中读取图片的generator
# image_generator = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
#     "./data/cifar2/test/",
#     target_size=(32, 32),
#     batch_size=20,
#     class_mode='binary')

# classdict = image_generator.class_indices
# print(classdict)


# def generator():
#     for features, label in image_generator:
#         yield (features, label)


# ds3 = tf.data.Dataset.from_generator(
#     generator, output_types=(tf.float32, tf.int32))


# plt.figure(figsize=(6, 6))
# for i, (img, label) in enumerate(ds3.unbatch().take(9)):
#     ax = plt.subplot(3, 3, i + 1)
#     ax.imshow(img.numpy())
#     ax.set_title("label = %d" % label)
#     ax.set_xticks([])
#     ax.set_yticks([])
# plt.show()


# 4 从csv文件构建数据管道

ds4 = tf.data.experimental.make_csv_dataset(
    file_pattern=["./data/titanic/train.csv", "./data/titanic/test.csv"],
    batch_size=3,
    label_name="Survived",
    na_value="",
    num_epochs=1,
    ignore_errors=True)

for data, label in ds4.take(2):
    print(data, label)

# 5 .从文本文件构建数据管道
ds5 = tf.data.TextLineDataset(
    filenames=["./data/titanic/train.csv", "./data/titanic/test.csv"]
).skip(1)  # 略去第一行header

for line in ds5.take(5):
    print(line)

# 6. 从文件路径 构建数据管道
ds6 = tf.data.Dataset.list_files("./data/cifar2/train/*/*.jpg")
for file in ds6.take(5):
    print(file)

# 7 从tfrecords 文件构建数据管道

import os


def create_tfrecords(inpath, outpath):
    writer = tf.io.TFRecordWriter(outpath)
    dirs = os.listdir(inpath)
    for index, name in enumerate(dirs):
        class_path = inpath + "/" + name + "/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = tf.io.read_file(img_path)
            #img = tf.image.decode_image(img)
            # img = tf.image.encode_jpeg(img) #统一成jpeg格式压缩
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.numpy()]))
                }))
            writer.write(example.SerializeToString())
    writer.close()


create_tfrecords("./data/cifar2/test/", "./data/cifar2_test.tfrecords/")

from matplotlib import pyplot as plt


def parse_example(proto):
    description = {'img_raw': tf.io.FixedLenFeature([], tf.string),
                   'label': tf.io.FixedLenFeature([], tf.int64)}
    example = tf.io.parse_single_example(proto, description)
    img = tf.image.decode_jpeg(example["img_raw"])  # 注意此处为jpeg格式
    img = tf.image.resize(img, (32, 32))
    label = example["label"]
    return(img, label)


ds7 = tf.data.TFRecordDataset(
    "./data/cifar2_test.tfrecords").map(parse_example).shuffle(3000)

%matplotlib inline
%config InlineBackend.figure_format = 'svg'
plt.figure(figsize=(6, 6))
for i, (img, label) in enumerate(ds7.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    ax.imshow((img / 255.0).numpy())
    ax.set_title("label = %d" % label)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
