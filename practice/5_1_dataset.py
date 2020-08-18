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

from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 定义一个从文件中读取图片的generator
image_generator = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
    "./data/cifar2/test/",
    target_size=(32, 32),
    batch_size=20,
    class_mode='binary')

classdict = image_generator.class_indices
print(classdict)


def generator():
    for features, label in image_generator:
        yield (features, label)


ds3 = tf.data.Dataset.from_generator(
    generator, output_types=(tf.float32, tf.int32))


plt.figure(figsize=(6, 6))
for i, (img, label) in enumerate(ds3.unbatch().take(9)):
    ax = plt.subplot(3, 3, i + 1)
    ax.imshow(img.numpy())
    ax.set_title("label = %d" % label)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
