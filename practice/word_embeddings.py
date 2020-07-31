import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds

embedding_layer = layers.Embedding(1000, 5)
result = embedding_layer(tf.constant([1, 2, 3]))
print(result.numpy())
