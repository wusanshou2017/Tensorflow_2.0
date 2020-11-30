import tensorflow as tf
import numpy as np
print(tf.__version__)

# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')

print(cpus)

# check availble device
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
