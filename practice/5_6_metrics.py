import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, losses, metrics

# 函数形式的自定义评估指标


@tf.function
def ks(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1,))
    y_pred = tf.reshape(y_pred, (-1,))
    length = tf.shape(y_true)[0]
    t = tf.math.top_k(y_pred, k=length, sorted=False)
    y_pred_sorted = tf.gather(y_pred, t.indices)
    y_true_sorted = tf.gather(y_true, t.indices)
    cum_positive_ratio = tf.truediv(
        tf.cumsum(y_true_sorted), tf.reduce_sum(y_true_sorted))
    cum_negative_ratio = tf.truediv(
        tf.cumsum(1 - y_true_sorted), tf.reduce_sum(1 - y_true_sorted))
    ks_value = tf.reduce_max(tf.abs(cum_positive_ratio - cum_negative_ratio))
    return ks_value


y_true = tf.constant([[1], [1], [1], [0], [1], [1], [1],
                      [0], [0], [0], [1], [0], [1], [0]])
y_pred = tf.constant([[0.6], [0.1], [0.4], [0.5], [0.7], [0.7], [0.7],
                      [0.4], [0.4], [0.5], [0.8], [0.3], [0.5], [0.3]])
tf.print(ks(y_true, y_pred))

# 类形式的自定义评估指标
class KS(metrics.Metric):

    def __init__(self, name="ks", **kwargs):
        super(KS, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(
            name="tp", shape=(101,), initializer="zeros")
        self.false_positives = self.add_weight(
            name="fp", shape=(101,), initializer="zeros")

    @tf.function
    def update_state(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, (-1,)), tf.bool)
        y_pred = tf.cast(100 * tf.reshape(y_pred, (-1,)), tf.int32)

        for i in tf.range(0, tf.shape(y_true)[0]):
            if y_true[i]:
                self.true_positives[y_pred[i]].assign(
                    self.true_positives[y_pred[i]] + 1.0)
            else:
                self.false_positives[y_pred[i]].assign(
                    self.false_positives[y_pred[i]] + 1.0)
        return (self.true_positives, self.false_positives)

    @tf.function
    def result(self):
        cum_positive_ratio = tf.truediv(
            tf.cumsum(self.true_positives), tf.reduce_sum(self.true_positives))
        cum_negative_ratio = tf.truediv(
            tf.cumsum(self.false_positives), tf.reduce_sum(self.false_positives))
        ks_value = tf.reduce_max(
            tf.abs(cum_positive_ratio - cum_negative_ratio))
        return ks_value


y_true = tf.constant([[1], [1], [1], [0], [1], [1], [1],
                      [0], [0], [0], [1], [0], [1], [0]])
y_pred = tf.constant([[0.6], [0.1], [0.4], [0.5], [0.7], [0.7],
                      [0.7], [0.4], [0.4], [0.5], [0.8], [0.3], [0.5], [0.3]])

myks = KS()
myks.update_state(y_true, y_pred)
tf.print(myks.result())
