import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, losses, regularizers, constraints


tf.keras.backend.clear_session()

model = models.Sequential()

model.add(layers.Dense(64, input_dim=64,
                       kernel_regularizer=regularizers.l2(0.01),
                       activity_regularizer=regularizers.l1(0.01),
                       kernel_constraint=constraints.MaxNorm(max_value=2, axis=0)))

model.add(layers.Dense(10,
                       kernel_regularizer=regularizers.l1_l2(0.01, 0.01), activation="sigmoid"))

model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy", metrics=["AUC"])

model.summary()


def focal_loss(gamma=2., alpha=.25):

    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        loss = -tf.sum(alpha * tf.pow(1. - pt_1, gamma) * tf.log(1e-07 + pt_1)) \
            - tf.sum((1 - alpha) * tf.pow(pt_0, gamma)
                     * tf.log(1. - pt_0 + 1e-07))
        return loss
    return focal_loss_fixed

class FocalLoss(losses.Loss):

    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        loss = -tf.sum(self.alpha * tf.pow(1. - pt_1, self.gamma) * tf.log(1e-07 + pt_1)) \
            - tf.sum((1 - self.alpha) * tf.pow(pt_0, self.gamma)
                     * tf.log(1. - pt_0 + 1e-07))
        return loss
