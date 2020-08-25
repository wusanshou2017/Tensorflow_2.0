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
