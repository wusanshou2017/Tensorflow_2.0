import tensorflow as tf
import random
from tensorflow import keras
from tensorflow.keras import backend as f
import numpy as np
import sys
import time
from utils import *


def load_data():
    f = open("lyrics.txt", "r", encoding="utf-8")
    words_lst = f.readlines()

    words_lst = [item.replace("\n", "").replace("\r", "")
                 for item in words_lst]

    char_lst = [char for item in words_lst for char in item]

    idx_to_char = list(set(char_lst))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)

    # print(vocab_size)

    corpus_indices = [char_to_idx[char] for char in char_lst]
    sample = corpus_indices[:20]
    # print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
    # print('indices:', sample)
    return (corpus_indices, char_to_idx, idx_to_char, vocab_size)


def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    # 减1是因为输出的索引是相应输入的索引加1
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # 返回从pos开始的长为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield np.array(X, ctx), np.array(Y, ctx)


def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    corpus_indices = np.array(corpus_indices)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size * batch_len].reshape((
        batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y


(corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data()


def to_onehot(X, size):
    return [tf.one_hot(x, size, dtype=tf.float32) for x in X.T]


# class RNNModel(keras.layers.Layer):
#     def __init__(self, rnn_layer1, rnn_layer2, vocab_size, **kwargs):
#         super(RNNModel, self).__init__(**kwargs)
#         self.rnn1 = rnn_layer1
#         self.rnn2 = rnn_layer2
#         self.vocab_size = vocab_size
#         self.dense = keras.layers.Dense(vocab_size)

#     def call(self, inputs, state):
#         # 将输入转置成(num_steps, batch_size)后获取one-hot向量表示
#         X = tf.one_hot(tf.transpose(inputs), self.vocab_size)
#         Y, state = self.rnn1(X, state)
#         # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
#         # 形状为(num_steps * batch_size, vocab_size)
#         Y, state = self.rnn2(Y, state)
#         output = self.dense(tf.reshape(Y, (-1, Y.shape[-1])))
#         return output, state


lr = 1e-4  # 注意调整学习率
num_hiddens = 256
num_steps = 35


num_epochs, num_steps, batch_size, lr, clipping_theta = 360, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['想', '不想']
lstm_layer = keras.layers.LSTM(
    num_hiddens, time_major=True, return_sequences=True, return_state=True)
model = RNNModel(lstm_layer, vocab_size)
train_and_predict_rnn_keras(model, num_hiddens, vocab_size,
                            corpus_indices, idx_to_char, char_to_idx,
                            num_epochs, num_steps, lr, clipping_theta,
                            batch_size, pred_period, pred_len, prefixes)
