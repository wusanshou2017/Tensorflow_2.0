import tensorflow as tf
import random
from tensorflow import keras
from tensorflow.keras import backend as f
import numpy as np
import sys
import time


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


class RNNModel(keras.layers.Layer):
    def __init__(self, rnn_layer1, rnn_layer2, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn1 = rnn_layer1
        self.rnn2 = rnn_layer2
        self.vocab_size = vocab_size
        self.dense = keras.layers.Dense(vocab_size)

    def call(self, inputs, state):
        # 将输入转置成(num_steps, batch_size)后获取one-hot向量表示
        X = tf.one_hot(tf.transpose(inputs), self.vocab_size)
        Y, state = self.rnn1(X, state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        Y, state = self.rnn2(Y, state)
        output = self.dense(tf.reshape(Y, (-1, Y.shape[-1])))
        return output, state

def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = data_iter_random
    else:
        data_iter_fn = data_iter_consecutive
    params = get_params()
    #loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    for epoch in range(num_epochs):
        if not is_random_iter:  # 如使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps)
        for X, Y in data_iter:
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens)
            # else:  # 否则需要使用detach函数从计算图分离隐藏状态
                # for s in state:
                # s.detach()
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(params)
                inputs = to_onehot(X, vocab_size)
                # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
                (outputs, state) = rnn(inputs, state, params)
                # 拼接之后形状为(num_steps * batch_size, vocab_size)
                outputs = tf.concat(outputs, 0)
                # Y的形状是(batch_size, num_steps)，转置后再变成长度为
                # batch * num_steps 的向量，这样跟输出的行一一对应
                y = Y.T.reshape((-1,))
                # print(Y,y)
                y = tf.convert_to_tensor(y, dtype=tf.float32)
                # 使用交叉熵损失计算平均分类误差
                l = tf.reduce_mean(
                    tf.losses.sparse_categorical_crossentropy(y, outputs))
                #l = loss(y,outputs)
                # print("loss",np.array(l))

            grads = tape.gradient(l, params)
            grads = grad_clipping(grads, clipping_theta)  # 裁剪梯度
            optimizer.apply_gradients(zip(grads, params))
            # sgd(params, lr, 1 , grads)  # 因为误差已经取过均值，梯度不用再做平均
            l_sum += np.array(l).item() * len(y)
            n += len(y)

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            # print(params)
            for prefix in prefixes:
                print(prefix)
                print(' -', predict_rnn(
                    prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, idx_to_char, char_to_idx))


def train_and_predict_rnn_keras(model, num_hiddens, vocab_size,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes):
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_consecutive(
            corpus_indices, batch_size, num_steps)
        state = model.get_initial_state(
            batch_size=batch_size, dtype=tf.float32)
        for X, Y in data_iter:
            with tf.GradientTape(persistent=True) as tape:
                (outputs, state) = model(X, state)
                y = Y.T.reshape((-1,))
                l = loss(y, outputs)

            grads = tape.gradient(l, model.variables)
            # 梯度裁剪
            grads = grad_clipping(grads, clipping_theta)
            optimizer.apply_gradients(
                zip(grads, model.variables))  # 因为已经误差取过均值，梯度不用再做平均
            l_sum += np.array(l).item() * len(y)
            n += len(y)

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_keras(
                    prefix, pred_len, model, vocab_size, idx_to_char,
                    char_to_idx))


lr = 1e-4  # 注意调整学习率
num_hiddens = 256
num_steps = 35
num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['想', '不想']
lstm_layer = keras.layers.LSTM(
    num_hiddens, time_major=True, return_sequences=True, return_state=True)
lstm_layer2 = keras.layers.LSTM(
    num_hiddens, time_major=True, return_sequences=True, return_state=True)
model = RNNModel(lstm_layer, lstm_layer2, vocab_size)
train_and_predict_rnn_keras(model, num_hiddens, vocab_size,
                            corpus_indices, idx_to_char, char_to_idx,
                            num_epochs, num_steps, lr, clipping_theta,
                            batch_size, pred_period, pred_len, prefixes)
