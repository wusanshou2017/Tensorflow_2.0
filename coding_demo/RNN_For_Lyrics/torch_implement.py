import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

import sys
sys.path.append("..")
import d2l_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print(torch.__version__, device)


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


(corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data()
lr = 1e-2
num_hiddens = 256

num_epochs, num_steps, batch_size, lr, clipping_theta = 1600, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['想', '不想']
gru_layer = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens)
model = d2l.RNNModel(gru_layer, vocab_size).to(device)
d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                  corpus_indices, idx_to_char, char_to_idx,
                                  num_epochs, num_steps, lr, clipping_theta,
                                  batch_size, pred_period, pred_len, prefixes)
