import tensorflow as tf
import numpy as numpy


def model_any(model, input_data, vocab_size, rnn_size=128, num_layer=2, batch_size=64, lr=0.001):
    ends_points = {}
    if model == "rnn":
        cell =
    elif model == "lstm":

    elif model == "gru":

    else:
        print("model_unknow...")
