import tensorflow as tf
import random


def load_data():
    f = open("lyrics.txt", "r", encoding="utf-8")
    words_lst = f.readlines()

    words_lst = [item.replace("\n", "").replace("\r", "")
                 for item in words_lst]

    char_lst = [char for item in words_lst for char in item]

    idx_to_char = list(set(char_lst))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)

    print(vocab_size)

    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    sample = corpus_indices[:20]
    print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
    print('indices:', sample)
