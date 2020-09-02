# encoding: utf-8

import tensorflow as tf

import re

assert(tf.__version__.startswith("2."))



tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples), target_vocab_size=2**13)

tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)

sample_string = 'Transformer is awesome.'

tokenized_string = tokenizer_en.encode(sample_string)
print ('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer_en.decode(tokenized_string)
print ('The original string: {}'.format(original_string))

assert original_string == sample_string


def preprocess_text(input_file):
<<<<<<< HEAD
    
=======
    f = open(input_file, "r", encoding="utf-8")
    all_lines = f.readlines()
    eng_lines = [line.split("\t")[0] for line in all_lines]
    cn_lines = [line.split("\t")[1] for line in all_lines]

    return eng_lines, cn_lines


eng_lines, cn_lines = preprocess_text("cmn.txt")
assert(len(eng_lines) == len(cn_lines))


def preprocess_sentence(w):

    w = w.lower().strip()
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # 除了 (a-z, A-Z, ".", "?", "!", ",")，将所有字符替换为空格
    # w = re.sub(r"[a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # 给句子加上开始和结束标记
    # 以便模型知道何时开始和结束预测
    return w


eng_lines = ["<start> " +
             preprocess_sentence(line) + " <end>" for line in eng_lines]
cn_lines = ["<start> " + " ".join(line) + " <end>" for line in cn_lines]

print(eng_lines[:10])
print(cn_lines[:10])


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')

    return tensor, lang_tokenizer


eng_tensor, eng_tokenizer = tokenize(eng_lines)
cn_tensor, cn_tokenizer = tokenize(cn_lines)
print(eng_tensor[:3])
print(cn_tensor[:3])


# def load_dataset(path, num_examples=None):
#     # 创建清理过的输入输出对
#     targ_lang, inp_lang = create_dataset(path, num_examples)

#     input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
#     target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

#     return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer
