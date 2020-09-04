# encoding: utf-8

import tensorflow as tf

import re

assert(tf.__version__.startswith("2."))


class DataPreprocess():

    def preprocess_text(self, input_file):

        f = open(input_file, "r", encoding="utf-8")
        all_lines = f.readlines()
        eng_lines = [line.split("\t")[0] for line in all_lines]
        cn_lines = [line.split("\t")[1] for line in all_lines]
        eng_lines = ["<start> " +
                     self.preprocess_sentence(line) + " <end>" for line in eng_lines]
        cn_lines = ["<start> " +
                    " ".join(line) + " <end>" for line in cn_lines]
        return eng_lines, cn_lines

    def preprocess_sentence(self, w):

        w = w.lower().strip()
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)

        w = w.rstrip().strip()

        return w

    @staticmethod
    def filter_max_length(self, x, y, max_length=40):

        return tf.logical_and(tf.size(x) <= max_length,
                              tf.size(y) <= max_length)

    def tokenize(self, lang):

        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
            filters='')

        lang_tokenizer.fit_on_texts(lang)

        tensor = lang_tokenizer.texts_to_sequences(lang)

        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                               padding='post')

        return tensor, lang_tokenizer


if __name__ == '__main__':
    dp = DataPreprocess()
    eng_lines, cn_lines = dp.preprocess_text("cmn.txt")
    eng_tensor, eng_tokenizer = dp.tokenize(eng_lines)
    cn_tensor, cn_tokenizer = dp.tokenize(cn_lines)
    BUFF_SIZE = len(cn_tensor)
    BATCH_SIZE = 64
    MAX_LENGTH = 40

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (cn_tensor, eng_tensor)).shuffle(BUFF_SIZE)
    # train_dataset = train_dataset.filter(dp.filter_max_length)
    # 将数据集缓存到内存中以加快读取速度。
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(
        BUFF_SIZE).padded_batch(BATCH_SIZE, ([64], [None]))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    cn_batch, en_batch = next(iter(train_dataset))
    print(cn_batch)
    print(cn_batch.shape)

    print(en_batch)
    print(en_batch.shape)
