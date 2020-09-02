import numpy as np
import pandas as pd 
import tensorflow as tf 
assert(tf.__version__.startswith("2."))

# from tqdm import tqdm

from tensorflow.keras import *


train_token_path="./data/imdb/train_token.csv"
test_token_path="./data/imdb/test_token.csv"

MAX_WORDS =10000
MAX_LEN =200
BATCH_SIZE =20

def parse_line(line):
	t = tf.strings,split(lien,"\t")
	label =tf.reshape(tf.cast(tf.strings.to_number(t[0],tf.int32),(-1,)))
	return (features,label)