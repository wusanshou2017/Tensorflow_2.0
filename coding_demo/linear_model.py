import tensorflow as tf 
from tensorflow import keras

assert (tf.__version__.startswith("2."))

train_file_path="../data/titanic/train.csv"
test_file_path="../data/titanic/test.csv"


LABEL_COLUMN ="Survived"
LABELS =[0,1]

def get_dataset(file_path,**kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=5, # Artificially small to make examples easier to show.
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,
      ignore_errors=True, 
      **kwargs)
    return dataset

raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)

def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key,value.numpy()))


show_batch(raw_train_data)

CSV_COLUMNS = ['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'class', 'deck', 'Embarked', 'alone']

temp_dataset = get_dataset(train_file_path, column_names=CSV_COLUMNS)

show_batch(temp_dataset)