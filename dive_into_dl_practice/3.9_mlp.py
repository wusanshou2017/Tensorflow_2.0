import tensorflow as tf 

import numpy as np

from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
batch_size = 256
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)
x_train = x_train/255.0
x_test = x_test/255.0
train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = tf.Variable(tf.random.normal(shape=(num_inputs, num_hiddens),mean=0, stddev=0.01, dtype=tf.float32))
b1 = tf.Variable(tf.zeros(num_hiddens, dtype=tf.float32))
W2 = tf.Variable(tf.random.normal(shape=(num_hiddens, num_outputs),mean=0, stddev=0.01, dtype=tf.float32))
b2 = tf.Variable(tf.random.normal([num_outputs], stddev=0.1))


def relu(x):
    return tf.math.maximum(x,0)

def net(X):
    X = tf.reshape(X, shape=[-1, num_inputs])
   
    h = relu(tf.matmul(X, W1) + b1)
    return tf.math.softmax(tf.matmul(h, W2) + b2)

def loss(y_hat,y_true):
    return tf.losses.sparse_categorical_crossentropy(y_true,y_hat)

num_epochs, lr = 5, 0.5
params = [W1, b1, W2, b2]


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for _, (X, y) in enumerate(data_iter):
        y = tf.cast(y,dtype=tf.int64)
        acc_sum += np.sum(tf.cast(tf.argmax(net(X), axis=1), dtype=tf.int64) == y)
        n += y.shape[0]
    return acc_sum / n

def train(net,train_iter,test_iter,loss,num_epochs,batch_size,params=None,lr=None,trainer=None):
	for epoch in range(num_epochs):
		train_l_sum,train_acc_sum,n =0.0,0.0,0
		for X,y in train_iter:
			with tf.GradientTape() as tape:
				y_hat = net(X)
				l =tf.reduce_sum(loss(y_hat,y))
			grads = tape.gradient(l,params)
			if trainer is None:
				for i, param in enumerate(params):
					param.assign_sub(lr*grads[i]/batch_size)
			else:
				trainer.apply_gradients(zip([grad/batch_size for grad in grads],params))
			y= tf.cast(y,dtype=tf.float32)
			train_l_sum +=l.numpy()
			train_acc_sum+= tf.reduce_sum(tf.cast(tf.argmax(y_hat, axis=1) == tf.cast(y, dtype=tf.int64), dtype=tf.int64)).numpy()
			n += y.shape[0]
		test_acc =evaluate_accuracy(test_iter,net)
		print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'% (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train(net,train_iter,test_iter,loss,num_epochs,batch_size,params,lr)