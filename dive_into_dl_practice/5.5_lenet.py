import tensorflow as tf
print(tf.__version__)

net = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                           activation='sigmoid', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='sigmoid'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='sigmoid'),
    tf.keras.layers.Dense(84, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='sigmoid')
])

X = tf.random.uniform((1, 28, 28, 1))
for layer in net.layers:
    X = layer(X)
    print(layer.name, 'output shape\t', X.shape)


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

train_images = tf.reshape(
    train_images, (train_images.shape[0], train_images.shape[1], train_images.shape[2], 1))
print(train_images.shape)

test_images = tf.reshape(
    test_images, (test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))

optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.9, momentum=0.0, nesterov=False)

net.compile(optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])


net.fit(train_images, train_labels, epochs=5, validation_split=0.1)

net.evaluate(test_images, test_labels, verbose=2)
