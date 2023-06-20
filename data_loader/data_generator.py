import numpy as np
import tensorflow as tf

class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here

        # self.input = np.ones((500, 784))
        # self.y = np.ones((500, 10))

        mnist = tf.keras.datasets.mnist

        (self.input, self.y), (_, __) = mnist.load_data()
        self.input = self.input.reshape(self.input.shape[0], 28, 28, 1).astype('float32')
        self.input = (self.input - 127.5) / 127.5  # Normalize the images to [-1, 1]

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]
