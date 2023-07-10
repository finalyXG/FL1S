import numpy as np
import tensorflow as tf
import random

class DataGenerator:
    def __init__(self, config):
        self.config = config
        mnist = tf.keras.datasets.mnist

        (self.input, self.y), (self.test_x, self.test_y) = mnist.load_data()
        self.input = self.input.reshape(self.input.shape[0], 28, 28, 1).astype('float32')
        self.input = (self.input - 127.5) / 127.5  # Normalize the images to [-1, 1]

        self.test_x = self.test_x.reshape(self.test_x.shape[0], 28, 28, 1).astype('float32')
        self.test_x = (self.test_x - 127.5) / 127.5  # Normalize the images to [-1, 1]

        self.test_batched = tf.data.Dataset.from_tensor_slices((self.test_x, self.test_y)).batch(self.config.batch_size)
        self.train_batched = tf.data.Dataset.from_tensor_slices(
            (self.input , self.y)).shuffle(10000).batch(self.config.batch_size)

    def next_batch(self, batch_size):
        idx = np.random.choice(len(self.y), batch_size)
        yield self.input[idx], self.y[idx]
