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
        
        self.test_batched = tf.data.Dataset.from_tensor_slices((self.test_x[:self.config.test_sample_num], self.test_y[:self.config.test_sample_num])).shuffle(10000).batch(self.config.batch_size,drop_remainder=True)
        # self.train_batched = tf.data.Dataset.from_tensor_slices(
        #     (self.input[:self.config.train_sample_num] , self.y[:self.config.train_sample_num])).shuffle(50000).batch(self.config.batch_size,drop_remainder=True)

    def next_batch(self, batch_size):
        idx = np.random.choice(len(self.y), batch_size)
        yield self.input[idx], self.y[idx]
