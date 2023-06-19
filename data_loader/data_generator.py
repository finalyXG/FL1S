import numpy as np
import tensorflow as tf

class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here

        # self.input = np.ones((500, 784))
        # self.y = np.ones((500, 10))

        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        self.input = np.concatenate((x_train , x_test),axis=0)
        print('-- input shape:',self.input.shape)
        self.y = np.concatenate((y_train , y_test),axis=0)

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]
