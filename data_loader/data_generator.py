import numpy as np
import tensorflow as tf
import random

class DataGenerator:
    def __init__(self, config):
        self.config = config
        tf.random.set_seed(config.data_random_seed)
        np.random.seed(config.data_random_seed)
        random.seed(config.data_random_seed)
        mnist = tf.keras.datasets.mnist

        (self.input, self.y), (self.test_x, self.test_y) = mnist.load_data()
        self.input = self.input.reshape(self.input.shape[0], 28, 28, 1).astype('float32')
        # self.input = (self.input - 127.5) / 127.5  # Normalize the images to [-1, 1]
        self.input = self.input  / 255  # Normalize the images to [0, 1]

        self.test_x = self.test_x.reshape(self.test_x.shape[0], 28, 28, 1).astype('float32')
        # self.test_x = (self.test_x - 127.5) / 127.5  # Normalize the images to [-1, 1]
        self.test_x = self.test_x / 255 
        # change y lable to one_hot
        self.y = tf.keras.utils.to_categorical(self.y, config.num_classes)
        self.test_y = tf.keras.utils.to_categorical(self.test_y, config.num_classes)

        self.clients = self.create_clients(config.num_clients)
        
    def next_batch(self, batch_size):
        idx = np.random.choice(len(self.y), batch_size)
        yield self.input[idx], self.y[idx]

    def create_clients(self, num_clients=2, initial='clients'):
        #create a list of client names
        client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]

        #randomize the data
        data_train = list(zip(self.input, self.y))
        data_test = list(zip(self.test_x, self.test_y))

        # random.shuffle(data_train)
        # random.shuffle(data_test)

        #shard data and place at each client
        self.client_train_size = len(data_train)//num_clients
        self.client_test_size = len(data_test) // num_clients

        #self.config.client_train_num
        shards = [(data_train[train_index:train_index + self.config.client_train_num],data_test[test_index:test_index+ self.config.client_test_num]) for train_index,test_index in zip(range(0, self.client_train_size*num_clients, self.client_train_size),range(0, self.client_test_size*num_clients,self.client_test_size))]

        #number of clients must equal number of shards
        assert(len(shards) == len(client_names))
        return {client_names[i] : shards[i] for i in range(len(client_names))}