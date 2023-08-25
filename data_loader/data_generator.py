import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import time
import seaborn as sns
from PIL import Image
import pandas as pd
import os
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
        if config.use_dirichlet_split_data:
            self.clients = self.split_data_dirichlet()
        else:
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

    def split_data_dirichlet(self, initial='clients'):
        """Split the dataset according to proportionsk sampled from a Dirichlet distribution, with alpha controlling the level of heterogeneity.

        :param dataset_train: the training dataset to split across users
        :param visualize: whether or not to visualize dataset split across users

        :return: user data splits as a list of torch.dataset.Subset objects
        """
        dataset_train = list(zip(self.input, self.y))
        if self.config.sample_ratio < 1:
            num_samples_keep = int(len(dataset_train) * self.config.sample_ratio)
            indices = np.random.permutation(len(dataset_train))[
                :num_samples_keep
            ] 
            dataset_train = np.array(dataset_train, dtype=object)[indices]

        print("len total train",len(dataset_train))

        x, y = zip(*dataset_train)

        # Getting indices for each class
        targets = np.argmax(y, axis=1)

        class_idxs = {}

        for i in range(self.config.num_classes):
            class_i_idxs = np.nonzero(targets == i)[0]
            np.random.shuffle(class_i_idxs)  # shuffling so that order doesn't matter
            class_idxs[i] = class_i_idxs

        # Sampling proportions for each user based on a Dirichlet distribution
        user_props = []  # will end up shape [config.num_clients x num_classes]
        for i in range(self.config.num_clients):
            props = np.random.dirichlet(
                np.repeat(self.config.alpha, self.config.num_classes)
            )  # sample the proportion of total samples that each class represents for a given user (will add to 1)
            user_props.append(props)

        user_props = np.array(user_props)
        scaled_props = user_props / np.sum(
            user_props, axis=0
        )  # scaling so that we add up to 100% of the data for each class (i.e., now we can distribute via these proportions and end up giving out all of the data)

        # Distributing data to users
        user_data_idxs = {i: [] for i in range(self.config.num_clients)}
        num_samples_per_user_per_class = {
            c: None for c in range(self.config.num_classes)
        }  # for visualization purposes

        for c in range(self.config.num_classes):
            num_pts_per_user = (scaled_props[:, c] * len(class_idxs[c])).astype(
                int
            )  # giving each user a number of samples based on their sampled proportion
            num_samples_per_user_per_class[c] = num_pts_per_user
            indices_per_user = [
                np.sum(num_pts_per_user[0 : i + 1]) for i in range(self.config.num_clients)
            ]  # sorting out indices for pulling out this data

            for i in range(self.config.num_clients):
                start_idx = indices_per_user[i - 1] if i - 1 >= 0 else 0
                end_idx = indices_per_user[i]
                user_data_idxs[i].extend(list(class_idxs[c][start_idx:end_idx]))

            # If we didn't quite distribute all data, distribute final samples uniformly at random
            if (indices_per_user[-1] - 1) < len(class_idxs[c]):
                remaining_idxs = class_idxs[c][indices_per_user[-1] :]

                for idx in remaining_idxs:
                    random_user = int(
                        np.random.choice([i for i in range(self.config.num_clients)], size=1)
                    )  # choose a user
                    user_data_idxs[random_user].append(idx)  # give the user this sample

                    num_samples_per_user_per_class[c][random_user] += 1

        print("user_data_idxs",[len(i) for i in user_data_idxs.values()])
        print("user_data_idxs",[i for i in user_data_idxs.keys()])
        # Wrapping dataset subset into client train dataset
        user_data = []
        for i in range(self.config.num_clients):
            user_data.append(np.array(dataset_train,dtype=object)[user_data_idxs[i]])

        # for element in set(targets):
        #     print(element," count: ", list(targets).count(element))

        client_names = ['{}_{}'.format(initial, i+1) for i in range(self.config.num_clients)]
       
       
        ### split test data
        test_datat = list(zip(self.test_x, self.test_y))
        client_test_size = len(test_datat) // self.config.num_clients

        user_test_data = [test_datat[test_index:test_index+ client_test_size] for test_index in range(0, client_test_size*self.config.num_clients,client_test_size)]
        dataset = list(zip(user_data, user_test_data))

        assert(len(dataset) == len(client_names))
        return {client_names[i] : dataset[i] for i in range(len(client_names))}