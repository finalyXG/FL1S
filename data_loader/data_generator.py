import numpy as np
import pandas as pd
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import time
import seaborn as sns
from PIL import Image
import pandas as pd
import scipy.io
import os
class DataGenerator:
    def __init__(self, config):
        self.config = config
        tf.random.set_seed(config.data_random_seed)
        np.random.seed(config.data_random_seed)
        random.seed(config.data_random_seed)
        if config.dataset == "mnist":
            mnist = tf.keras.datasets.mnist
            config.num_channels = 1
            config.num_classes = 10
            config.image_size = 28

            (self.input, self.y), (self.test_x, self.test_y) = mnist.load_data()
            self.input = self.input.reshape(self.input.shape[0], 28, 28, config.num_channels).astype('float32')
            self.test_x = self.test_x.reshape(self.test_x.shape[0], 28, 28, config.num_channels).astype('float32')

        elif config.dataset == "fashion":
            fashion_mnist = tf.keras.datasets.fashion_mnist
            config.num_channels = 1
            config.num_classes = 10
            config.image_size = 28

            (self.input, self.y), (self.test_x, self.test_y) = fashion_mnist.load_data()
            self.input = self.input.reshape(self.input.shape[0], 28, 28, config.num_channels).astype('float32')
            self.test_x = self.test_x.reshape(self.test_x.shape[0], 28, 28, config.num_channels).astype('float32')
        elif config.dataset == "svhn":
            config.num_channels = 3
            config.num_classes = 10
            config.image_size = 32
            trainData = scipy.io.loadmat('./data_loader/svhn/train_32x32.mat')#load data
            self.input, self.y = trainData["X"], trainData["y"]

            testData = scipy.io.loadmat('./data_loader/svhn/test_32x32.mat')#load test data
            self.test_x, self.test_y = testData["X"], testData["y"]

            self.input = np.moveaxis(self.input, -1, 0).astype('float32')
            self.test_x = np.moveaxis(self.test_x, -1, 0).astype('float32')

            self.y = self.y.squeeze()
            self.test_y = self.test_y.squeeze()
            self.y[np.where(self.y == 10)] = 0
            self.test_y[np.where(self.test_y == 10)] = 0
        elif config.dataset == "elliptic":
            config.num_classes = 2
            df_classes = pd.read_csv("../elliptic_bitcoin_dataset/elliptic_txs_classes.csv")
            df_features = pd.read_csv("../elliptic_bitcoin_dataset/elliptic_txs_features.csv", header=None)
            colNames1 = {'0': 'txId', '1': "Time step"}
            colNames2 = {str(ii+2): "Local_feature_" + str(ii+1) for ii in range(93)}
            colNames3 = {str(ii+95): "Aggregate_feature_" + str(ii+1) for ii in range(72)}
            colNames = dict(colNames1, **colNames2, **colNames3 )
            colNames = {int(jj): item_kk for jj,item_kk in colNames.items()}
            df_features = df_features.rename(columns=colNames)
            df_classes.loc[df_classes['class'] == '1', 'class'] = 1
            df_classes.loc[df_classes['class'] == '2', 'class'] = 0
            df = df_features.merge(df_classes, how='inner', on='txId')
            df = df.drop(df[df['class'] == 'unknown'].index)
            df = df.sample(frac=1).reset_index(drop=True)
            train_data = df[df['Time step'] < 35]
            test_data = df[df['Time step'] >= 35]
            neg, pos = np.bincount(train_data['class'])
            config.elliptic_initial_bias = np.log([pos/neg])
            self.input, self.y = np.array(train_data.drop(["class","txId"],axis=1)).astype('float32'), np.array(train_data["class"]).astype('float32')
            self.test_x, self.test_y = np.array(test_data.drop(["class","txId"],axis=1)).astype('float32'), np.array(test_data["class"]).astype('float32')
            config.input_feature_size = self.input.shape[-1]
        else:
            raise NotImplementedError(
                f"Dataset '{config.data_random_seed}' has not been implemented, please choose either mnist, svhn or fashion"
            )
        if config.dataset != "elliptic":
            self.input = self.input  / 255 # Normalize the images to [0, 1]
            self.test_x = self.test_x  / 255
            self.dataset_train = list(zip(self.input, self.y))
            self.data_test = list(zip(self.test_x, self.test_y))
            # change y lable to one_hot
            # self.y = tf.keras.utils.to_categorical(self.y, config.num_classes)
            # self.test_y = tf.keras.utils.to_categorical(self.test_y, config.num_classes)
            if config.use_dirichlet_split_data:
                self.clients = self.split_data_dirichlet()
            else:
                self.clients = self.create_clients()
        else:
            self.dataset_train = list(zip(self.input, self.y))
            self.data_test = list(zip(self.test_x, self.test_y))
            if self.config.sample_ratio < 1:
                num_samples_keep = int(len(self.dataset_train) * self.config.sample_ratio)
                indices = np.random.permutation(len(self.dataset_train))[
                    :num_samples_keep
                ]
                self.dataset_train = np.array(self.dataset_train, dtype=object)[indices]
            x, targets = zip(*self.dataset_train)
            new_train_data = pd.concat([pd.DataFrame(x), pd.DataFrame(targets,columns=['class'])], axis=1)
            print(new_train_data.shape)
            user_train_data = []
            timestamp_range = int(35/config.num_clients)
            print("timestamp_range: ",timestamp_range)
            print(config.clients_name)
            print("---------")
            for index, timestamp in enumerate(range(1,35,timestamp_range)):
                print("client_", index+1," timestamp: [",timestamp,"->",timestamp+timestamp_range,")")
                client_data = new_train_data[(new_train_data[0] >= float(timestamp)) & (new_train_data[0] < float(timestamp)+timestamp_range)]
                client_input = np.array(client_data.drop(["class"],axis=1)).astype('float32')
                client_y = np.array(client_data["class"]).astype('float32')
                for k,v in client_data.groupby("class"):
                    print(k,": ",v.shape)
                client_data = list(zip(client_input, client_y))
                user_train_data.append(client_data)
            self.data_test = list(zip(self.test_x, self.test_y))
            self.client_test_size = len(self.data_test) // config.num_clients
            user_test_data = [self.data_test[test_index:test_index+ self.client_test_size] for test_index in range(0, self.client_test_size*self.config.num_clients, self.client_test_size)]
            dataset = list(zip(user_train_data, user_test_data))
            client_names = ['{}_{}'.format('clients', i+1) for i in range(self.config.num_clients)]

            self.clients = {client_names[i] : dataset[i] for i in range(len(client_names))}
        
    def next_batch(self, batch_size):
        idx = np.random.choice(len(self.y), batch_size)
        yield self.input[idx], self.y[idx]

    def create_clients(self, initial='clients'):
        #create a list of client names
        client_names = ['{}_{}'.format(initial, i+1) for i in range(self.config.num_clients)]

        #randomize the data
        # random.shuffle(data_train)
        # random.shuffle(data_test)
        # #shard data and place at each client
        self.client_train_size = len(self.dataset_train)//self.config.num_clients
        self.client_test_size = len(self.data_test) // self.config.num_clients
        shards = [(self.dataset_train[train_index:train_index + self.config.client_train_num],self.data_test[test_index:test_index+ self.config.client_test_num]) for train_index,test_index in zip(range(0, self.client_train_size*self.config.num_clients, self.client_train_size),range(0, self.client_test_size*self.config.num_clients,self.client_test_size))]

        #number of clients must equal number of shards
        assert(len(shards) == len(client_names))
        return {client_names[i] : shards[i] for i in range(len(client_names))}

    def split_data_dirichlet(self, initial='clients'):
        """Split the dataset according to proportionsk sampled from a Dirichlet distribution, with alpha controlling the level of heterogeneity.

        :param dataset_train: the training dataset to split across users
        :param visualize: whether or not to visualize dataset split across users

        :return: user data splits as a list of torch.dataset.Subset objects
        """
        dataset_train = self.dataset_train
        if self.config.sample_ratio < 1:
            num_samples_keep = int(len(dataset_train) * self.config.sample_ratio)
            indices = np.random.permutation(len(dataset_train))[
                :num_samples_keep
            ] 
            dataset_train = np.array(dataset_train, dtype=object)[indices]

        print("len total train",len(dataset_train))

        x, targets = zip(*dataset_train)
        targets = np.array(targets)
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
        test_data = list(zip(self.test_x, self.test_y))
        self.client_test_size = len(self.data_test) // self.config.num_clients

        user_test_data = [test_data[test_index:test_index+ self.client_test_size] for test_index in range(0, self.client_test_size*self.config.num_clients, self.client_test_size)]
        dataset = list(zip(user_data, user_test_data))

        assert(len(dataset) == len(client_names))
        return {client_names[i] : dataset[i] for i in range(len(client_names))}