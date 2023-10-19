
import numpy as np
import pickle
import tensorflow as tf
from models.example_model import Classifier, ClassifierElliptic
import argparse
from data_loader.data_generator import DataGenerator

import copy
def concatenate_feature_labels(config, tail_path):
    for index, client_name in enumerate(config.features_central_client_name_list):
        path = f"./script_tmp/stage_1/{config.dataset}/{client_name}/{tail_path}"
        if index:
            tmp_feature = np.load(f"{path}/real_train_features.npy",allow_pickle=True)
            if config.feature_type == 'fake':
                tmp_feature = np.load(f"/Users/yangingdai/Downloads/GAN_Templtate/tmp/{client_name}/{config.fake_features_version_list[index]}/fake_features.npy",allow_pickle=True)
            tmp_labels = np.load(f"{path}/train_y.npy",allow_pickle=True)
            feature = np.concatenate((feature, tmp_feature),axis=0)
            labels = np.concatenate((labels, tmp_labels),axis=0)
        else:   #when index == 0 initial feature and labels
            feature = np.load(f"{path}/real_train_features.npy",allow_pickle=True)
            if config.feature_type == 'fake':
                feature = np.load(f"/Users/yangingdai/Downloads/GAN_Templtate/tmp/{client_name}/{config.fake_features_version_list[index]}/fake_features.npy",allow_pickle=True)
            labels = np.load(f"{path}/train_y.npy",allow_pickle=True)
    return feature, labels

def model_avg_init(config, cls):
    weight_object = []
    if config.clients_1_model_path is not None:
        cls.load_weights(tf.train.latest_checkpoint(config.clients_1_model_path)).expect_partial()
        weight_object.append(copy.deepcopy(cls.weights))
    if config.clients_2_model_path is not None:
        cls.load_weights(tf.train.latest_checkpoint(config.clients_2_model_path)).expect_partial()
        weight_object.append(copy.deepcopy(cls.weights))
    if config.clients_3_model_path is not None:
        cls.load_weights(tf.train.latest_checkpoint(config.clients_3_model_path)).expect_partial()
        weight_object.append(copy.deepcopy(cls.weights))
    if config.clients_4_model_path is not None:
        cls.load_weights(tf.train.latest_checkpoint(config.clients_4_model_path)).expect_partial()
        weight_object.append(copy.deepcopy(cls.weights))
    if config.clients_5_model_path is not None:
        cls.load_weights(tf.train.latest_checkpoint(config.clients_5_model_path)).expect_partial()
        weight_object.append(copy.deepcopy(cls.weights))

    data_amts = len(weight_object)
    if data_amts > 1:
        w_avg = copy.deepcopy(weight_object[0])
        for index in range(len(w_avg)):
            w_avg[index] = w_avg[index]/data_amts
            for i in range(1, data_amts):
                w_avg[index] += weight_object[i][index]/data_amts
        cls.set_weights(copy.deepcopy(w_avg))
    return cls

def create_feature_dataset(config, client_data):
    '''
    generate initial client feature to dataset
    '''
    dataset_dict = {}
    indices, feature_idx = None, None
    (train_data, test_data) = client_data
    client_train_data_num = len(train_data)
    for layer_num in config.features_ouput_layer_list:
        if config.use_assigned_epoch_feature:
            tail_path = f"assigned_epoch/{config.use_assigned_epoch_feature}/{layer_num}_layer_output/"
        else:
            tail_path = f"{layer_num}_layer_output/"
        feature, labels = concatenate_feature_labels(config, tail_path)
        config.total_features_num = len(labels)
        feature_dataset = list(zip(feature, labels))
        if config.take_feature_ratio < 1 and indices is None:
            num_feature_keep = int(config.total_features_num * config.take_feature_ratio)
            indices = np.random.permutation(config.total_features_num)[
                    :num_feature_keep
                ]
        if indices:  #take_feature_ratio
            feature_dataset = np.array(feature_dataset, dtype=object)[indices]
        if not config.update_feature_by_epoch and config.feature_match_train_data and feature_idx is None:
            #Convert the number of initial client feature to be the same as client_data
            feature_idx = np.random.choice(range(config.total_features_num), size=client_train_data_num, replace=True)
        if feature_idx is not None:
            feature_dataset = np.array(feature_dataset, dtype=object)[feature_idx]
        if not config.update_feature_by_epoch:
            feature, labels = zip(*feature_dataset)
            feature_dataset = tf.data.Dataset.from_tensor_slices(
                    (np.array(feature), np.array(labels)))
        dataset_dict[layer_num] = feature_dataset
    if not config.update_feature_by_epoch and not config.feature_match_train_data:
        train_data_idx = np.random.choice(range(client_train_data_num), size=config.total_features_num, replace=True)
        train_data = np.array(train_data, dtype=object)[train_data_idx]
    client_data = (train_data, test_data)
    return dataset_dict, client_data

def mian(config, cls):
    teacher_list = []
    avg_model = None
    feature_dataset = None
    feature_center = None
    if config.use_initial_model_weight:
        avg_model = model_avg_init(config, cls)
    return teacher_list, avg_model, feature_dataset, feature_center

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # General command line arguments for all models
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist"
    )
    parser.add_argument(
        "--clients_name",
        type=str,
        help="Name of client",
        default="clients_1",
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        help="Fraction of training data to make available to users",
        default=1,
    )
    parser.add_argument(
        "--take_feature_ratio",
        type=float,
        help="Fraction of pre-feature data to make available to users",
        default=1,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Measure of heterogeneity (higher is more homogeneous, lower is more heterogenous)",
        default=10,
    )
    parser.add_argument("--model_class", type=str,  default="Classifier")  #use which version as initial client( only for initial_client==0)
    parser.add_argument("--features_central_client_name_list", type=str, nargs='+', default=["clients_1"])  #use which version as initial client( only for initial_client==0)
    parser.add_argument("--features_central_version_list", type=str,nargs='+',  default=["0"])  #use which version as initial client( only for initial_client==0)
    parser.add_argument(
        "--feature_type",
        type=str,
        help="choose to usereal or fake feature",
        default="real",
    )
    parser.add_argument("--input_feature_size", type=int, default=165)
    parser.add_argument("--fake_features_version_list", type=str,nargs='+',  default=["0"])  #use which version as initial client( only for initial_client==0)
    parser.add_argument("--use_assigned_epoch_feature", type=int, default=0)  #use 0 means False==> use feature in best local acc( only for initial_client==0)
    parser.add_argument("--use_dirichlet_split_data", type=int, default=1)  #use 0 means False, 1 means True
    parser.add_argument("--feature_match_train_data", type=int, default=1)  #1 means set the length of feature to be the same as the length of train data, 0 reverse
    parser.add_argument("--use_initial_model_weight", type=int, default=0)  #use which version as initial client( only for initial_client==0)

    parser.add_argument("--learning_rate", type=float,default=0.001) 
    parser.add_argument("--batch_size", type=int, default=32) 

    parser.add_argument("--features_ouput_layer_list", help="The index of features output Dense layer",nargs='+',type=int, default=[-2])
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--feature_dim", type=int, default=128)

    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--clients_1_model_path", type=str, default=None)
    parser.add_argument("--clients_2_model_path", type=str, default=None)
    parser.add_argument("--clients_3_model_path", type=str, default=None)
    parser.add_argument("--clients_4_model_path", type=str, default=None)
    parser.add_argument("--clients_5_model_path", type=str, default=None)

    parser.add_argument("--max_to_keep", type=int, default=1)
    parser.add_argument("--use_same_kernel_initializer", type=int, default=1)

    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument("--data_random_seed", type=int, default=1693)

    args = parser.parse_args()
    args.use_dirichlet_split_data = bool(args.use_dirichlet_split_data)
    print("client:", args.clients_name)
    print("features_central_version_list:", args.features_central_version_list)
    print("features_ouput_layer_list:",args.features_ouput_layer_list)
    print("use_dirichlet_split_data",args.use_dirichlet_split_data)

    if args.dataset != "elliptic":
        cls = Classifier(args)
    else:
        cls = ClassifierElliptic(args)
    teacher_list, avg_model, feature_dataset, feature_center = mian(args, cls)

    #verify result
    # data = DataGenerator(args)
    # client_data = data.clients[args.clients_name]
    # (train_data, test_data) = client_data
    # train_x, train_y = zip(*train_data)
    # train_x  = tf.reshape(train_x, [-1,args.input_feature_size])
    # pre = avg_model(train_x[:10])
    # print("pre top 10", pre[0])

