
import numpy as np
import tensorflow as tf
import argparse
import copy
from script.model import Classifier

def get_features_centre(features, labels): 
    feature_output_layer_feature_avg_dic = {i:{} for i in features.keys()}
    for layer_num, feature_data in features.items():
        feature_data, labels = np.array(feature_data), np.array(labels)
        for label in set(labels):
            label_index = np.where(labels==label)
            feature_in_label = feature_data[label_index]
            avg_feature = tf.reduce_mean(feature_in_label, axis=0) 
            feature_output_layer_feature_avg_dic[layer_num][label] = avg_feature
    return feature_output_layer_feature_avg_dic

def concatenate_feature_labels(config):
    feature_list = []
    label_list = []
    features_centre_dict_list = []
    if config.clients_1_feature_path is not None:
        tmp_feature = np.load(f"{config.clients_1_feature_path}/real_features.npy",allow_pickle=True).item()
        feature_list.append(tmp_feature)
        tmp_label = np.load(f"{config.clients_1_feature_path}/label.npy",allow_pickle=True)
        label_list.append(tmp_label)
        features_centre_dict = get_features_centre(tmp_feature, tmp_label)
        features_centre_dict_list.append(features_centre_dict)
    if config.clients_2_feature_path is not None:
        tmp_feature = np.load(f"{config.clients_2_feature_path}/real_features.npy",allow_pickle=True).item()
        feature_list.append(tmp_feature)
        tmp_label = np.load(f"{config.clients_2_feature_path}/label.npy",allow_pickle=True)
        label_list.append(tmp_label)
        features_centre_dict = get_features_centre(tmp_feature, tmp_label)
        features_centre_dict_list.append(features_centre_dict)
    if config.clients_3_feature_path is not None:
        tmp_feature = np.load(f"{config.clients_3_feature_path}/real_features.npy",allow_pickle=True).item()
        feature_list.append(tmp_feature)
        tmp_label = np.load(f"{config.clients_3_feature_path}/label.npy",allow_pickle=True)
        label_list.append(tmp_label)
        features_centre_dict = get_features_centre(tmp_feature, tmp_label)
        features_centre_dict_list.append(features_centre_dict)
    if config.clients_4_feature_path is not None:
        tmp_feature = np.load(f"{config.clients_4_feature_path}/real_features.npy",allow_pickle=True).item()
        feature_list.append(tmp_feature)
        tmp_label = np.load(f"{config.clients_4_feature_path}/label.npy",allow_pickle=True)
        label_list.append(tmp_label)
        features_centre_dict = get_features_centre(tmp_feature, tmp_label)
        features_centre_dict_list.append(features_centre_dict)
    if config.clients_5_feature_path is not None:
        tmp_feature = np.load(f"{config.clients_5_feature_path}/real_features.npy",allow_pickle=True).item()
        feature_list.append(dict(tmp_feature))
        tmp_label = np.load(f"{config.clients_5_feature_path}/label.npy",allow_pickle=True)
        label_list.append(tmp_label)
        features_centre_dict = get_features_centre(tmp_feature, tmp_label)
        features_centre_dict_list.append(features_centre_dict)
    dataset_dict = {}
    feature_dict = {}
    labels_dict = {}
    for index, feature in enumerate(feature_list):
        for layer_num in config.features_ouput_layer_list:
            if index:
                feature_dict[layer_num] = np.concatenate((feature_dict[layer_num], feature[layer_num]),axis=0)
                labels_dict[layer_num] = np.concatenate((labels_dict[layer_num], label_list[index]),axis=0)
            else:   #when index == 0 initial feature and labels
                feature_dict[layer_num] = feature[layer_num]
                labels_dict[layer_num] = label_list[index]

    total_features_centre_dict = {layer_num: None for layer_num in feature_dict.keys()}
    for layer_num, v in feature_dict.items():
        dataset_dict[layer_num] = list(zip(v, labels_dict[layer_num]))
        for features_centre_dict in features_centre_dict_list:
            if total_features_centre_dict[layer_num]:
                for label, features_centre in features_centre_dict[layer_num].items():
                    total_features_centre_dict[layer_num][label] = tf.stack([total_features_centre_dict[layer_num][label],features_centre])
                    total_features_centre_dict[layer_num][label] = tf.reduce_mean(total_features_centre_dict[layer_num][label], axis=0) 
            else:
                total_features_centre_dict[layer_num] = features_centre_dict[layer_num]
    return dataset_dict, total_features_centre_dict

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

def mian(config, cls, test_sample):
    avg_model = None
    feature_dataset = None
    feature_center = None
    if config.use_initial_model_weight:
        avg_model = model_avg_init(config, cls)
    feature_data_dict, feature_center = concatenate_feature_labels(config)
    avg_model(test_sample)
    avg_model.save_weights(f"script_tmp/server/{config.dataset}/{config.clients_name}/model_avg/cp-{1:04d}.ckpt")
    np.save(f"script_tmp/server/{config.dataset}/{config.clients_name}/feature_center",feature_center)
    np.save(f"script_tmp/server/{config.dataset}/{config.clients_name}/real_features",feature_data_dict)
    return avg_model, feature_dataset, feature_center

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # General command line arguments for all models
    parser.add_argument(
        "--stage",
        type=int,
        default=0
    )
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
        "--whether_use_transformer_model",
        type=int,
        default=0
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
    parser.add_argument("--teacher_repeat", type=int, default=0)

    parser.add_argument("--features_ouput_layer_list", help="The index of features output Dense layer",nargs='+',type=int, default=[-2])
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_channels", type=int, default=1)

    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--clients_1_model_path", type=str, default=None)
    parser.add_argument("--clients_2_model_path", type=str, default=None)
    parser.add_argument("--clients_3_model_path", type=str, default=None)
    parser.add_argument("--clients_4_model_path", type=str, default=None)
    parser.add_argument("--clients_5_model_path", type=str, default=None)

    parser.add_argument("--clients_1_feature_path", type=str, default=None)
    parser.add_argument("--clients_2_feature_path", type=str, default=None)
    parser.add_argument("--clients_3_feature_path", type=str, default=None)
    parser.add_argument("--clients_4_feature_path", type=str, default=None)
    parser.add_argument("--clients_5_feature_path", type=str, default=None)

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
    args.teacher_repeat = bool(args.teacher_repeat)
    cls = Classifier(args)
    if args.dataset != "elliptic":
        test_sample = np.random.rand(3, args.image_size, args.image_size, args.num_channels)
    else:
        test_sample = np.random.rand(3, args.input_feature_size)

    avg_model, feature_dataset, feature_center = mian(args, cls, test_sample)