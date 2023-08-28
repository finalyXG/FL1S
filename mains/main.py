import tensorflow as tf
from data_loader.data_generator import DataGenerator
from trainers.example_trainer import Trainer
from models.example_model import Classifier
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from tensorboard.plugins.hparams import api as hp
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from client import clients_main
import numpy as np
import pandas as pd
import argparse
import os
import time
import pickle
import random

def show_features_distribution(config, client_name,version_num):
    label = np.load(f"./tmp/{client_name}/{version_num}/features_label.npy",allow_pickle=True)#[:config.test_feature_num]
    feature = np.load(f"./tmp/{client_name}/{version_num}/real_features.npy",allow_pickle=True)#[:config.test_feature_num]
    with open(f'tmp/clients_1/{version_num}/features_central.pkl','rb') as fp: 
        features_central = pickle.load(fp)
        central_label = list(features_central.keys())
        central_features = np.array(list(features_central.values())).reshape([config.num_classes,-1])
    ##show features distribution generate from each client
    reshape_features = tf.concat([feature, central_features],0)  #add feature center 
    tsne = TSNE(n_components=2, verbose=1, random_state=config.random_seed)
    z = tsne.fit_transform(reshape_features)

    df = pd.DataFrame()
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    #transfor one_hot to int
    labels = np.argmax(label, axis=1)
    labels = tf.concat([labels, central_label],0)   #add feature center label

    df['classes'] = labels 
    ax = sns.scatterplot(x="comp-1", y="comp-2", hue=df.classes.tolist(),
                    palette=sns.color_palette("hls", 10),
                    data=df)
    
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    for i,label in zip(z[-10:],central_label):
        ax.text(i[0], i[1], label)
    if not os.path.exists('./img/'):
        os.makedirs('./img/')
    plt.savefig("./img/%s.png"%version_num,bbox_inches='tight')
    plt.close()

def show_clients_features_distribution(config, all_features, features_label,num_clients, clients_length, client1_version,version):
    # num_clients = len(version.split("_"))
    ## get coresponding clients_1 feature center
    # with open(f'tmp/clients_1/{client1_version}/assigned_epoch/80/-2_layer_output/features_central.pkl','rb') as fp: 
    #     features_central = pickle.load(fp)
    #     central_label = list(features_central.keys())
    #     central_features = np.array(list(features_central.values())).reshape([config.num_classes,-1])
    ##show features distribution generate from each client
    reshape_features = tf.reshape(all_features,[sum(clients_length),-1])
    # reshape_features = tf.concat([reshape_features, central_features],0)  #add feature center 
    tsne = TSNE(n_components=2, verbose=1, random_state=config.random_seed)
    z = tsne.fit_transform(reshape_features)

    df = pd.DataFrame()
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    #transfor one_hot to int
    labels = np.argmax(features_label, axis=1)
    # labels = tf.concat([labels, central_label],0)   #add feature center label
    # distinguish each client feature
    # for index,num in enumerate(range(0, num_clients*config.test_feature_num, config.test_feature_num)):
    #     df.loc[num:num+config.test_feature_num-1,'y'] = "client_%d"%(index+1)
    for index in range(len(clients_length)):
        df.loc[sum(clients_length[:index]):sum(clients_length[:index+1])-1,'y'] = "client_%d"%(index+1)        
    df.loc[sum(clients_length):,'y'] = "client_1 center"

    df['classes'] = labels 
    ax = sns.scatterplot(x="comp-1", y="comp-2", hue=df.classes.tolist(), style=df.y.tolist(),
                    palette=sns.color_palette("hls", 10),
                    data=df)
    
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    # for i,label in zip(z[-10:],central_label):
    #     ax.text(i[0], i[1], label)
    if not os.path.exists('./img/'):
        os.makedirs('./img/')
    plt.savefig("./img/%s.png"%version,bbox_inches='tight')
    plt.close()

def use_npy_generate_feature(config):
    combine_feature = {}  #store feature conbination
    for client_name in sorted(next(os.walk("./tmp/"))[1]): # get all dir from "./tmp/" path
        ## combine label
        if client_name == "clients_1":
            labels = np.load(f"./tmp/{client_name}/0/features_label.npy",allow_pickle=True)
        else:  
            version_num = next(os.walk(f"./tmp/{client_name}"))[1][0]
            cur_label = np.load(f"./tmp/{client_name}/{version_num}/features_label.npy",allow_pickle=True)
            labels = tf.concat([labels,cur_label],0)
        
        ## combine feature
        for version_num in next(os.walk(f"./tmp/{client_name}"))[1]:  
            print("client_name",client_name,"version_num",version_num)
            feature = np.load(f"./tmp/{client_name}/{version_num}/real_features.npy",allow_pickle=True)
            if client_name != "clients_1":
                client1_version = version_num.split("_")[-1]
                cur_version = version_num.split("_")[0]
                pre_client_num = int(client_name.split("_")[-1]) -1
                pre_combine_feature = combine_feature[f"{client1_version}_clients_{pre_client_num}"]
                for key,pre_feature in pre_combine_feature.items():
                    combine_features = tf.concat([pre_feature,feature],0)
                    cur_combine_version = f"{key}_{cur_version}"
                    if f"{client1_version}_{client_name}" in combine_feature:
                        combine_feature[f"{client1_version}_{client_name}"][cur_combine_version] = combine_features
                    else:
                        combine_feature[f"{client1_version}_{client_name}"] = {cur_combine_version: combine_features}
                    show_features_distribution(config, combine_features, labels, client1_version, "npy"+cur_combine_version)
            
            else: #client_name == "clients_1":
                combine_feature[f"{version_num}_{client_name}"] = {version_num: feature}

def main(config):
    # cls_optimizer = tf.keras.optimizers.legacy.Adam()
    # checkpoint = tf.train.Checkpoint(optimizer=cls_optimizer,
    #                                     cls=cls, max_to_keep=tf.Variable(1))
    cls = Classifier(config)
    data = DataGenerator(config)
    combine_feature = {}

    for client_name in sorted(next(os.walk(f"./tmp/"))[1]):
        (train_data, _) = data.clients[client_name]
        train_x,train_y = zip(*train_data)
        train_x,train_y = np.array(train_x),np.array(train_y)
        cls(train_x)
        ## combine label
        if client_name == "clients_1":
            labels = train_y[:config.test_feature_num]
        else:  
            labels = tf.concat([labels, train_y[:config.test_feature_num]],0)

        ## combine feature
        for version_num in next(os.walk(f"./tmp/{client_name}"))[1]: 
            print("client_name",client_name," version_num",version_num)
            checkpoint_dir = f'./tmp/{client_name}/{version_num}/cls_training_checkpoints/local/'
            # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
            cls.load_weights(tf.train.latest_checkpoint(checkpoint_dir))#.expect_partial()
            feature = cls.get_features(train_x[:config.test_feature_num])
            if client_name != "clients_1":
                client1_version = int(version_num.split("_")[-1])
                cur_version = int(version_num.split("_")[0])
                pre_client_num = int(client_name.split("_")[-1]) -1

                pre_combine_feature = combine_feature[f"{client1_version}_clients_{pre_client_num}"]
                for key,pre_feature in pre_combine_feature.items():
                    combine_features = tf.concat([pre_feature,feature],0)
                    cur_combine_version = f"{key}_{cur_version}"
                    if f"{client1_version}_{client_name}" in combine_feature:
                        combine_feature[f"{client1_version}_{client_name}"][cur_combine_version] = combine_features
                    else:
                        combine_feature[f"{client1_version}_{client_name}"] = {cur_combine_version: combine_features}
                    show_features_distribution(config, combine_features, labels, client1_version, cur_combine_version)
            
            else: #client_name == "clients_1":
                combine_feature[f"{version_num}_{client_name}"] = {version_num: feature}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # General command line arguments for all models
    parser.add_argument("--test_feature_num", type=int, default=500)
    parser.add_argument("--client_train_num", type=int, default=2000)
    parser.add_argument("--client_test_num", type=int, default=2000)
    parser.add_argument("--features_ouput_layer", help="The index of features output Dense layer",type=int, default=2)
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--buffer_size", type=int, default=5000)
    parser.add_argument("--latent_dim", type=int, default=16)
    # parser.add_argument("--feature_dim", type=int, default=16)
    parser.add_argument("--max_to_keep", type=int, default=5)
    parser.add_argument("--num_classes", type=int, default=10)

    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--random_seed", type=int, default=10)

    parser.add_argument("--exp_name", type=str, default="example")
    parser.add_argument("--logdir", type=str, default="logs/hparam_tuning")

    args = parser.parse_args()
    tf.random.set_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    print("num_clients:", args.num_clients)
    print("test_feature_num:", args.test_feature_num)
    main(args)   #use model generate feature
    # use_npy_generate_feature(args)
