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

def show_features_distribution(config, all_features, features_label, client1_version,version):
    ## get coresponding clients_1 feature center
    with open(f'tmp/clients_1/{client1_version}/features_central.pkl','rb') as fp: 
        features_central = pickle.load(fp)
        central_label = list(features_central.keys())
        central_features = np.array(list(features_central.values())).reshape([config.num_classes,-1])
    ##show features distribution generate from each client
    reshape_features = tf.reshape(all_features,[config.num_clients * config.test_feature_num,-1])
    reshape_features = tf.concat([reshape_features, central_features],0)  #add feature center 
    tsne = TSNE(n_components=2, verbose=1, random_state=config.random_seed)
    z = tsne.fit_transform(reshape_features)

    df = pd.DataFrame()
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    #transfor one_hot to int
    labels = np.argmax(features_label, axis=1)
    labels = tf.concat([labels, central_label],0)   #add feature center label
    # distinguish each client feature
    for index,num in enumerate(range(0, config.num_clients*config.test_feature_num, config.test_feature_num)):
        df.loc[num:num+config.test_feature_num-1,'y'] = "client_%d"%(index+1)
    df.loc[config.num_clients*config.test_feature_num:,'y'] = "client_1 center"

    df['classes'] = labels 

    ax = sns.scatterplot(x="comp-1", y="comp-2", hue=df.classes.tolist(), style=df.y.tolist(),
                    palette=sns.color_palette("hls", 10),
                    data=df)
    
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    for i,label in zip(z[-10:],central_label):
        ax.text(i[0], i[1], label)
    if not os.path.exists('./img/'):
        os.makedirs('./img/')
    plt.savefig("./img/%s.png"%version,bbox_inches='tight')
    plt.close()

def use_npy_generate_feature(config):
    features_dict = {}  # store each client feature
    combine_feature = {}  #store feature conbination
    for client_name in sorted(next(os.walk("./tmp/"))[1]): # get all dir from "./tmp/" path
        for version_num in next(os.walk(f"./tmp/{client_name}"))[1]:  
            print("client_name",client_name,"version_num",version_num)
            features_dict[client_name+"_"+version_num] = np.load(f"./tmp/{client_name}/{version_num}/real_features.npy",allow_pickle=True)
            if client_name != "clients_1":
                client1_version = version_num.split("_")[-1]
                cur_version = version_num.split("_")[0]
                combine_feature[version_num] = tf.concat([features_dict["clients_1"+"_"+client1_version],features_dict[client_name+"_"+version_num]],0)
        ## combine label
        if client_name == "clients_1":
            labels = np.load(f"./tmp/{client_name}/{version_num}/features_label.npy",allow_pickle=True)
        else:  #for client_2
            cur_label = np.load(f"./tmp/{client_name}/{version_num}/features_label.npy",allow_pickle=True)
            labels = tf.concat([labels,cur_label],0)
    for key,values in combine_feature.items():
        client1_version = key.split("_")[-1]
        client2_version = key.split("_")[0]
        show_features_distribution(config, values, labels, client1_version, f'npy_client1_version_{client1_version}_client2_version_{client2_version}')

def main(config):
    # cls_optimizer = tf.keras.optimizers.legacy.Adam()
    # checkpoint = tf.train.Checkpoint(optimizer=cls_optimizer,
    #                                     cls=cls, max_to_keep=tf.Variable(1))
    cls = Classifier(config)
    data = DataGenerator(config)
    clients_features = {}
    combine_feature = {}

    for client_name in sorted(next(os.walk(f"./tmp/"))[1]):
        (train_data, _) = data.clients[client_name]
        train_x,train_y = zip(*train_data)
        train_x,train_y = np.array(train_x),np.array(train_y)
        ## combine label
        if client_name == "clients_1":
            labels = train_y[:config.test_feature_num]
        else:  #only generate 2 client img
            labels = tf.concat([labels, train_y[:config.test_feature_num]],0)

        clients_features[client_name] = []
        for version_num in next(os.walk(f"./tmp/{client_name}"))[1]: 
            print("client_name",client_name," version_num",version_num)
            checkpoint_dir = f'./tmp/{client_name}/{version_num}/cls_training_checkpoints/local/'
            # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
            cls(train_x)
            cls.load_weights(tf.train.latest_checkpoint(checkpoint_dir))#.expect_partial()
            feature = cls.get_features(train_x[:config.test_feature_num])
            clients_features[client_name].append(feature)
            if client_name != "clients_1":
                client1_version = int(version_num.split("_")[-1])
                cur_version = int(version_num.split("_")[0])
                features = tf.concat([clients_features["clients_1"][client1_version],feature],0)
                show_features_distribution(config, features, labels, client1_version, f'client1_version_{client1_version}_{client_name}_version_{cur_version}')
            else:   #client_name == "clients_1"
                combine_feature[version_num] = feature

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
