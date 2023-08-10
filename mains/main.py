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
