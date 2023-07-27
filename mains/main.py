import tensorflow as tf
from data_loader.data_generator import DataGenerator
from trainers.example_trainer import Trainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from tensorboard.plugins.hparams import api as hp
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import random
from client import clients_main
import numpy as np
import pandas as pd
def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])

    data = DataGenerator(config)
    client_names= list(data.clients.keys())
    random.shuffle(client_names)
    local_acc_list = []
    global_acc_list = []
    for index,client_name in enumerate(client_names):
        local_acc, global_acc, fake_features = clients_main(config, data.clients[client_name], data.test_x,data.test_y, client_name)
        local_acc_list.append(local_acc)
        global_acc_list.append(global_acc)
        if index != 0:
            compare_feature = tf.concat([compare_feature,fake_features],0)
        else:
            compare_feature = fake_features

    reshape_compare_feature = tf.reshape(compare_feature,[config.num_clients*config.test_feature_num,-1])

    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(reshape_compare_feature)

    df = pd.DataFrame()
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    #transfor one_hot to interge
    labels = np.argmax(data.test_y[:config.test_feature_num], axis=1)
    #show discriminator output
    for index,num in enumerate(range(0, config.num_clients*config.test_feature_num, config.test_feature_num)):
        df.loc[num:num+config.test_feature_num-1,'y'] = "client_%d"%index
        df.loc[num:num+config.test_feature_num-1,'classes'] = labels 

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.classes.tolist(), style=df.y.tolist(),
                    palette=sns.color_palette("hls", 10),
                    data=df)
    plt.savefig("compare_client_fake_features_img.png")
    # plt.close()


if __name__ == '__main__':
    main()
