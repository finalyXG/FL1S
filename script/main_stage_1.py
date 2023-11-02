
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import numpy as np
import os
import argparse
import pickle
import random
import openpyxl
import copy
import time, datetime
from data_loader.data_generator import DataGenerator
from script.model import Classifier
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        #save feature, feature center in cur_epoch model 
        if epoch in self.model.config.initial_client_ouput_feat_epochs:
            path = f"/Users/yangingdai/Downloads/GAN_Tensorflow-Project-Template/script_tmp/stage_1/{self.model.config.dataset}/{self.model.config.clients_name}/assigned_epoch/{epoch}"
            if not os.path.exists(path):
                os.makedirs(path)
            model.save_weights(f"{path}/cp-{epoch:04d}.ckpt")
            real_features = self.model.get_features(train_x)
            np.save(f"{path}/real_features",real_features)
            np.save(f"{path}/label",train_y)
        for metric in model.metrics:
            metric.reset_states()
        for metric in model.compiled_metrics._metrics:
            metric.reset_states()

    def on_test_begin(self, logs=None):
        for metric in model.metrics:
            metric.reset_states()
        for metric in model.compiled_metrics._metrics:
            metric.reset_states()
        
class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print()
        print(f"Epoch: {epoch+1} ", end='')
        for k,v in logs.items():
            if 'val' not in k:
                if "f1score" in k:
                    print(f"{k} is {v[0]*100:.3f}, ", end='')
                elif "loss" in k:
                    print(f"\n {k} is {v:.3f}, ", end='')
                else:
                    print(f"{k} is {v*100:.3f}, ", end='')
        print()

    def on_test_end(self, logs=None):
        print("test metrics:", end='')
        for k,v in logs.items():
            if "f1score" in k:
                print(f"{k} is {v[0]*100:.3f}, ", end='')
            elif "loss" in k:
                print(f"\n {k} is {v:.3f}, ", end='')
            else:
                print(f"{k} is {v*100:.3f}, ", end='')
        print()

def generate_initial_feature_center(config):
    """
    feature_center_dict[feature_layer_num][data_label]
    """
    initial_feature_center = [tf.random.normal([config.feature_dim, 1])]
    feature_center_dict = {}
    for layer_num in config.features_ouput_layer_list:
        for _ in range(config.num_classes-1):
            while True:
                flag = 1
                tmp_feature = tf.random.normal([config.feature_dim, 1])
                for feature in initial_feature_center:
                    feature = tf.reshape(feature, [-1,])
                    tmp_feature = tf.reshape(tmp_feature, [-1,])
                    cos_sim = tf.tensordot(feature, tmp_feature,axes=1)/(tf.linalg.norm(feature)*tf.linalg.norm(tmp_feature)+0.001)
                    if cos_sim > config.initial_feature_center_cosine_threshold:
                        flag = 0
                        break
                if flag == 1:
                    initial_feature_center.append(tmp_feature)
                    break
        feature_center_dict[layer_num] = {label: feature_center for label, feature_center in zip(range(config.num_classes), initial_feature_center)}
    return feature_center_dict

def main(config, model, train_data, test_data, global_test_data):
    tf.random.set_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    train_x, train_y = zip(*train_data)
    test_x, test_y = zip(*test_data)
    global_test_x, global_test_y  = zip(*global_test_data)
    train_x, train_y = np.array(train_x),np.array(train_y)
    test_x, test_y = np.array(test_x),np.array(test_y)
    global_test_x, global_test_y = np.array(global_test_x),np.array(global_test_y)
    if len(train_y) >  config.batch_size:
        train_data = tf.data.Dataset.from_tensor_slices(
            (train_x,train_y)).batch(config.batch_size,drop_remainder=True)
    else:
        train_data = tf.data.Dataset.from_tensor_slices(
            (train_x,train_y)).batch(config.batch_size)   #shuffle(len(train_y))
    test_data = tf.data.Dataset.from_tensor_slices(
        (test_x, test_y)).batch(config.batch_size)
    global_test_data = tf.data.Dataset.from_tensor_slices(
        (global_test_x, global_test_y)).batch(config.batch_size)  
        
    if config.dataset == "elliptic":
        metrics_list = [tf.keras.metrics.BinaryAccuracy(name='cls_accuracy'),
                        tf.keras.metrics.Recall(name='recall'),
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.F1Score(threshold=0.5, name='f1score')]
    else:
        metrics_list = [tf.keras.metrics.SparseCategoricalAccuracy(name='cls_accuracy')]
        
    if config.model_save_metrics == "acc":
        monitor = 'val_cls_accuracy'
    elif config.model_save_metrics == "f1":
        monitor = 'val_f1score'

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(config.learning_rate),
                    metrics= metrics_list,
                    loss=tf.keras.metrics.Mean(name='loss'),
                    run_eagerly = True)

    checkpoint_filepath = f'/Users/yangingdai/Downloads/GAN_Tensorflow-Project-Template/script_tmp/stage_1/{config.dataset}/{config.clients_name}/checkpoint/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor=monitor,
        mode='max',
        save_best_only=True)

    if config.validation_data == "local_test_data":
        validation_data = test_data
    elif config.validation_data == "global_test_data":
        validation_data = global_test_data
    history = model.fit(train_data, epochs=config.cls_num_epochs, verbose=0, shuffle=False, validation_data=validation_data, callbacks=[CustomCallback(), model_checkpoint_callback, LossAndErrorPrintingCallback() ])
    test_score = model.evaluate(validation_data, callbacks=[LossAndErrorPrintingCallback(),CustomCallback()],return_dict=True, verbose=0)

    model.load_weights(checkpoint_filepath)
    test_score = model.evaluate(validation_data, callbacks=[LossAndErrorPrintingCallback(),CustomCallback()],return_dict=True, verbose=0)
    print(model.get_features(train_x)[-2][:2])
    np.save(f"script_tmp/stage_1/{config.dataset}/{config.clients_name}/real_features",model.get_features(train_x))   #save feature as a dict
    np.save(f"script_tmp/stage_1/{config.dataset}/{config.clients_name}/label",train_y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # General command line arguments for all models
    parser.add_argument(
        "--stage",
        type=int,
        default=1
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist"
    )
    parser.add_argument(
        "--validation_data",
        type=str,
        default="local_test_data"
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
    parser.add_argument("--initial_client", type=int, default=1)  #use 0 and 1 to replace False and True 
    parser.add_argument(
        "--initial_client_ouput_feat_epochs", 
        type=int, 
        nargs='+', 
        help="Define generate trained feature data in which epoch",
        default=[-1]
        ) 
    parser.add_argument(
        "--whether_initial_feature_center", 
        type=int, 
        help="Whether send random initial feature center to initial client",
        default=0
        ) 
    parser.add_argument(
        "--initial_feature_center_cosine_threshold",
        type=float, 
        help="Define the consine similarity socre each initial feature center have to reach",
        default=0.5
    )
    parser.add_argument( "--model_save_metrics", type=str, default="acc")
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--soft_target_loss_weight", type=float, default=0.0) 
    parser.add_argument("--hidden_rep_loss_weight", type=float, default=0.0)
    parser.add_argument("--teacher_num", type=int, default=1)
    parser.add_argument("--teacher_repeat", type=int, default=0)
    parser.add_argument("--features_central_client_name_list", type=str, nargs='+', default=["clients_1"])  #use which version as initial client( only for initial_client==0)
    parser.add_argument("--features_central_version_list", type=str,nargs='+',  default=["0"])  #use which version as initial client( only for initial_client==0)
    parser.add_argument(
        "--feature_type",
        type=str,
        help="choose to usereal or fake feature",
        default="real",
    )
    parser.add_argument("--fake_features_version_list", type=str,nargs='+',  default=["0"])  #use which version as initial client( only for initial_client==0)
    parser.add_argument("--use_initial_model_weight", type=int, default=0)  #use which version as initial client( only for initial_client==0)
    parser.add_argument("--use_assigned_epoch_feature", type=int, default=0)  #use 0 means False==> use feature in best local acc( only for initial_client==0)
    parser.add_argument("--use_dirichlet_split_data", type=int, default=1)  #use 0 means False, 1 means True
    parser.add_argument("--use_same_kernel_initializer", type=int, default=1)
    parser.add_argument("--feature_match_train_data", type=int, default=1)  #1 means set the length of feature to be the same as the length of train data, 0 reverse
    parser.add_argument("--update_feature_by_epoch", type=int, default=0)
    parser.add_argument("--cls_num_epochs", type=int, default=20)

    parser.add_argument("--original_cls_loss_weight_list", type=float, default=1.0)
    parser.add_argument("--feat_loss_weight_list", type=float, default=1.0)
    parser.add_argument("--cos_loss_weight", type=float, default=5.0) 
    parser.add_argument("--learning_rate", type=float,default=0.001) 
    parser.add_argument("--batch_size", type=int, default=32) 

    parser.add_argument("--features_ouput_layer_list", help="The index of features output Dense layer",nargs='+',type=int, default=[-2])
    parser.add_argument("--GAN_num_epochs", type=int, default=1)
    parser.add_argument("--test_feature_num", type=int, default=500)
    parser.add_argument("--test_sample_num", help="The number of real features and fake features in tsne img", type=int, default=500) 
    parser.add_argument("--gp_weight", type=int, default=10.0)
    parser.add_argument("--discriminator_extra_steps", type=int, default=3)
    parser.add_argument("--num_examples_to_generate", type=int, default=16)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--max_to_keep", type=int, default=5)

    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--client_train_num", type=int, default=1000)
    parser.add_argument("--client_test_num", type=int, default=1000)
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument("--data_random_seed", type=int, default=1693)

    parser.add_argument("--exp_name", type=str, default="example")
    parser.add_argument("--logdir", type=str, default="logs/hparam_tuning")

    args = parser.parse_args()
    args.initial_client = bool(args.initial_client)
    args.use_initial_model_weight = bool(args.use_initial_model_weight)
    args.use_dirichlet_split_data = bool(args.use_dirichlet_split_data)
    args.update_feature_by_epoch = bool(args.update_feature_by_epoch)
    args.whether_initial_feature_center = bool(args.whether_initial_feature_center)
    args.teacher_repeat = bool(args.teacher_repeat)
    print("client:", args.clients_name)
    print("Whether initial_client:", args.initial_client)
    print("features_central_version_list:", args.features_central_version_list)
    if not args.use_dirichlet_split_data:
        print("client_train_num:", args.client_train_num)
        print("client_test_num:", args.client_test_num)
    print("cls_num_epochs:", args.cls_num_epochs)
    print("initial_client_ouput_feat_epoch:", args.initial_client_ouput_feat_epochs)
    print("features_ouput_layer_list:",args.features_ouput_layer_list)
    print("use_dirichlet_split_data",args.use_dirichlet_split_data)
    data = DataGenerator(args)
    client_data = data.clients[args.clients_name]
    (train_data, test_data) = client_data
    train_x, train_y = zip(*train_data)
    args.class_rate = train_y.count(0)/train_y.count(1)

    global_test_data = zip(data.test_x, data.test_y)
    if args.whether_initial_feature_center:
        tf.random.set_seed(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        args.initial_feature_center = generate_initial_feature_center(args)
    model = Classifier(args)

    if not os.path.exists(f"script_tmp/stage_1/{args.dataset}/{args.clients_name}/"):
        os.makedirs(f"script_tmp/stage_1/{args.dataset}/{args.clients_name}/")
    record_hparams_file = open(f"./script_tmp/stage_1/{args.dataset}/{args.clients_name}/hparams_record.txt", "wt")
    for key,value in vars(args).items():
        record_hparams_file.write(f"{key}: {value}")
        record_hparams_file.write("\n")
    record_hparams_file.close()
    main(args, model, train_data, test_data, global_test_data)

    # #old code result
    # if args.path == 1:
    #     path = "tmp/clients_1/68/-2_layer_output"
    # else:
    #     path = "script_tmp/stage_1/elliptic/clients_1/-2_layer_output"
    # label = np.load(f"{path}/train_y.npy",allow_pickle=True)
    # feature = np.load(f"{path}/real_train_features.npy",allow_pickle=True)

    # checkpoint_dir = f'tmp/clients_1/68/cls_training_checkpoints/local/'
    # model.load_weights(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    # model_feature = model.get_features(train_x)

    # with open(f'{path}/features_central.pkl','rb') as fp: 
    #     features_central = pickle.load(fp) #load features_central pre-saved
    # #feature  label=  model_feature=   features_central
    # print(feature)

# export PYTHONPATH=/Users/yangingdai/Downloads/GAN_Tensorflow-Project-Template; python script/main_stage_1.py --batch_size 32 --learning_rate 0.001 --alpha 10 --use_dirichlet_split_data 1 --cls_num_epochs 80 --initial_client 1 --features_ouput_layer_list -2 --feature_dim 50 --num_clients 5 --dataset elliptic --clients_name clients_5 --sample_ratio 0.1 --whether_initial_feature_center 1  0.9651442
# export PYTHONPATH=/Users/yangingdai/Downloads/GAN_Tensorflow-Project-Template; python script/main_stage_1.py --batch_size 32 --learning_rate 0.001 --alpha 10 --use_dirichlet_split_data 1 --cls_num_epochs 80 --initial_client 1 --features_ouput_layer_list -2 --feature_dim 50 --num_clients 5 --dataset elliptic --clients_name clients_4 --sample_ratio 0.1 --whether_initial_feature_center 1  0.9495192
# export PYTHONPATH=/Users/yangingdai/Downloads/GAN_Tensorflow-Project-Template; python script/main_stage_1.py --batch_size 32 --learning_rate 0.001 --alpha 10 --use_dirichlet_split_data 1 --cls_num_epochs 80 --initial_client 1 --features_ouput_layer_list -2 --feature_dim 50 --num_clients 5 --dataset elliptic --clients_name clients_3 --sample_ratio 0.1 --whether_initial_feature_center 1  0.95402646
# export PYTHONPATH=/Users/yangingdai/Downloads/GAN_Tensorflow-Project-Template; python script/main_stage_1.py --batch_size 32 --learning_rate 0.001 --alpha 10 --use_dirichlet_split_data 1 --cls_num_epochs 80 --initial_client 1 --features_ouput_layer_list -2 --feature_dim 50 --num_clients 5 --dataset elliptic --clients_name clients_2 --sample_ratio 0.1 --whether_initial_feature_center 1  0.96153843
# export PYTHONPATH=/Users/yangingdai/Downloads/GAN_Tensorflow-Project-Template; python script/main_stage_1.py --batch_size 32 --learning_rate 0.001 --alpha 10 --use_dirichlet_split_data 1 --cls_num_epochs 80 --initial_client 1 --features_ouput_layer_list -2 --feature_dim 50 --num_clients 5 --dataset elliptic --clients_name clients_1 --sample_ratio 0.1 --whether_initial_feature_center 1  0.9609375
# export PYTHONPATH=/Users/yangingdai/Downloads/GAN_Tensorflow-Project-Template; python mains/client.py --batch_size 32 --learning_rate 0.001 --alpha 10 --use_dirichlet_split_data 1 --cls_num_epochs 80 --initial_client 1 --features_ouput_layer_list -2 --feature_dim 50 --num_clients 5 --dataset elliptic --clients_name clients_1 --sample_ratio 0.1 --whether_initial_feature_center 1 

# export PYTHONPATH=/Users/yangingdai/Downloads/GAN_Tensorflow-Project-Template; python mains/get_real_feature.py  --batch_size 32 --learning_rate 0.001 --alpha 10 --use_dirichlet_split_data 1 --cls_num_epochs 80 --initial_client 1 --features_ouput_layer_list -2 --feature_dim 50 --num_clients 5 --dataset elliptic --clients_name clients_5 --sample_ratio 0.1 --whether_initial_feature_center 1 --version_num 0
# export PYTHONPATH=/Users/yangingdai/Downloads/GAN_Tensorflow-Project-Template; python mains/get_real_feature.py  --batch_size 32 --learning_rate 0.001 --alpha 10 --use_dirichlet_split_data 1 --cls_num_epochs 80 --initial_client 1 --features_ouput_layer_list -2 --feature_dim 50 --num_clients 5 --dataset elliptic --clients_name clients_4 --sample_ratio 0.1 --whether_initial_feature_center 1 --version_num 0
# export PYTHONPATH=/Users/yangingdai/Downloads/GAN_Tensorflow-Project-Template; python mains/get_real_feature.py  --batch_size 32 --learning_rate 0.001 --alpha 10 --use_dirichlet_split_data 1 --cls_num_epochs 80 --initial_client 1 --features_ouput_layer_list -2 --feature_dim 50 --num_clients 5 --dataset elliptic --clients_name clients_3 --sample_ratio 0.1 --whether_initial_feature_center 1 --version_num 0
# export PYTHONPATH=/Users/yangingdai/Downloads/GAN_Tensorflow-Project-Template; python mains/get_real_feature.py  --batch_size 32 --learning_rate 0.001 --alpha 10 --use_dirichlet_split_data 1 --cls_num_epochs 80 --initial_client 1 --features_ouput_layer_list -2 --feature_dim 50 --num_clients 5 --dataset elliptic --clients_name clients_2 --sample_ratio 0.1 --whether_initial_feature_center 1 --version_num 0
# export PYTHONPATH=/Users/yangingdai/Downloads/GAN_Tensorflow-Project-Template; python mains/get_real_feature.py  --batch_size 32 --learning_rate 0.001 --alpha 10 --use_dirichlet_split_data 1 --cls_num_epochs 80 --initial_client 1 --features_ouput_layer_list -2 --feature_dim 50 --num_clients 5 --dataset elliptic --clients_name clients_1 --sample_ratio 0.1 --whether_initial_feature_center 1 --version_num 0