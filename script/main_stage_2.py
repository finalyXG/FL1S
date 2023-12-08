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
from script.model import Classifier, GAN, reduction_number, reduction_rate, epochs, score0_target1_num, smaller_half_number
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset = None):
        self.dataset = dataset

    def on_epoch_begin(self, epoch, logs=None):
        model.changed_label1_num = 0
        model.original_label1_num = 0
        model.changed_label0_num = 0
        model.original_label0_num = 0
        if self.dataset:
            if model.feature_data and model.config.update_feature_by_epoch:
                feature_dataset = {}
                np.random.seed(model.config.random_seed)
                feature_idx = np.random.choice(range(model.config.total_features_num), size=model.train_data_num, replace=True)
                for k, v in model.feature_data.items():
                    v = copy.deepcopy(np.array(v, dtype=object)[feature_idx])
                    feature, labels = zip(*v)
                    v = tf.data.Dataset.from_tensor_slices(
                        (np.array(feature), np.array(labels))).shuffle(len(labels))
                    feature_dataset[k] = v
                all_dataset = list(feature_dataset.values())
                all_dataset.append(model.train_data)
                all_train_data  = tf.data.Dataset.zip(tuple(all_dataset)).batch(model.config.batch_size)   #,drop_remainder=True
                self.dataset = all_train_data
        return super().on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        #save feature, feature center in cur_epoch model 
        if epoch in self.model.config.initial_client_ouput_feat_epochs:
            path = f"/Users/yangingdai/Downloads/GAN_Tensorflow-Project-Template/script_tmp/stage_1/{self.model.config.dataset}/{self.model.config.clients_name}/assigned_epoch/{epoch}"
            if not os.path.exists(path):
                os.makedirs(path)
            model.save_weights(f"{path}/cp-{epoch:04d}.ckpt")
            real_features = self.model.get_features(model.train_x)
            np.save(f"{path}/real_features",real_features)
            np.save(f"{path}/label",model.train_y)
        model.epochs.assign(epoch)
        for metric in model.metrics:
            metric.reset_states()
        for metric in model.compiled_metrics._metrics:
            metric.reset_states()
        print("original class rate:",model.original_label0_num/model.original_label1_num )
        print("changed class rate:",model.changed_label0_num/model.changed_label1_num )

    def on_test_begin(self, logs=None):
        for metric in model.metrics:
            metric.reset_states()
        for metric in model.compiled_metrics._metrics:
            metric.reset_states()
        
class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print()
        print(f"Epoch: {epoch} ", end='')
        for k,v in logs.items():
            if 'val' not in k:
                if "f1score" in k:
                    print(f"{k} is {v[0]:.5f}, ", end='')
                elif "loss" in k:
                    print(f"\n {k} is {v:.5f}, ", end='')
                else:
                    print(f"{k} is {v:.5f}, ", end='')
        print()

    def on_test_end(self, logs=None):
        print("test metrics:", end='')
        for k,v in logs.items():
            if "f1score" in k:
                print(f"{k} is {v[0]:.5f}, ", end='')
            elif "loss" in k:
                print(f"\n {k} is {v:.5f}, ", end='')
            else:
                print(f"{k} is {v:.5f}, ", end='')
        print()

class RecordTableCallback(tf.keras.callbacks.Callback):
    def on_test_end(self, logs=None):
        if not os.path.exists(f'script_tmp/stage_2/metrics_record.xlsx'):
            #create new exccel to record metrics in all clients
            global_workbook = openpyxl.Workbook() 
            global_worksheet = global_workbook.create_sheet("0", 0)
            for col_num,col_index in enumerate(["current_time",'dataset',"clients_name","use_initial_model_weight","cos_loss_weight","feat_loss_weight","hidden_rep_loss_weight","soft_target_loss_weight","temperature","loss_weight","gamma","change_ground_truth", "loss",  "rr", 'acc', "rd",'f1', 'recall', 'precision']):
                global_worksheet.cell(row=1, column=col_num+1, value = col_index) 
        else: 
            global_workbook = openpyxl.load_workbook(f'script_tmp/stage_2/metrics_record.xlsx')
            global_worksheet = global_workbook['0']
        global_worksheet.append([datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), model.config.dataset,model.config.clients_name, model.config.use_initial_model_weight, model.config.cos_loss_weight, model.config.feat_loss_weight, model.config.hidden_rep_loss_weight, model.config.soft_target_loss_weight, model.config.temperature, model.config.client_loss_weight, model.config.gamma, model.config.change_ground_truth, float(logs['loss']),float(logs['reduction_rate']),float(logs['cls_accuracy']),int(logs['reduction_number']),float(logs['f1score']),float(logs['recall']),float(logs['precision'])])
        global_workbook.save('script_tmp/stage_2/metrics_record.xlsx')

def create_feature_dataset(config, totoal_feature_data, train_data):
    '''
    generate initial client feature to dataset
    '''
    dataset_dict = {}
    indices, feature_idx = None, None
    client_train_data_num = len(train_data)
    for layer_num in config.features_ouput_layer_list:
        feature_dataset = totoal_feature_data[layer_num]
        config.total_features_num = len(feature_dataset)
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
    return dataset_dict, train_data

def get_teacher_list(config):
    teacher = Classifier(config)
    teacher_list = []
    if config.teacher_1_model_path is not None:
        teacher.load_weights(tf.train.latest_checkpoint(config.teacher_1_model_path)).expect_partial()
        teacher_list.append(copy.deepcopy(teacher))
    if config.teacher_2_model_path is not None:
        teacher.load_weights(tf.train.latest_checkpoint(config.teacher_2_model_path)).expect_partial()
        teacher_list.append(copy.deepcopy(teacher))
    if config.teacher_3_model_path is not None:
        teacher.load_weights(tf.train.latest_checkpoint(config.teacher_3_model_path)).expect_partial()
        teacher_list.append(copy.deepcopy(teacher))
    if config.teacher_4_model_path is not None:
        teacher.load_weights(tf.train.latest_checkpoint(config.teacher_4_model_path)).expect_partial()
        teacher_list.append(copy.deepcopy(teacher))
    if config.teacher_5_model_path is not None:
        teacher.load_weights(tf.train.latest_checkpoint(config.teacher_5_model_path)).expect_partial()
        teacher_list.append(copy.deepcopy(teacher))
    if len(teacher_list) > 1:
        if config.teacher_repeat:
            teacher_idx = np.random.choice(range(len(teacher_list)), size=config.teacher_num, replace=True)
        else:
            teacher_idx = np.random.choice(range(len(teacher_list)), size=config.teacher_num, replace=False)
        teacher_list = np.array(teacher_list)[teacher_idx]
    return teacher_list

def main(config, model, train_data, test_data, global_test_data):
    tf.random.set_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    #read data
    if config.use_initial_model_weight:
        model.load_weights(tf.train.latest_checkpoint(config.model_avg_weight_path)).expect_partial()
    if config.feature_center_path is not None and config.cos_loss_weight != float(0):
        pre_features_central = np.load(config.feature_center_path,allow_pickle=True).item()
    else:
        pre_features_central = None
    model.set_pre_features_central(pre_features_central)

    if config.feature_path is not None and config.feat_loss_weight != float(0):
        feature_data = np.load(config.feature_path,allow_pickle=True).item()  #dict, key:feature_layer num, value: corresponding feature data
        feature_data, train_data = create_feature_dataset(config, feature_data,  train_data)
    else:
        feature_data = None
    model.set_feature_data(feature_data)
    teacher_list = get_teacher_list(config)
    model.set_teacher_list(teacher_list)

    tf.random.set_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    train_x, train_y = zip(*train_data)
    model.set_train_x_train_y(train_x, train_y)
    class_rate = train_y.count(0)/train_y.count(1)
    model.set_train_data_num(len(train_y))
    test_x, test_y = zip(*test_data)
    global_test_x, global_test_y = zip(*global_test_data)
    train_x, train_y = np.array(train_x),np.array(train_y)
    test_x, test_y = np.array(test_x),np.array(test_y)
    global_test_x, global_test_y = np.array(global_test_x), np.array(global_test_y)

    train_data = tf.data.Dataset.from_tensor_slices((train_x,train_y))#.shuffle(len(train_y))  ,drop_remainder=True
    model.set_train_data(train_data)
    test_data = tf.data.Dataset.from_tensor_slices(
        (test_x, test_y)).batch(config.batch_size)
    global_test_data = tf.data.Dataset.from_tensor_slices(
        (global_test_x,global_test_y)).batch(config.batch_size)

    if config.dataset == "elliptic":
        model.set_loss_weight(class_rate)
        metrics_list = [reduction_number(), reduction_rate(), epochs(), score0_target1_num(), smaller_half_number(),
                        tf.keras.metrics.BinaryAccuracy(name='cls_accuracy'),
                        tf.keras.metrics.Recall(name='recall'),
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.F1Score(threshold=0.5, name='f1score')]
    else:
        metrics_list = [tf.keras.metrics.SparseCategoricalAccuracy(name='cls_accuracy')]
    
    if config.model_save_metrics == "acc":
        monitor = 'val_cls_accuracy'
    elif config.model_save_metrics == "f1":
        monitor = 'val_f1score'
    elif config.model_save_metrics == "rd":
        monitor = 'val_reduction_number'
    elif config.model_save_metrics == "rr":
        monitor = 'val_reduction_rate'


    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(config.learning_rate),
                    metrics= metrics_list,
                    loss=tf.keras.metrics.Mean(name='loss'),
                    run_eagerly = True)
    checkpoint_filepath = f'/Users/yangingdai/Downloads/GAN_Tensorflow-Project-Template/script_tmp/stage_2/{config.dataset}/{config.clients_name}/checkpoint/checkpoint'
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

    if feature_data is None:
        if len(train_y) >  config.batch_size:
            all_train_data = train_data.batch(config.batch_size,drop_remainder=True)
        else:
            all_train_data = train_data.batch(config.batch_size)
    elif not config.update_feature_by_epoch:
        all_dataset = list(feature_data.values())
        all_dataset.append(train_data)
        all_train_data  = tf.data.Dataset.zip(tuple(all_dataset)).batch(config.batch_size)
    elif config.update_feature_by_epoch:
        feature_dataset = {}
        feature_idx = np.random.choice(range(model.config.total_features_num), size=model.train_data_num, replace=True)
        for k, v in model.feature_data.items():
            v = copy.deepcopy(np.array(v, dtype=object)[feature_idx])
            feature, labels = zip(*v)
            v = tf.data.Dataset.from_tensor_slices(
                (np.array(feature), np.array(labels)))#.shuffle(len(labels))
            feature_dataset[k] = v
        all_dataset = list(feature_dataset.values())
        all_dataset.append(model.train_data)
        all_train_data  = tf.data.Dataset.zip(tuple(all_dataset)).batch(model.config.batch_size)   #,drop_remainder=True
                
    history = model.fit(all_train_data, epochs=config.cls_num_epochs, verbose=0, shuffle=True, validation_data=validation_data, callbacks=[CustomCallback(all_train_data), model_checkpoint_callback, LossAndErrorPrintingCallback() ])
    test_score = model.evaluate(validation_data, callbacks=[LossAndErrorPrintingCallback(),CustomCallback()],return_dict=True, verbose=0)

    model.load_weights(checkpoint_filepath)
    test_score = model.evaluate(validation_data, callbacks=[LossAndErrorPrintingCallback(),CustomCallback()],return_dict=True, verbose=0)
    np.save(f"script_tmp/stage_2/{config.dataset}/{config.clients_name}/{version_num}/real_features",model.get_features(train_x))   #save feature as a dict
    np.save(f"script_tmp/stage_2/{config.dataset}/{config.clients_name}/{version_num}/label",train_y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # General command line arguments for all models
    parser.add_argument(
        "--stage",
        type=int,
        default=2
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist"
    )
    parser.add_argument(
        "--input_feature_overlap_num",
        type=int,
        default=165
    )
    parser.add_argument(
        "--change_ground_truth",
        type=int,
        default=0
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.2
    )
    parser.add_argument(
        "--skip_if_all_0",
        type=int,
        default=0
    )
    parser.add_argument(
        "--whether_use_transformer_model",
        type=int,
        default=0
    )
    parser.add_argument(
        "--client_loss_weight",
        type=float,
        default=0
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.0
    )
    parser.add_argument(
        "--input_feature_no_overlap_num",
        type=int,
        default=0
    )
    parser.add_argument(
        "--extend_null_feature_num",
        type=int,
        default=0
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
    parser.add_argument("--teacher_1_model_path", type=str, default=None)
    parser.add_argument("--teacher_2_model_path", type=str, default=None)
    parser.add_argument("--teacher_3_model_path", type=str, default=None)
    parser.add_argument("--teacher_4_model_path", type=str, default=None)
    parser.add_argument("--teacher_5_model_path", type=str, default=None)
    parser.add_argument( "--model_avg_weight_path", type=str, default=None)
    parser.add_argument( "--feature_path", type=str, default=None)
    parser.add_argument( "--feature_center_path", type=str, default=None)
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
        default="real"
    )
    parser.add_argument("--fake_features_version_list", type=str,nargs='+',  default=["0"])  #use which version as initial client( only for initial_client==0)
    parser.add_argument("--use_initial_model_weight", type=int, default=0)  #use which version as initial client( only for initial_client==0)
    parser.add_argument("--use_assigned_epoch_feature", type=int, default=0)  #use 0 means False==> use feature in best local acc( only for initial_client==0)
    parser.add_argument("--use_dirichlet_split_data", type=int, default=1)  #use 0 means False, 1 means True
    parser.add_argument("--use_same_kernel_initializer", type=int, default=1)
    parser.add_argument("--feature_match_train_data", type=int, default=1)  #1 means set the length of feature to be the same as the length of train data, 0 reverse
    parser.add_argument("--update_feature_by_epoch", type=int, default=0)
    parser.add_argument("--cls_num_epochs", type=int, default=20)

    parser.add_argument("--original_cls_loss_weight", type=float, default=1.0)
    parser.add_argument("--feat_loss_weight", type=float, default=1.0)
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
    if 'clients_zero_feature_index' in args:
        train_x, train_y = zip(*train_data)
        test_x, test_y = zip(*test_data)
        train_x, test_x = np.array(train_x), np.array(test_x)
        if type(args.clients_zero_feature_index) == dict:
            print("mute feature len",len(args.clients_zero_feature_index[args.clients_name]))
            train_x[:,args.clients_zero_feature_index[args.clients_name]] = 0
            test_x[:,args.clients_zero_feature_index[args.clients_name]] = 0
        else:  #
            print("all overlap, mute feature len",len(args.clients_zero_feature_index))
            train_x[:,args.clients_zero_feature_index] = 0
            test_x[:,args.clients_zero_feature_index] = 0
          
        train_data = zip(train_x, train_y)
        test_data = zip(test_x, test_y)
    global_test_data = zip(data.test_x, data.test_y)
    model = Classifier(args)

    main(args, model, train_data, test_data, global_test_data)