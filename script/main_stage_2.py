import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import numpy as np
import os
import argparse
import pickle
import random
import openpyxl
import copy
import time
from models.example_model import Classifier, ClassifierElliptic, C_Discriminator,C_Generator, AC_Discriminator, AC_Generator
from data_loader.data_generator import DataGenerator

def create_feature_dataset(config, totoal_feature_data, train_data):
    '''
    generate initial client feature to dataset
    '''
    dataset_dict = {}
    indices, feature_idx = None, None
    client_train_data_num = len(train_data)
    for layer_num in config.features_ouput_layer_list:
        feature_dataset = totoal_feature_data[layer_num]
        feature, labels = zip(*feature_dataset)
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

def count_features_central_distance(model, features_central, x, y):
    feature = model.get_features(x)
    accumulate_loss = 0
    for k,v in feature.items():
        for vector,label in zip(v,y): 
            pre_vector = features_central[k][label.numpy()]
            vector = tf.reshape(vector, [-1,])
            pre_vector = tf.reshape(pre_vector, [-1,])
            cos_sim = tf.tensordot(vector, pre_vector,axes=1)/(tf.linalg.norm(vector)*tf.linalg.norm(pre_vector)+0.001)
            accumulate_loss += 1 - cos_sim
    return accumulate_loss / len(y)

def get_features_central(config, model, x, y):  
    feature_output_layer_feature_avg_dic = {i:{} for i in config.features_ouput_layer_list}
    for label in set(y):
        label_index = np.where(y==label)
        feature_list = model.get_features(x[label_index])
        for k,v in feature_list.items():
            avg_feature = tf.reduce_mean(v, axis=0) 
            feature_output_layer_feature_avg_dic[k][label] = avg_feature
    return feature_output_layer_feature_avg_dic

def get_teacher_list(config, teacher):
    teacher_list = []
    if config.teacher_1_model_path is not None:
        teacher.load_weights(tf.train.latest_checkpoint(config.teacher_1_model_path)).expect_partial()
        teacher_list.append(copy.deepcopy(teacher))
    if config.teacher_2_model_path is not None:
        teacher.load_weights(tf.train.latest_checkpoint(config.teacher_1_model_path)).expect_partial()
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

def main(model, train_data, test_data, global_test_data, config):
    tf.random.set_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    #read data
    if config.feature_center_path is not None and config.cos_loss_weight != float(0):
        pre_features_central = np.load(config.feature_center_path,allow_pickle=True).item()
    else:
        pre_features_central = None

    if config.feature_path is not None:  # and config.feat_loss_weight != float(0)
        if config.feat_loss_weight == float(0):
            config.feature_match_train_data = 1
        feature_data = np.load(config.feature_path,allow_pickle=True).item()  #dict, key:feature_layer num, value: corresponding feature data
        feature_data, train_data = create_feature_dataset(config, feature_data,  train_data)
    else:
        feature_data = None
    teacher_list = get_teacher_list(config, copy.deepcopy(model))

    if config.use_initial_model_weight:
        model.load_weights(tf.train.latest_checkpoint(config.model_avg_weight_path)).expect_partial()
    tf.random.set_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    train_x, train_y = zip(*train_data)
    test_x, test_y = zip(*test_data)
    global_test_x, global_test_y = zip(*global_test_data)
    class_rate = train_y.count(0)/train_y.count(1)
    print("class_rate",class_rate)
    train_x, train_y = np.array(train_x),np.array(train_y)
    test_x, test_y = np.array(test_x),np.array(test_y)
    global_test_x, global_test_y = np.array(global_test_x), np.array(global_test_y)

    train_data = tf.data.Dataset.from_tensor_slices((train_x,train_y)).shuffle(len(train_y))
    test_data = tf.data.Dataset.from_tensor_slices(
        (test_x, test_y)).shuffle(len(test_y)).batch(config.batch_size,drop_remainder=True)
    global_test_data = tf.data.Dataset.from_tensor_slices(
        (global_test_x,global_test_y)).shuffle(len(global_test_y)).batch(config.batch_size,drop_remainder=True)

    cls_train_loss = tf.keras.metrics.Mean(name='cls_train_loss')
    cls_train_distance_loss = tf.keras.metrics.Mean(name='cls_train_distance_loss')
    cls_train_classify_loss = tf.keras.metrics.Mean(name='cls_train_classify_loss')
    cls_train_feature_loss = tf.keras.metrics.Mean(name='cls_train_feature_loss')
    cls_test_loss = tf.keras.metrics.Mean(name='cls_test_loss')
    cls_optimizer = tf.keras.optimizers.legacy.Adam(config.learning_rate)
    
    if config.dataset != "elliptic":
        img_loss_fn_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        feature_loss_fn_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        cls_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='cls_train_accuracy')
        cls_test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='cls_test_accuracy')
        cls_global_test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='cls_global_test_accuracy')
    
    else:
        config.importance_rate = (config.train_data_importance_rate * class_rate).astype('float32')
        print("importance_rate",config.importance_rate)
        weights = tf.constant(config.importance_rate)
        cls_train_accuracy = tf.keras.metrics.BinaryAccuracy(name='cls_train_accuracy')
        cls_test_accuracy = tf.keras.metrics.BinaryAccuracy(name='cls_test_accuracy')
        cls_global_test_accuracy = tf.keras.metrics.BinaryAccuracy(name='cls_global_test_accuracy')

    cls_train_elliptic_recall = tf.keras.metrics.Recall(name='elliptic_train_recall')
    cls_train_elliptic_precision = tf.keras.metrics.Precision(name='elliptic_train_precision')
    cls_train_elliptic_f1 = tf.keras.metrics.F1Score(threshold=0.5, name='elliptic_train_f1score')

    cls_test_elliptic_recall = tf.keras.metrics.Recall(name='elliptic_test_recall')
    cls_test_elliptic_precision = tf.keras.metrics.Precision(name='elliptic_test_precision')
    cls_test_elliptic_f1 = tf.keras.metrics.F1Score(threshold=0.5, name='elliptic_test_f1score')

    cls_global_test_elliptic_recall = tf.keras.metrics.Recall(name='elliptic_global_test_recall')
    cls_global_test_elliptic_precision = tf.keras.metrics.Precision(name='elliptic_global_test_precision')
    cls_global_test_elliptic_f1 = tf.keras.metrics.F1Score(threshold=0.5,name='elliptic_global_test_f1score')  

    CLS_train_log_dir = 'logs/CLS_gradient_tape/train'
    CLS_test_log_dir = 'logs/CLS_gradient_tape/test'
    CLS_compare_test_log_dir = 'logs/CLS_gradient_tape/global'

    CLS_train_summary_writer = tf.summary.create_file_writer(CLS_train_log_dir)
    CLS_test_summary_writer = tf.summary.create_file_writer(CLS_test_log_dir)
    CLS_compare_test_acc_summary_writer = tf.summary.create_file_writer(CLS_compare_test_log_dir)

    local_cls_metrics_list = []
    global_cls_metrics_list = []
    global_cls_f1_list = []
    checkpoint_dir = f'./script_tmp/stage_2/{config.dataset}/{config.clients_name}/cls_training_checkpoints/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir+'local')
        os.makedirs(checkpoint_dir+'global')
    if feature_data is None:
        if len(train_y) >  config.batch_size:
            batched_train_data = train_data.batch(config.batch_size,drop_remainder=True)
        else:
            batched_train_data = train_data.batch(config.batch_size)
    elif not config.update_feature_by_epoch:
        all_dataset = list(feature_data.values())
        all_dataset.append(train_data)

    #start training     
    for cur_epoch in range(config.cls_num_epochs+1):
        #train step
        if feature_data:
            if config.update_feature_by_epoch:
                feature_dataset = {}
                feature_idx = np.random.choice(range(config.total_features_num), size=len(train_data), replace=True)
                for k, v in feature_data.items():
                    v = copy.deepcopy(np.array(v, dtype=object)[feature_idx])
                    feature, labels = zip(*v)
                    v = tf.data.Dataset.from_tensor_slices(
                        (np.array(feature), np.array(labels)))
                    feature_dataset[k] = v
                all_dataset = list(feature_dataset.values())
                all_dataset.append(train_data)
            if len(train_data) >  config.batch_size:
                batched_train_data  = tf.data.Dataset.zip(tuple(all_dataset)).shuffle(len(train_data)).batch(config.batch_size,drop_remainder=True)
            else:
                batched_train_data  = tf.data.Dataset.zip(tuple(all_dataset)).shuffle(len(train_data)).batch(config.batch_size)
        for batch_data in batched_train_data:
            if type(batch_data[0]) != tuple:
                x, y = batch_data
            else:  #type(batch_data[0]) == tuple  means batch_data contains feature dataset
                x, y = batch_data[-1]  #last element in batch_data is client train_data
            with tf.GradientTape() as tape:
                loss = 0.0
                predictions = model(x, training=True)
                for teacher in teacher_list:
                    if config.soft_target_loss_weight != float(0):
                        teacher_predictions = teacher(x, training=False)
                        soft_targets = tf.nn.softmax(predictions/config.T)
                        soft_prob = tf.nn.log_softmax(teacher_predictions/config.T)
                        soft_targets_loss = -tf.math.reduce_sum(soft_targets * soft_prob) / soft_prob.shape[0] * (config.T**2)
                        loss += config.soft_target_loss_weight * soft_targets_loss

                    if config.hidden_rep_loss_weight != float(0):
                        teacher_features = teacher.get_features(x)
                        student_feature = model.get_features(x)
                        for layer_num, feature in teacher_features.items():
                            teacher_features = tf.reshape(feature, [-1,])
                            student_feature = tf.reshape(student_feature[layer_num], [-1,])
                            cos_sim = tf.tensordot(teacher_features, student_feature,axes=1)/(tf.linalg.norm(teacher_features)*tf.linalg.norm(student_feature)+0.001)
                            hidden_rep_loss = 1 - cos_sim
                            loss += config.hidden_rep_loss_weight * hidden_rep_loss

                predictions = tf.nn.softmax(predictions)
                if config.dataset == "elliptic":
                    y_true = tf.expand_dims(y, axis=1)
                    predictions = predictions[:,1]
                    predictions = tf.expand_dims(predictions, axis=1)
                    label_loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=predictions, pos_weight=weights)
                    cls_train_elliptic_f1(y_true, predictions)
                    cls_train_elliptic_precision(y_true, predictions)
                    cls_train_elliptic_recall(y_true, predictions)
                else: 
                    label_loss = img_loss_fn_cls(y, predictions)
                cls_train_classify_loss(label_loss)
                loss += label_loss * config.original_cls_loss_weight

                feature_loss, total_feature_loss = 0.0, 0.0
                if pre_features_central is not None:
                    distance_loss = count_features_central_distance(model, pre_features_central, x, y)
                    cls_train_distance_loss(distance_loss)
                    loss += (distance_loss*config.cos_loss_weight)
                if type(batch_data[0]) == tuple and config.feat_loss_weight != float(0):
                    for layer_num, (features, features_label) in zip(feature_data.keys(), batch_data[:-1]):
                        feature_predictions = model.call_2(layer_num, features)   #get prediction through feature in layer_num output
                        feature_predictions = tf.nn.softmax(feature_predictions)
                        if config.dataset != "elliptic":
                            feature_loss = feature_loss_fn_cls(features_label, feature_predictions)
                        else:
                            feature_true = tf.expand_dims(features_label, axis=1)
                            feature_predictions = feature_predictions[:,1]
                            feature_predictions = tf.expand_dims(feature_predictions, axis=1)
                            feature_loss = tf.nn.weighted_cross_entropy_with_logits(labels=feature_true, logits=feature_predictions, pos_weight=weights)
                        loss += (feature_loss*config.feat_loss_weight)
                        total_feature_loss += feature_loss
            gradients = tape.gradient(loss, model.trainable_variables)
            cls_optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            cls_train_loss(loss)
            cls_train_accuracy(y, predictions)

        with CLS_train_summary_writer.as_default():
            tf.summary.scalar('cls_loss_'+config.dataset, cls_train_loss.result(), step=cur_epoch) 
            tf.summary.scalar('cls_accuracy_'+config.dataset, cls_train_accuracy.result(), step=cur_epoch)
            tf.summary.scalar('cls_distance_loss_'+config.dataset, cls_train_distance_loss.result(), step=cur_epoch) 
            tf.summary.scalar('cls_feature_loss_'+config.dataset, cls_train_feature_loss.result(), step=cur_epoch) 
            tf.summary.scalar('cls_classify_loss_'+config.dataset, cls_train_classify_loss.result(), step=cur_epoch) 

        #test step
        for x,y in test_data:
            predictions = model(x, training=False)
            if config.dataset == "elliptic":
                y_true = tf.expand_dims(y, axis=1)
                predictions = tf.nn.softmax(predictions)
                predictions = predictions[:,1]
                predictions = tf.expand_dims(predictions, axis=1)
                loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=predictions, pos_weight=weights)
                cls_test_elliptic_f1(y_true, predictions)
                cls_test_elliptic_precision(y_true, predictions)
                cls_test_elliptic_recall(y_true, predictions)
            else:
                loss = img_loss_fn_cls(y, predictions)
            cls_test_loss(loss)
            cls_test_accuracy(y, predictions)
        # recoed test result on tensorboard
        with CLS_test_summary_writer.as_default():
            tf.summary.scalar('cls_loss_'+config.dataset, cls_test_loss.result(), step=cur_epoch)
            tf.summary.scalar('cls_accuracy_'+config.dataset, cls_test_accuracy.result(), step=cur_epoch)
            tf.summary.scalar('compare_cls_accuracy_'+config.dataset, cls_test_accuracy.result(), step=cur_epoch)
            
        #global test
        for (X_test, Y_test) in global_test_data:
            predictions = model(X_test, training=False)
            if config.dataset == "elliptic":
                y_true = tf.expand_dims(Y_test, axis=1)
                predictions = tf.nn.softmax(predictions)
                predictions = predictions[:,1]
                predictions = tf.expand_dims(predictions, axis=1)
                cls_global_test_elliptic_f1(y_true, predictions)
                cls_global_test_elliptic_precision(y_true, predictions)
                cls_global_test_elliptic_recall(y_true, predictions)
            cls_global_test_accuracy(Y_test, predictions)
        
        with CLS_compare_test_acc_summary_writer.as_default():
            tf.summary.scalar('compare_cls_accuracy_'+config.dataset, cls_global_test_accuracy.result(), step=cur_epoch)
            
        #save feature, feature center in cur_epoch model
        if cur_epoch in config.initial_client_ouput_feat_epochs:
            path = f"script_tmp/stage_2/{config.dataset}/{config.client_name}/assigned_epoch/{cur_epoch}"
            os.makedirs(path)
            model.save_weights(f"{path}/cp-{cur_epoch:04d}.ckpt")
            features_central = get_features_central(train_x,train_y)
            real_features = model.get_features(train_x)
            np.save(f"{path}/feature_center",features_central)
            np.save(f"{path}/real_features",zip(real_features, train_y))
            
        #store metric result 
        if config.model_save_metrics == "acc":
            local_result = cls_test_accuracy.result()
            global_result = cls_global_test_accuracy.result()
        elif config.model_save_metrics == "f1":
            local_result = cls_test_elliptic_f1.result()
            global_result = cls_global_test_elliptic_f1.result()
        local_cls_metrics_list.append(local_result)
        global_cls_metrics_list.append(global_result)
        # save model weight in best loval metrics
        if local_result == max(local_cls_metrics_list):
            del_list = os.listdir(checkpoint_dir+'local')
            for f in del_list:
                file_path = os.path.join(checkpoint_dir+'local', f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            model.save_weights(f"{checkpoint_dir}/local/cp-{cur_epoch:04d}.ckpt")
        # save model weight in best global metrics
        if global_result == max(global_cls_metrics_list):
            del_list = os.listdir(checkpoint_dir+'global')
            for f in del_list:
                file_path = os.path.join(checkpoint_dir+'global', f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            model.save_weights(f"{checkpoint_dir}/global/cp-{cur_epoch:04d}.ckpt")

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}, Global Test Accuracy: {}'
        print (template.format(cur_epoch+1,
                                cls_train_loss.result(), 
                                cls_train_accuracy.result()*100,
                                cls_test_loss.result(), 
                                cls_test_accuracy.result()*100,
                                cls_global_test_accuracy.result()*100,))
        if config.dataset == "elliptic":
            with CLS_train_summary_writer.as_default():
                tf.summary.scalar('cls_f1_'+config.dataset, cls_train_elliptic_f1.result()[0], step=cur_epoch) 
            global_cls_f1_list.append(cls_global_test_elliptic_f1.result())
            template = 'Train Recall: {}, Train Precision: {}, Train F1: {}, Test Recall: {}, Test Precision: {}, Test F1: {}, Global Test Recall: {}, Global Test Precision: {}, Global Test F1: {}'
            print (template.format(cls_train_elliptic_recall.result()*100, 
                                cls_train_elliptic_precision.result()*100,
                                cls_train_elliptic_f1.result()*100,
                                cls_test_elliptic_recall.result()*100, 
                                cls_test_elliptic_precision.result()*100,
                                cls_test_elliptic_f1.result()*100,
                                cls_global_test_elliptic_recall.result()*100,
                                cls_global_test_elliptic_precision.result()*100,
                                cls_global_test_elliptic_f1.result()*100,))

        # Reset metrics every epoch
        cls_train_loss.reset_states()
        cls_test_loss.reset_states()
        cls_train_accuracy.reset_states()
        cls_test_accuracy.reset_states()
        cls_global_test_accuracy.reset_states()
        cls_train_distance_loss.reset_states()
        cls_train_feature_loss.reset_states()
        cls_train_classify_loss.reset_states()
        cls_test_elliptic_recall.reset_states()
        cls_test_elliptic_f1.reset_states()
        cls_test_elliptic_precision.reset_states()
        cls_global_test_elliptic_f1.reset_states()
        cls_global_test_elliptic_precision.reset_states()
        cls_global_test_elliptic_recall.reset_states()
        cls_train_elliptic_recall.reset_states()
        cls_train_elliptic_f1.reset_states()
        cls_train_elliptic_precision.reset_states()

    best_global_metric = max(global_cls_metrics_list)
    max_global_index = global_cls_metrics_list.index(best_global_metric)
    print("max_global_f1_index",max_global_index,"best_global_f1",best_global_metric)
    #load model in best local acc, get and save features
    max_local_acc_model = tf.train.latest_checkpoint(checkpoint_dir+'local')
    model.load_weights(max_local_acc_model)
    features_centre = get_features_central(config, model, train_x,train_y)
    np.save(f"script_tmp/stage_2/{config.dataset}/{config.clients_name}/features_centre",features_centre)
    np.save(f"script_tmp/stage_2/{config.dataset}/{config.clients_name}/real_features",zip(model.get_features(train_x),train_y))

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
    if args.dataset != "elliptic":
        model = Classifier(args)
    else:
        model = ClassifierElliptic(args)
    client_data = data.clients[args.clients_name]
    (train_data, test_data) = client_data
    global_test_data = zip(data.test_x, data.test_y)
    if not os.path.exists(f"script_tmp/stage_2/{args.dataset}/{args.clients_name}/"):
        os.makedirs(f"script_tmp/stage_2/{args.dataset}/{args.clients_name}/")
    record_hparams_file = open(f"./script_tmp/stage_2/{args.dataset}/{args.clients_name}/hparams_record.txt", "wt")
    for key,value in vars(args).items():
        record_hparams_file.write(f"{key}: {value}")
        record_hparams_file.write("\n")
    record_hparams_file.close()
    main(model, train_data, test_data, global_test_data, args)

# export PYTHONPATH=/Users/yangingdai/Downloads/GAN_Tensorflow-Project-Template; python script/main_stage_2.py --batch_size 32 --learning_rate 0.001 --alpha 10 --use_dirichlet_split_data 1 --cls_num_epochs 5 --initial_client 0 --num_clients 5 --feature_dim 50 --dataset elliptic --clients_name clients_1 --sample_ratio 0.1 --features_ouput_layer_list -2 --model_avg_weight_path script_tmp/server/elliptic/clients_1/model_avg --feature_path script_tmp/server/elliptic/clients_1/real_features.npy --feature_center_path script_tmp/server/elliptic/clients_1/feature_center.npy --use_initial_model_weight 1 --cos_loss_weight 1 --feat_loss_weight 1 --feature_match_train_data 1 --update_feature_by_epoch 1 --model_save_metrics f1 --teacher_1_model_path script_tmp/stage_1/elliptic/clients_1/cls_training_checkpoints/local --teacher_2_model_path script_tmp/stage_1/elliptic/clients_2/cls_training_checkpoints/local --teacher_3_model_path script_tmp/stage_1/elliptic/clients_3/cls_training_checkpoints/local --teacher_4_model_path script_tmp/stage_1/elliptic/clients_4/cls_training_checkpoints/local --teacher_5_model_path script_tmp/stage_1/elliptic/clients_5/cls_training_checkpoints/local