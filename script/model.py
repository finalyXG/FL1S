
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import numpy as np
import os
import argparse
import pickle
import random
import copy
import time
from data_loader.data_generator import DataGenerator
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Multiply, LeakyReLU, Embedding, Dropout,  Reshape, BatchNormalization
from tensorflow.python.training.tracking.data_structures import NoDependency

class Classifier(tf.keras.Model):
    def __init__(self, config):
        super(Classifier, self).__init__()
        tf.random.set_seed(config.random_seed)
        self.config = config
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()
        self.init_saver()
        # init metric
        if config.dataset == "elliptic":
            self.layer_build = self.layer_build_elliptic

        # init kernel_initializer and model layer
        if config.use_same_kernel_initializer:
            self.layer_build("glorot_uniform")
        else:
            if config.clients_name == "clients_1": 
                self.layer_build("glorot_uniform")
            elif config.clients_name == "clients_2":
                self.layer_build("glorot_normal")
            else:
                self.layer_build("random_normal")

        self.cls_train_distance_loss = tf.keras.metrics.Mean(name='cls_train_distance_loss')
        self.loss_fn_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

        if config.stage == 1:
            print("train_step_stage_1")
            self.train_step = self.train_step_stage_1
        elif config.stage == 2:
            self.train_step = self.train_step_stage_2
            self.feature_loss_fn_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
            self.cls_train_classify_loss = tf.keras.metrics.Mean(name='cls_train_classify_loss')
            self.cls_train_feature_loss = tf.keras.metrics.Mean(name='cls_train_feature_loss')

    def set_pre_features_central(self, pre_features_central):
        self.pre_features_central = NoDependency(pre_features_central)

    def set_teacher_list(self, teacher_list):
        self.teacher_list = teacher_list

    def set_feature_data(self, feature_data):
        self.feature_data = NoDependency(feature_data)
        if type(feature_data) == dict:
            self.feature_data_keys = feature_data.keys()

    def set_train_data_num(self, train_data_num):
        self.train_data_num = train_data_num

    def set_train_data(self, train_data):
        self.train_data = train_data
    
    def set_train_x_train_y(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

    def set_loss_weight(self, class_rate):
        self.config.importance_rate = (self.config.train_data_importance_rate * class_rate).astype('float32')
        print("importance_rate",self.config.importance_rate)
        self.loss_weights = tf.constant(self.config.importance_rate)

    def layer_build(self, kernel_initializer):
        self.cov_1 = Conv2D(16, kernel_size=(5, 5), input_shape=(self.config.image_size,self.config.image_size,), kernel_initializer=kernel_initializer)
        self.pool_1 = MaxPooling2D((2, 2))
        self.cov_2 = Conv2D(8, kernel_size=(5, 5), kernel_initializer=kernel_initializer)
        self.pool_2 = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.dense_1 = Dense(512, activation='relu', kernel_initializer=kernel_initializer)
        self.dense_2 = Dense(self.config.num_classes,  kernel_initializer=kernel_initializer)
        self.feature_layers = [self.cov_1, self.pool_1, self.cov_2, self.pool_2, self.flatten,self.dense_1,self.dense_2]

    def layer_build_elliptic(self, kernel_initializer):
        self.dense_1 = Dense(50, activation=tf.nn.leaky_relu , input_shape=(self.config.input_feature_size,), kernel_initializer=kernel_initializer)
        self.dense_2 = Dense(50, activation=tf.nn.leaky_relu , input_shape=(self.config.input_feature_size,), kernel_initializer=kernel_initializer)
        self.dense_3 = Dense(50, activation=tf.nn.leaky_relu , input_shape=(self.config.input_feature_size,), kernel_initializer=kernel_initializer)
        self.dense_4 = Dense(2, kernel_initializer=kernel_initializer)  
        self.feature_layers = [self.dense_1,self.dense_2,self.dense_3,self.dense_4]


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Checkpoint(max_to_keep=tf.Variable(self.config.max_to_keep , dtype=tf.int64)) 
        # save function that saves the checkpoint in the path defined in the config file
    def save(self):
        print("Saving model...")
        self.saver.save( self.config.checkpoint_dir, self.global_step_tensor)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(latest_checkpoint)
            print("Model loaded")

    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.compat.v1.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.compat.v1.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def call(self, x):
        for layer in self.feature_layers:
            x = layer(x)
        return x
    
    def call_2(self, layer_num, x):
        for layer in self.feature_layers[layer_num:]:
            x = layer(x)
        return x

    def train_step_stage_1(self, batch_data):
        x, y = batch_data
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            predictions = tf.nn.softmax(predictions)
            y_true = tf.expand_dims(y, axis=1)
            if self.config.dataset == "elliptic":
                predictions = predictions[:,1]
                predictions = tf.expand_dims(predictions, axis=1)
                loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=predictions, pos_weight=self.loss_weights)
            else: 
                loss = self.loss_fn_cls(y, predictions)
            
            if self.config.whether_initial_feature_center:
                distance_loss = self.count_features_central_distance(self.config.initial_feature_center, x, y)
                self.cls_train_distance_loss(distance_loss)
                loss += (distance_loss* self.config.cos_loss_weight)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        result = {}
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
                result[metric.name] = metric.result()

        for metric in self.compiled_metrics._metrics:
            metric.update_state(y_true, predictions)
            result[metric.name] = metric.result()
        return result

    def train_step_stage_2(self, batch_data):
        if type(batch_data[0]) != tuple:
            x, y = batch_data
        else:  #type(batch_data[0]) == tuple  means batch_data contains feature dataset
            x, y = batch_data[-1]  #last element in batch_data is client train_data
        with tf.GradientTape() as tape:
            loss = 0.0
            predictions = self(x, training=True)
            for teacher in self.teacher_list:
                if self.config.soft_target_loss_weight != float(0):
                    teacher_predictions = teacher(x, training=False)
                    soft_targets = tf.nn.softmax(predictions/self.config.T)
                    soft_prob = tf.nn.log_softmax(teacher_predictions/self.config.T)
                    soft_targets_loss = -tf.math.reduce_sum(soft_targets * soft_prob) / soft_prob.shape[0] * (self.config.T**2)
                    loss += self.config.soft_target_loss_weight * soft_targets_loss
                
                if self.config.hidden_rep_loss_weight != float(0):
                    teacher_features = teacher.get_features(x)
                    student_feature = self.get_features(x)
                    for layer_num, feature in teacher_features.items():
                        teacher_features = tf.reshape(feature, [-1,])
                        student_feature = tf.reshape(student_feature[layer_num], [-1,])
                        cos_sim = tf.tensordot(teacher_features, student_feature,axes=1)/(tf.linalg.norm(teacher_features)*tf.linalg.norm(student_feature)+0.001)
                        hidden_rep_loss = 1 - cos_sim
                        loss += self.config.hidden_rep_loss_weight * hidden_rep_loss

            predictions = tf.nn.softmax(predictions)
            y_true = tf.expand_dims(y, axis=1)
            if self.config.dataset == "elliptic":
                predictions = predictions[:,1]
                predictions = tf.expand_dims(predictions, axis=1)
                label_loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=predictions, pos_weight=self.loss_weights)
            else: 
                label_loss = self.loss_fn_cls(y, predictions)
            loss += label_loss * self.config.original_cls_loss_weight
            feature_loss, total_feature_loss = 0.0, 0.0
            if self.pre_features_central is not None:
                distance_loss = self.count_features_central_distance(self.pre_features_central, x, y)
                self.cls_train_distance_loss(distance_loss)
                loss += (distance_loss*self.config.cos_loss_weight)
            if type(batch_data[0]) == tuple and self.config.feat_loss_weight != float(0):
                for layer_num, (features, features_label) in zip(self.feature_data_keys, batch_data[:-1]):
                    feature_predictions = self.call_2(layer_num, features)   #get prediction through feature in layer_num output
                    feature_predictions = tf.nn.softmax(feature_predictions)
                    if self.config.dataset != "elliptic":
                        feature_loss = self.feature_loss_fn_cls(features_label, feature_predictions)
                    else:
                        feature_true = tf.expand_dims(features_label, axis=1)
                        feature_predictions = feature_predictions[:,1]
                        feature_predictions = tf.expand_dims(feature_predictions, axis=1)
                        feature_loss = tf.nn.weighted_cross_entropy_with_logits(labels=feature_true, logits=feature_predictions, pos_weight=self.loss_weights)
                    loss += (feature_loss*self.config.feat_loss_weight)
                    total_feature_loss += feature_loss

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        result = {}
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
                result[metric.name] = metric.result()

        for metric in self.compiled_metrics._metrics:
            metric.update_state(y_true, predictions)
            result[metric.name] = metric.result()
        return result
    
    def test_step(self, data):
        x, y = data
        predictions = self(x, training=False)
        y_true = tf.expand_dims(y, axis=1)
        if self.config.dataset == "elliptic":
            predictions = tf.nn.softmax(predictions)
            predictions = predictions[:,1]
            predictions = tf.expand_dims(predictions, axis=1)
            loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=predictions, pos_weight=self.loss_weights)
        else:
            loss = self.loss_fn_cls(y, predictions)

        result = {}
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
                result[metric.name] = metric.result()
        for metric in self.compiled_metrics._metrics:
            metric.update_state(y_true, predictions)
            result[metric.name] = metric.result()
        return result

    def get_features(self, inputs):
        feature_list = {}
        for features_ouput_layer_num in self.config.features_ouput_layer_list:
            x = tf.identity(inputs)
            for layer in self.feature_layers[:features_ouput_layer_num]:
                x = layer(x)
            feature_list[features_ouput_layer_num] = x
        return feature_list

    def count_features_central_distance(self, features_central, x, y):
        feature = self.get_features(x)
        accumulate_loss = 0
        for k,v in feature.items():
            for vector,label in zip(v,y): 
                pre_vector = features_central[k][label.numpy()]
                vector = tf.reshape(vector, [-1,])
                pre_vector = tf.reshape(pre_vector, [-1,])
                cos_sim = tf.tensordot(vector/(tf.linalg.norm(vector)+1e-6), pre_vector/(tf.linalg.norm(pre_vector)+1e-6),axes=1)#/(tf.linalg.norm(vector)*tf.linalg.norm(pre_vector)+1e-9) #tf.tensordot(vector, pre_vector,axes=1)
                accumulate_loss += 1 - cos_sim
        return accumulate_loss / len(y)

    def get_features_central(self, x, y):  
        feature_output_layer_feature_avg_dic = {i:{} for i in self.config.features_ouput_layer_list}
        for label in set(y):
            label_index = np.where(y==label)
            feature_list = self.get_features(x[label_index])
            for k,v in feature_list.items():
                avg_feature = tf.reduce_mean(v, axis=0) 
                feature_output_layer_feature_avg_dic[k][label] = avg_feature
        return feature_output_layer_feature_avg_dic