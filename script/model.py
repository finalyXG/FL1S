
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
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Multiply, LeakyReLU, Embedding, Dropout,  concatenate, Reshape, BatchNormalization, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.python.training.tracking.data_structures import NoDependency

class reduction_number(keras.metrics.Metric):
    def __init__(self, name = 'rd', **kwargs):
        super(reduction_number, self).__init__(**kwargs)
        self.rd = self.add_weight('rd', initializer = 'zeros')

    def update_state(self, y_true, y_pred):
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        index = tf.where(tf.equal(y_true, 1)).numpy()  
        index = tf.squeeze(index) 
        p_min_s = tf.sort(tf.gather(y_pred,index), direction='ASCENDING')[0]
        rd = len(y_pred[y_pred<p_min_s])
        self.rd.assign_add(tf.reduce_sum(tf.cast(rd, self.dtype)))
    
    def reset_state(self):
        self.rd.assign(0)

    def result(self):
        return self.rd 

class reduction_rate(keras.metrics.Metric):
    def __init__(self, name = 'rr', **kwargs):
        super(reduction_rate, self).__init__(**kwargs)
        self.rr = self.add_weight('rr', initializer = 'zeros')

    def update_state(self, y_true, y_pred):
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        index = tf.where(tf.equal(y_true, 1)).numpy()  
        index = tf.squeeze(index) 
        p_min_s = tf.sort(tf.gather(y_pred,index), direction='ASCENDING')[0]
        rd = len(y_pred[y_pred<p_min_s])
        target_0_num = len(tf.where(tf.equal(y_true, 0)).numpy())
        rr = rd/target_0_num
        self.rr.assign_add(tf.reduce_sum(tf.cast(rr, self.dtype)))
    
    def reset_state(self):
        self.rr.assign(0)

    def result(self):
        return self.rr 
    
class smaller_half_number(keras.metrics.Metric):
    def __init__(self, name = 'smaller_half_number', **kwargs):
        super(smaller_half_number, self).__init__(**kwargs)
        self.smaller_half_number = self.add_weight('smaller_half_number', initializer = 'zeros')

    def update_state(self, y_true, y_pred):
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        index = tf.where(tf.equal(y_true, 1)).numpy()  
        index = tf.squeeze(index) 
        target_1_score = tf.gather(y_pred,index)
        smaller_half_number = len(target_1_score[target_1_score<0.5])
        self.smaller_half_number.assign_add(tf.reduce_sum(tf.cast(smaller_half_number, self.dtype)))
    
    def reset_state(self):
        self.smaller_half_number.assign(0)

    def result(self):
        return self.smaller_half_number 
    
class epochs(keras.metrics.Metric):
    def __init__(self, name = 'epochs', **kwargs):
        super(epochs, self).__init__(**kwargs)
        self.epochs = self.add_weight('epochs', initializer = 'zeros')

    def update_state(self, epochs):
        self.epochs.assign_add(tf.reduce_sum(tf.cast(epochs, self.dtype)))
    
    def reset_state(self):
        self.epochs.assign(0)

    def result(self):
        return self.epochs

class score0_target1_num(keras.metrics.Metric):
    def __init__(self, name = 'score0_target1_num', **kwargs):
        super(score0_target1_num, self).__init__(**kwargs)
        self.score0_target1_num = self.add_weight('score0_target1_num', initializer = 'zeros')

    def update_state(self, y_true, y_pred):
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        index = tf.where(tf.equal(y_true, 1)).numpy()  
        index = tf.squeeze(index) 
        sort_1_score = tf.sort(tf.gather(y_pred,index), direction='ASCENDING')
        score0_target1_num = len(sort_1_score[sort_1_score==0])
        self.score0_target1_num.assign_add(tf.reduce_sum(tf.cast(score0_target1_num, self.dtype)))

    def reset_state(self):
        self.score0_target1_num.assign(0)

    def result(self):
        return self.score0_target1_num

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
        self.epochs = tf.Variable(initial_value=0, trainable=False)
        # init metric
        if config.dataset == "elliptic":
            if config.whether_use_transformer_model:
                self.layer_build = self.layer_build_elliptic_transformer
            else:
                self.layer_build = self.layer_build_elliptic
        elif config.dataset == "real_data":
            self.layer_build = self.layer_build_real_data
            self.call = self.call_real_data
            self.call_2 = self.call_2_real_data
            self.get_features = self.get_features_real_data

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
        elif self.config.dataset == "real_data":
            self.feature_data_keys = self.config.real_data_feature_type

    def set_train_data_num(self, train_data_num):
        self.train_data_num = train_data_num

    def set_train_data(self, train_data):
        self.train_data = train_data
    
    def set_train_x_train_y(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

    def set_version_num(self, version_num):
        self.version_num = version_num

    def set_loss_weight(self, class_rate):
        if self.config.client_loss_weight != int(0):
            self.loss_weights = self.config.client_loss_weight
        else:
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

    def layer_build_elliptic_transformer(self, kernel_initializer):
        self.dense_1 = keras.layers.Dense(50, activation="relu", kernel_initializer=kernel_initializer)
        # self.dense_2 = keras.layers.Dense(50, activation="relu", kernel_initializer=kernel_initializer)
        self.attention1 = keras.layers.MultiHeadAttention(num_heads=4, key_dim=4, dropout=self.config.dropout_rate, name = "attention1")
        # self.attention2 = keras.layers.MultiHeadAttention(num_heads=4, key_dim=4, output_shape=2, name = "attention2")
        self.dense_2 = keras.layers.Dense(2, activation="linear", kernel_initializer=kernel_initializer)
        self.feature_layers = [self.dense_1,  self.attention1, self.dense_2] #self.attention1, self.dropout1,

    def layer_build_real_data(self, kernel_initializer):
        depth = 10
        if self.config.total_test_size > self.config.batch_size:#must larger than batch_size and  global_sample_num
            pos_encoding_dim_1 = self.config.total_test_size
        else:
            pos_encoding_dim_1 = self.config.batch_size   
        self.pos_encoding = self.positional_encoding(self.config.max_transaction_num, depth)
        self.pos_encoding = tf.repeat(self.pos_encoding[tf.newaxis, :, :], pos_encoding_dim_1, axis=0)
        self.country_embedding = tf.keras.layers.Embedding(input_dim=self.config.feature_space.features['tx.country'].preprocessor.vocabulary_size(), output_dim=4, name = "tx.country_embedding",trainable=True)
        self.transaction_embedding = tf.keras.layers.Embedding(input_dim=self.config.feature_space.features['tx.transaction'].preprocessor.vocabulary_size(), output_dim=4, name = "tx.transaction_embedding",trainable=True)
        self.segment_embedding = tf.keras.layers.Embedding(input_dim=self.config.feature_space.features['segment'].preprocessor.vocabulary_size(), output_dim=2, name = "segment_embedding",trainable=True)
        self.family_embedding = tf.keras.layers.Embedding(input_dim=self.config.feature_space.features['family'].preprocessor.vocabulary_size(), output_dim=2, name = "family_embedding",trainable=True)
        self.p2_embedding = tf.keras.layers.Embedding(input_dim=self.config.feature_space.features['pp.p2'].preprocessor.vocabulary_size(), output_dim=2, name = "pp.p2_embedding",trainable=True)
        self.embedding_layers = [self.country_embedding,self.transaction_embedding,self.segment_embedding,self.family_embedding, self.p2_embedding]

        self.country_attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=2, dropout=0.2, output_shape=(2), name = "tx.country_attention")
        self.transaction_attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=2, dropout=0.2, output_shape=(2),  name = "tx.transaction_attention")
        self.amount_attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=2, dropout=0.2, output_shape=(2),  name = "tx.amount_attention")
        self.p1_attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=2, dropout=0.2, output_shape=(2),  name = "pp.p1_attention")
        self.p2_attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=2, dropout=0.2, output_shape=(2),  name = "pp.p2_attention")
        self.dict_feature_attention_layers = [self.country_attention,self.transaction_attention,self.amount_attention,self.p1_attention,self.p2_attention]

        self.tx_attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=2, dropout=0.2, output_shape=(2), name = "tx_attention")
        self.pp_attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=2, dropout=0.2, output_shape=(2), name = "pp_attention")

        self.attention2 = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=2, dropout=0.2, output_shape=(64),  name = "attention2")
        self.dense_1 = keras.layers.Dense(2, activation="linear", kernel_initializer=kernel_initializer)

        self.feature_layers = [self.attention2, self.dense_1] 
    
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
            if "attention" in layer.name:
                x = tf.expand_dims(x, axis=-1)
                x = layer(x, x)
                x = tf.reshape(x, (x.shape[0], -1))
            else:
                x = layer(x)
        return x

    def positional_encoding(self, max_length, depth):
        depth = depth/2
        positions = np.arange(max_length)[:, np.newaxis]     # (seq, 1)
        depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
        angle_rates = 1 / (10000**depths)         # (1, depth)
        angle_rads = positions * angle_rates      # (pos, depth)
        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1) 
        return tf.cast(pos_encoding, dtype=tf.float32)

    def concat_real_data_sequence(self, bd):
        #embedding str type faeture
        tmp = {}
        for k, v in bd.items():
            tmp[k] = v
        
        for embedding_layer in self.embedding_layers:
            k = embedding_layer.name.split('_')[0]
            tmp[k] = embedding_layer(bd[k])
            if len(tmp[k].shape) > 3:  #for subsequence
                tmp[k] = tf.squeeze(tmp[k])
            tmp[k] += self.pos_encoding[:tmp[k].shape[0], :tmp[k].shape[1],:tmp[k].shape[2]]

        tx_features_list = []
        pp_features_list = []
        other_feature_list = []
        for k,v in tmp.items():
            if k not in self.config.disable_lv1_namaes and k.split(".")[0] not in self.config.disable_lv1_namaes:
                if k.split(".")[0] == 'tx':
                    if k.split(".")[1] not in self.config.disable_tx_namaes:
                        tx_features_list.append(v)
                elif k.split(".")[0] == 'pp':
                    if k.split(".")[1] not in self.config.disable_pp_namaes:
                        pp_features_list.append(v)
                else:
                    other_feature_list.append(v)

        if self.config.model_type == 'MHA02':
            attention2_input_list = other_feature_list
            if tx_features_list:
                tx_features = tf.concat(tx_features_list, -1)
                tx_features = tf.concat(tx_features_list, -1)
                tx_output = self.tx_attention(tx_features[:,:1,:], tx_features, attention_mask=self.attention_mask['tx.amount'])
                attention2_input_list.append(tx_output)

            if pp_features_list:
                pp_features = tf.concat(pp_features_list, -1)
                pp_output = self.pp_attention(pp_features[:,:1,:], pp_features, attention_mask=self.attention_mask['pp.p1'])
                attention2_input_list.append(pp_output)
            attention2_input = tf.concat(attention2_input_list, 1)
        elif self.config.model_type == 'MHA01':
            dict_feature_ouput = []
            for attention_layer in self.dict_feature_attention_layers:
                k = attention_layer.name.split("_")[0]
                if 'tx' in self.config.disable_lv1_namaes and k.split(".")[0] == 'tx':
                    continue
                if 'pp' in self.config.disable_lv1_namaes and k.split(".")[0] == 'pp':
                    continue
                if k.split(".")[0] == 'tx' and k.split(".")[1] in self.config.disable_tx_namaes:
                    continue
                if k.split(".")[0] == 'pp' and k.split(".")[1] in self.config.disable_pp_namaes:
                    continue
                output = attention_layer(tmp[k][:,:1,:],tmp[k], attention_mask=self.attention_mask[k])
                dict_feature_ouput.append(output)
            attention2_input = tf.concat(dict_feature_ouput + other_feature_list, 1)
        return attention2_input

    def call_real_data(self, bd):
        attention2_input = self.concat_real_data_sequence(bd)
        output = self.attention2(attention2_input[:,:1,:], attention2_input)
        output = tf.squeeze(output)
        output = self.dense_1(output)
        return output

    def call_2(self, layer_num, x):
        for layer in self.feature_layers[layer_num:]:
            if "attention" in layer.name:
                x = tf.expand_dims(x, axis=-1)
                x = layer(x, x)
                x = tf.reshape(x, (x.shape[0], -1))
            else:
                x = layer(x)
        return x

    def call_2_real_data(self, layer_num, x):
        if layer_num == 'before_attention':
            x = self.attention2(x[:,:1,:], x)
            x = tf.squeeze(x)
        x = self.dense_1(x)
        return x
    
    def pre_preccess_real_data(self, batch_data):
        self.attention_mask = {}
        for k,v in batch_data.items():
            if isinstance(v,tf.RaggedTensor):
                self.attention_mask[k] = tf.ones(tf.shape(v), dtype=tf.int32)
                self.attention_mask[k] = self.attention_mask[k].to_tensor(default_value=0)
                if v.dtype == 'string':
                    batch_data[k] = batch_data[k].to_tensor(default_value='-1')
                else:
                    batch_data[k] = batch_data[k].to_tensor(default_value=-1)
        batch_data = self.config.feature_space(batch_data)

        y = tf.identity(tf.squeeze(batch_data['target']))
        y = tf.cast(y, dtype=tf.float32)
        del batch_data['target']
        #set float type feature
        for k,v in batch_data.items():
            if v.dtype != 'string':
                batch_data[k] = tf.cast(tf.squeeze(batch_data[k] ),dtype=tf.float32)
                batch_data[k] = tf.expand_dims(batch_data[k], axis=-1)  #(32, 197, 1)
        return batch_data,y
    
    def train_step_stage_1(self, batch_data):
        if self.config.dataset == "real_data":
            x, y = self.pre_preccess_real_data(batch_data)
        else:
            x, y = batch_data
        result = {}
        if self.config.skip_if_all_0 and len(y[y==0]) == self.config.batch_size:
            for metric in self.metrics:
                result[metric.name] = metric.result()
            for metric in self.compiled_metrics._metrics:
                result[metric.name] = metric.result()
            return result
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            predictions = predictions / self.config.temperature
            predictions = tf.nn.softmax(predictions)
            y_true = tf.expand_dims(y, axis=1)
            if self.config.dataset == "elliptic" or self.config.dataset == "real_data":   
                cfc = tf.keras.losses.CategoricalFocalCrossentropy(alpha = [1,self.config.client_loss_weight], gamma=self.config.gamma) #reduction=tf.keras.losses.Reduction.NONE
                y_true_one_hot = tf.keras.utils.to_categorical(y_true, self.config.num_classes)
                loss = cfc(y_true=y_true_one_hot, y_pred=predictions)
                predictions = predictions[:,1]
                predictions = tf.expand_dims(predictions, axis=1)
                # loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=predictions, pos_weight=self.loss_weights)
            else: 
                loss = self.loss_fn_cls(y, predictions)
            
            if self.config.whether_initial_feature_center:
                distance_loss = self.count_features_central_distance(self.config.initial_feature_center, x, y)
                self.cls_train_distance_loss(distance_loss)
                loss += (distance_loss* self.config.cos_loss_weight)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
                result[metric.name] = metric.result()

        for metric in self.compiled_metrics._metrics:
            if metric.name != "reduction_number" and metric.name != "reduction_rate" and metric.name != "score0_target1_num" and metric.name != "smaller_half_number" and metric.name != "epochs": 
                metric.update_state(y_true, predictions)
                result[metric.name] = metric.result()
            elif metric.name == "epochs":
                metric.update_state(self.epochs)
                result[metric.name] = metric.result()
        return result

    def train_step_stage_2(self, batch_data):
        if self.config.dataset == "real_data":
            x, y = self.pre_preccess_real_data(batch_data[-1])
        else:
            if type(batch_data[0]) != tuple:
                x, y = batch_data
            else:  #type(batch_data[0]) == tuple  means batch_data contains feature dataset
                x, y = batch_data[-1]  #last element in batch_data is client train_data
        result = {}
        if self.config.skip_if_all_0 and len(y[y==0]) == self.config.batch_size:
            for metric in self.metrics:
                result[metric.name] = metric.result()
            return result
        with tf.GradientTape() as tape:
            loss = 0.0
            predictions = self(x, training=True)
            y_change = np.array(tf.identity(y))
            for teacher in self.teacher_list:
                if self.config.dataset == "real_data":
                    teacher.attention_mask = self.attention_mask
                if self.config.change_ground_truth:
                    teacher_predictions = teacher(x, training=False)
                    teacher_predictions = np.argmax(teacher_predictions, axis=1)
                    index = tf.where(tf.equal(teacher_predictions, 1)).numpy()  
                    index = tf.squeeze(index, axis=1) 
                    y_change[index.numpy()] = 1

                if self.config.soft_target_loss_weight != float(0):
                    teacher_predictions = teacher(x, training=False)
                    soft_targets = tf.nn.softmax(predictions/self.config.T)
                    soft_prob = tf.nn.log_softmax(teacher_predictions/self.config.T)
                    soft_targets_loss = -tf.math.reduce_sum(soft_targets * soft_prob) / soft_prob.shape[0] * (self.config.T**2)
                    loss += self.config.soft_target_loss_weight * soft_targets_loss
                
                if self.config.hidden_rep_loss_weight != float(0):
                    teacher_features = teacher.get_features(x)
                    student_features = self.get_features(x)
                    for layer_num, t_feature in teacher_features.items():
                        t_feature = tf.reshape(t_feature, [-1,])
                        s_feature = tf.reshape(student_features[layer_num], [-1,])
                        cos_sim = tf.tensordot(t_feature, s_feature,axes=1)/(tf.linalg.norm(t_feature)*tf.linalg.norm(s_feature)+0.001)
                        hidden_rep_loss = 1 - cos_sim
                        loss += self.config.hidden_rep_loss_weight * hidden_rep_loss
            predictions = predictions / self.config.temperature
            predictions = tf.nn.softmax(predictions)
            y_true = tf.expand_dims(y, axis=1)
            if self.config.dataset == "elliptic" or self.config.dataset == "real_data":
                cfc = tf.keras.losses.CategoricalFocalCrossentropy(alpha = [1,self.config.client_loss_weight], gamma=self.config.gamma)
                if self.config.change_ground_truth:
                    index = tf.where(tf.equal(y, 1)).numpy() 
                    self.original_label1_num += len(index)
                    index = tf.where(tf.equal(y_change, 1)).numpy()  
                    self.changed_label1_num += len(index)

                    index = tf.where(tf.equal(y, 0)).numpy() 
                    self.original_label0_num += len(index)
                    index = tf.where(tf.equal(y_change, 0)).numpy()  
                    self.changed_label0_num += len(index)

                    y_change = tf.expand_dims(y_change, axis=1)
                    y_true_one_hot = tf.keras.utils.to_categorical(y_change, self.config.num_classes)
                else:   
                    y_true_one_hot = tf.keras.utils.to_categorical(y_true, self.config.num_classes)
                label_loss = cfc(y_true=y_true_one_hot, y_pred=predictions)
                predictions = predictions[:,1]
                predictions = tf.expand_dims(predictions, axis=1)
                # label_loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=predictions, pos_weight=self.loss_weights)
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
                    if self.config.dataset != "elliptic" or self.config.dataset != "real_data":
                        feature_loss = self.feature_loss_fn_cls(features_label, feature_predictions)
                    else:
                        feature_true = tf.expand_dims(features_label, axis=1)

                        feature_true = tf.keras.utils.to_categorical(feature_true, self.config.num_classes)
                        feature_loss = cfc(y_true=feature_true, y_pred=feature_predictions)

                        # feature_predictions = feature_predictions[:,1]
                        # feature_predictions = tf.expand_dims(feature_predictions, axis=1)
                        # feature_loss = tf.nn.weighted_cross_entropy_with_logits(labels=feature_true, logits=feature_predictions, pos_weight=self.loss_weights)
                    loss += (feature_loss*self.config.feat_loss_weight)
                    total_feature_loss += feature_loss

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
                result[metric.name] = metric.result()

        for metric in self.compiled_metrics._metrics:
            if metric.name != "reduction_number" and metric.name != "reduction_rate" and metric.name != "score0_target1_num" and metric.name != "smaller_half_number" and metric.name != "epochs": 
                metric.update_state(y_true, predictions)
                result[metric.name] = metric.result()
            elif metric.name == "epochs":
                metric.update_state(self.epochs)
                result[metric.name] = metric.result()
        return result
    
    def test_step(self, data):
        if self.config.dataset == "real_data":
            x, y = self.pre_preccess_real_data(data)
        else:
            if len(data) == 2:
                x, y = data
            else:
                x,y = data[0]
        predictions = self(x, training=False)
        y_true = tf.expand_dims(y, axis=1)
        if self.config.dataset == "elliptic" or self.config.dataset == "real_data":
            predictions = tf.nn.softmax(predictions)
            cfc = tf.keras.losses.CategoricalFocalCrossentropy(alpha = [1,self.config.client_loss_weight],  gamma=self.config.gamma)
            y_true_one_hot = tf.keras.utils.to_categorical(y_true, self.config.num_classes)
            loss = cfc(y_true=y_true_one_hot, y_pred=predictions)
            predictions = predictions[:,1]
            predictions = tf.expand_dims(predictions, axis=1)
            # loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=predictions, pos_weight=self.loss_weights)
        else:
            loss = self.loss_fn_cls(y, predictions)

        result = {}
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
                result[metric.name] = metric.result()
        for metric in self.compiled_metrics._metrics:
            if metric.name != "epochs":
                metric.update_state(y_true, predictions)
                result[metric.name] = metric.result()
            else:
                metric.update_state(self.epochs)
                result[metric.name] = metric.result()
        return result

    def get_features(self, inputs):
        feature_list = {}
        for features_ouput_layer_num in self.config.features_ouput_layer_list:
            x = tf.identity(inputs)
            for layer in self.feature_layers[:features_ouput_layer_num]:
                if "attention" in layer.name:
                    x = tf.expand_dims(x, axis=-1)
                    x = layer(x, x)
                    x = tf.reshape(x, (x.shape[0], -1))
                else:
                    x = layer(x)
            feature_list[features_ouput_layer_num] = x
        return feature_list

    def get_features_real_data(self, data):
        if isinstance(data, dict):  ##for batched data
            features = self.concat_real_data_sequence(data)
            feature_dict = {}
            for type_name in self.config.real_data_feature_type:
                if type_name == 'before_attention':
                    feature_dict[type_name] = features
                if type_name == 'after_attention':
                    after_attention_features = self.attention2(features[:,:1,:], features)
                    after_attention_features = tf.squeeze(after_attention_features)
                    feature_dict[type_name] = after_attention_features
            return feature_dict
        else:  #data == whole unproccess train_data
            data_num = data.reduce(tf.constant(0), lambda x, _: x + 1)
            data = data.batch(tf.cast(data_num, dtype= tf.int64))  #set as one batch
            for num, i in enumerate(data):
                assert(num == 0)
                bd, y = self.pre_preccess_real_data(i)
                features = self.concat_real_data_sequence(bd)
            feature_dict = {}
            for type_name in self.config.real_data_feature_type:
                if type_name == 'before_attention':
                    feature_dict[type_name] = features
                if type_name == 'after_attention':
                    after_attention_features = self.attention2(features[:,:1,:], features)
                    after_attention_features = tf.squeeze(after_attention_features)
                    feature_dict[type_name] = after_attention_features
            for k,v in feature_dict.items():
                feature_dict[k] = tf.reshape(v,[v.shape[0],-1])
            return feature_dict, y

    def count_features_central_distance(self, features_central, x, y):
        feature = self.get_features(x)
        accumulate_loss = 0
        for k,v in feature.items():
            for vector,label in zip(v,y): 
                if self.config.stage == 1:
                    if self.config.dataset != "real_data":
                        label = label.numpy()
                    pre_vector = features_central[k][label]
                elif self.config.stage == 2: 
                    pre_vector = features_central[label.numpy()]  
                vector = tf.reshape(vector, [-1,])
                pre_vector = tf.reshape(pre_vector, [-1,])
                cos_sim = tf.tensordot(vector/(tf.linalg.norm(vector)+1e-6), pre_vector/(tf.linalg.norm(pre_vector)+1e-6),axes=1)
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
    
class GAN(tf.keras.Model):
    def __init__(self, config, discriminator, generator, model):
        super().__init__()
        self.config = config
        self.discriminator = discriminator
        self.generator = generator
        self.cls = model
        self.latent_dim = config.latent_dim   #generator input dim
        self.d_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = tf.keras.metrics.Mean(name="g_loss")

    def compile(self, d_optimizer, g_optimizer, loss_fn_binary, loss_fn_categorical, binary_accuracy, categorical_accuracy, run_eagerly):
        super().compile(run_eagerly = run_eagerly)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn_binary = loss_fn_binary
        self.loss_fn_categorical = loss_fn_categorical
        self.binary_accuracy = binary_accuracy
        self.categorical_accuracy = categorical_accuracy

    def generate_fake_features(self, y):
        noise = tf.random.normal([len(y), self.latent_dim])
        y = tf.expand_dims(y, axis=1)
        seed = tf.concat(
            [noise, y], axis=1
            )
        fake_features = self.generator(seed)
        return fake_features
    
    #features_version
    def gradient_penalty(self, real_features, generated_features):
        """ features_version
        Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([real_features.shape[0], 1], 0.0, 1.0)
        diff = generated_features - real_features
        interpolated = real_features + alpha * diff
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)
            pred, _ = pred[:, :1], pred[:, 1:]

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @tf.function
    # Modify Train step for GAN
    def train_step(self,data):
        if self.config.dataset == "real_data":
            # x, y = self.cls.pre_preccess_real_data(data)
            real_features, y = data
        else:
            x, y = data
            #use cls model to generate real features
            real_features = self.cls.get_features(x)
        y = tf.expand_dims(y, axis=1)
        if len(real_features) == 1:
            if self.config.dataset != "real_data":
                real_features = real_features[self.config.features_ouput_layer_list[0]]
            else:
                real_features = real_features[self.config.real_data_feature_type[0]]
        # Train the discriminator first.
        for i in range(self.config.discriminator_extra_steps):
            noise = tf.random.normal([y.shape[0], self.latent_dim])
            random_vector_labels = tf.concat(
            [noise, y], axis=1
            )
            with tf.GradientTape() as disc_tape:
                generated_features = self.generator(random_vector_labels, training=True)
                # Combine them with real images. Note that we are concatenating the labels
                # with these images here.
                combined_features = tf.concat(
                    [real_features, generated_features], axis=0
                )

                # Assemble labels discriminating real from fake images.
                binary_labels = tf.concat(  
                    [tf.ones((y.shape[0], 1)), tf.zeros((y.shape[0], 1))], axis=0
                )

                # Add random noise to the labels - important trick!
                # labels += 0.05 * tf.random.uniform(tf.shape(labels))
                
                category_label = tf.concat(  
                    [y, y], axis=0
                )
            
                predictions = self.discriminator(combined_features, training=True)
                binary_predictions, category_predictions = predictions[:, :1], predictions[:, 1:]
                disc_binary_cost = self.loss_fn_binary(binary_labels,binary_predictions)
                disc_categorical_cost = self.loss_fn_categorical(category_label, category_predictions)
    
                # Calculate the gradient penalty
                gp = self.gradient_penalty(real_features, generated_features)    
                # Add the gradient penalty to the original discriminator loss
                disc_loss = disc_binary_cost +  disc_categorical_cost  +  gp * self.config.gp_weight

            # Calculate Gradient
            grad_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(grad_disc, self.discriminator.trainable_variables))

        # Train the generator
        # Get the latent vector
        noise = tf.random.normal([y.shape[0], self.latent_dim])
        random_vector_labels = tf.concat(
            [noise, y], axis=1
            )
        # Assemble labels that say "all real images". 
        misleading_labels = tf.ones((y.shape[0], 1))
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_features = self.generator(random_vector_labels, training=True)
            
            fake_predictions = self.discriminator(generated_features, training=True)
            binary_fake_predictions, category_fake_predictions = fake_predictions[:, :1], fake_predictions[:, 1:]

            gen_binary_loss = self.loss_fn_binary(misleading_labels, binary_fake_predictions) #lead classify to 
            gen_categorical_loss  = self.loss_fn_categorical(y, category_fake_predictions) 
            gen_loss = gen_binary_loss + gen_categorical_loss

        grad_gen = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grad_gen, self.generator.trainable_variables))

        # Update metrics and return their value.
        self.d_loss_tracker.update_state(disc_loss)
        self.g_loss_tracker.update_state(gen_loss)
        self.binary_accuracy.update_state(binary_labels, binary_predictions)
        self.categorical_accuracy.update_state(category_label, category_predictions)
        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
            "binary_accuracy": self.binary_accuracy.result(),
            "categorical_accuracy": self.categorical_accuracy.result(),
        }
    
    @tf.function
    def test_step(self,data):
        if self.config.dataset == "real_data":
            real_features, y = data
        else:
            x, y = data
            #use cls model to generate real features
            real_features = self.cls.get_features(x)

        if len(real_features) == 1 and self.config.dataset != "real_data":
                real_features = real_features[self.config.features_ouput_layer_list[0]]
        noise = tf.random.normal([y.shape[0], self.latent_dim])
        y = tf.expand_dims(y, axis=1)
        random_vector_labels = tf.concat(
            [noise, y], axis=1
            )

        generated_features = self.generator(random_vector_labels, training=False)
        combined_features = tf.concat(
            [real_features, generated_features], axis=0
        )

        # Assemble labels discriminating real from fake images.
        binary_labels = tf.concat(  #set real img classify to zero
            [tf.ones((y.shape[0], 1)), tf.zeros((y.shape[0], 1))], axis=0
        )
        misleading_labels = tf.ones((y.shape[0], 1))

        category_label = tf.concat(  
                    [y, y], axis=0
                )
        predictions = self.discriminator(combined_features, training=False)
        binary_predictions, category_predictions = predictions[:, :1], predictions[:, 1:]
        
        #using fake img's output to count generator loss
        gen_binary_loss = self.loss_fn_binary(misleading_labels,binary_predictions[y.shape[0]:])
        gen_categorical_loss  = self.loss_fn_categorical(y, category_predictions[y.shape[0]:]) 
        gen_loss = gen_binary_loss + gen_categorical_loss
        
        disc_binary_cost = self.loss_fn_binary(binary_labels, binary_predictions)
        disc_categorical_cost = self.loss_fn_categorical(category_label, category_predictions)
        # Calculate the gradient penalty
        gp = self.gradient_penalty(real_features, generated_features)     
        # Add the gradient penalty to the original discriminator loss
        disc_loss = disc_binary_cost + disc_categorical_cost + gp * self.config.gp_weight  
        
        # Update metrics and return their value.
        self.d_loss_tracker.update_state(disc_loss)
        self.g_loss_tracker.update_state(gen_loss)
        self.binary_accuracy.update_state(binary_labels, binary_predictions)
        self.categorical_accuracy.update_state(category_label, category_predictions)
        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
            "binary_accuracy": self.binary_accuracy.result(),
            "categorical_accuracy": self.categorical_accuracy.result(),
        }