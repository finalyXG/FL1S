import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Multiply, LeakyReLU, Embedding, Dropout,  Reshape, BatchNormalization
from tensorflow.keras import Model
import numpy as np
import time
class BaseModel(Model):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        tf.random.set_seed(config.random_seed)
        self.config = config
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()
        # self.build_model()
        self.init_saver()

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
            # self.increment_cur_epoch_tensor = tf.compat.v1.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        # with tf.variable_scope('global_step'):
        with tf.compat.v1.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')


# class Classifier(BaseModel):
#     def __init__(self,config):
#         super(Classifier, self).__init__(config=config)
#         self.num_classes = config.num_classes
#         self.cov_1 = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28,28,)  , kernel_initializer='glorot_normal')
#         self.cov_2 = Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='glorot_normal') 
        
    #     self.flatten = Flatten()
    #     self.dense_1 = Dense(49, activation='relu', kernel_initializer='glorot_normal')
    #     self.dense_2 = Dense(16, activation='relu',kernel_initializer='glorot_normal')
    #     self.dense_3 = Dense(self.num_classes, activation='softmax', kernel_initializer='glorot_normal')
    #     self.feature_layers = [self.cov_1, self.cov_2, self.flatten, self.dense_1,self.dense_2, self.dense_3]

    # def call(self, inputs):
    #     x = self.cov_1(inputs)
    #     x = self.cov_2(x)
    #     x = self.flatten(x)
    #     x = self.dense_1(x)
    #     x = self.dense_2(x)
    #     return self.dense_3(x)
    
    # def call_2(self, x):
    #     #x: intial clients features
    #     for layer in self.feature_layers[self.config.features_ouput_layer:]:
    #         x = layer(x)
    #     return x
    
    # def get_features(self, x):
    #     for layer in self.feature_layers[:self.config.features_ouput_layer]:
    #         x = layer(x)
    #     return x

# Classifier in fed_cvae
class Classifier(BaseModel):
    def __init__(self,config):
        super(Classifier, self).__init__(config=config)
        if config.use_same_kernel_initializer:
            print("use_same_kernel_initializer")
            self.layer_build("glorot_uniform")
        else:
            if config.clients_name == "clients_1": 
                print("for clients_1 use default initial model")
                self.layer_build("glorot_uniform")
            elif config.clients_name == "clients_2":
                print("for clients_2 use glorot_normal")
                self.layer_build("glorot_normal")
            else:
                print("for other client use random_normal")
                self.layer_build("random_normal")
        self.feature_layers = [self.cov_1, self.pool_1, self.cov_2, self.pool_2, self.flatten,self.dense_1,self.dense_2]

    def layer_build(self, kernel_initializer):
        self.cov_1 = Conv2D(32, kernel_size=(5, 5), input_shape=(self.config.image_size,self.config.image_size,), kernel_initializer=kernel_initializer)
        self.pool_1 = MaxPooling2D((2, 2))
        self.cov_2 = Conv2D(64, kernel_size=(5, 5), kernel_initializer=kernel_initializer)
        self.pool_2 = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.dense_1 = Dense(512, activation='relu', kernel_initializer=kernel_initializer)
        self.dense_2 = Dense(self.config.num_classes, activation='softmax', kernel_initializer=kernel_initializer)

    def call(self, inputs):
        x = self.cov_1(inputs)
        x = self.pool_1(x)
        x = self.cov_2(x)
        x = self.pool_2(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        return self.dense_2(x)
    
    def call_2(self, x):
        #for other client
        for layer in self.feature_layers[self.config.features_ouput_layer:]:
            x = layer(x)
        return x
    
    def get_features(self, x):
        for layer in self.feature_layers[:self.config.features_ouput_layer]:
            x = layer(x)
        return x
    
class ClassifierElliptic(BaseModel):
    def __init__(self,config):
        super(ClassifierElliptic, self).__init__(config=config)
        if config.use_same_kernel_initializer:
            print("use_same_kernel_initializer")
            self.layer_build("glorot_uniform")
        else:
            if config.clients_name == "clients_1": 
                print("for clients_1 use default initial model")
                self.layer_build("glorot_uniform")
            elif config.clients_name == "clients_2":
                print("for clients_2 use glorot_normal")
                self.layer_build("glorot_normal")
            else:
                print("for other client use random_normal")
                self.layer_build("random_normal")
        self.feature_layers = [self.dense_1,self.dense_2]#,self.dense_4,self.dense_5,self.dense_6,self.dense_3,self.dense_7

    # def layer_build(self, kernel_initializer):
    #     output_bias = tf.keras.initializers.Constant(self.config.elliptic_initial_bias)
    #     self.dense_1 = Dense(512, activation=tf.nn.leaky_relu , input_shape=(self.config.input_feature_size,), kernel_initializer=kernel_initializer)
    #     self.dense_2 = Dense(256, activation=tf.nn.leaky_relu, kernel_initializer=kernel_initializer )
    #     self.dense_3 = Dense(128, activation=tf.nn.leaky_relu, kernel_initializer=kernel_initializer)
    #     # self.dense_4 = Dense(64, activation=tf.nn.leaky_relu, kernel_initializer=kernel_initializer)
    #     # self.dense_5 = Dense(64, activation=tf.nn.leaky_relu, kernel_initializer=kernel_initializer)
    #     # self.dense_6 = Dense(64, activation=tf.nn.leaky_relu, kernel_initializer=kernel_initializer)
    #     self.dense_7 = Dense(1, activation='sigmoid', kernel_initializer=kernel_initializer)  #,bias_initializer=output_bias

    # def call(self, x):
    #     x = self.dense_1(x)
    #     x = self.dense_2(x)
    #     x = self.dense_3(x)
    #     # x = self.dense_4(x)
    #     # x = self.dense_5(x)
    #     # x = self.dense_6(x)
    #     return self.dense_7(x)

    def layer_build(self, kernel_initializer):
        output_bias = tf.keras.initializers.Constant(self.config.elliptic_initial_bias)
        self.dense_1 = Dense(50, activation=tf.nn.leaky_relu , input_shape=(self.config.input_feature_size,), kernel_initializer=kernel_initializer)
        self.dense_2 = Dense(1, activation='sigmoid', kernel_initializer=kernel_initializer)#, bias_initializer=output_bias

    def call(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)
    
    def call_2(self, x):
        #for other client
        for layer in self.feature_layers[self.config.features_ouput_layer:]:
            x = layer(x)
        return x
    
    def get_features(self, x):
        for layer in self.feature_layers[:self.config.features_ouput_layer]:
            x = layer(x)
        return x
    
class C_Discriminator(BaseModel):
    def __init__(self,config):
        super(C_Discriminator, self).__init__(config=config)
        self.flatten = Flatten()
        self.dense_1 = Dense(512, activation=tf.nn.leaky_relu )
        self.dense_2 = Dense(256, activation=tf.nn.leaky_relu )
        self.dense_3 = Dense(128, activation=tf.nn.leaky_relu)
        self.dense_4 = Dense(1)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return self.dense_4(x)

class C_Generator(BaseModel):
    def __init__(self,config):
        super(C_Generator, self).__init__(config=config)
        self.dense_1 = Dense(config.generator_dense1_dim, activation=tf.nn.relu)
        self.dense_2 = Dense(config.generator_dense2_dim, activation=tf.nn.relu)
        self.dense_3 = Dense(config.generator_dense3_dim, activation=tf.nn.relu)
        self.dense_4 = Dense(config.latent_dim)

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return self.dense_4(x)
    
class AC_Discriminator(BaseModel):
    def __init__(self,config):
        super(AC_Discriminator, self).__init__(config=config)
        self.flatten = Flatten()
        self.dense_1 = Dense(512, activation=tf.nn.leaky_relu )
        self.dense_2 = Dense(256, activation=tf.nn.leaky_relu )
        self.dense_3 = Dense(128, activation=tf.nn.leaky_relu)
        self.dense_4 = Dense(1+self.config.num_classes)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        return x[:, :1], x[:, 1:]
    
# for generator features
class AC_Generator(BaseModel):
    def __init__(self,config):
        super(AC_Generator, self).__init__(config=config)
        self.dense_1 = Dense(32, activation=tf.nn.relu)
        self.dense_2 = Dense(64, activation=tf.nn.relu)
        self.dense_3 = Dense(128, activation=tf.nn.relu)
        self.dense_4 = Dense(config.latent_dim)
    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return self.dense_4(x)


## for generator img
# class AC_Generator(BaseModel):
#     def __init__(self,config):
#         super(AC_Generator, self).__init__(config=config)
#         self.dense_1 = Dense(config.generator_dense1_dim, activation=tf.nn.relu)
#         self.dense_2 = Dense(config.generator_dense2_dim, activation=tf.nn.relu)
#         self.dense_3 = Dense(config.generator_dense3_dim, activation=tf.nn.relu)
#         self.dense_4 = Dense(28 * 28)
#         self.reshape = Reshape((28, 28, 1))

#     def call(self, inputs):
#         x = self.dense_1(inputs)
#         x = self.dense_2(x)
#         x = self.dense_3(x)
#         x = self.dense_4(x)
#         return self.reshape(x)