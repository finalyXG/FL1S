import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten,Multiply,LeakyReLU, Embedding,Dropout, Conv2D, Reshape, BatchNormalization
from tensorflow.keras import Model

class BaseModel(Model):
    def __init__(self, config):
        super(BaseModel, self).__init__()
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
        self.dense_1 = Dense(256, activation=tf.nn.relu)
        self.dense_2 = Dense(512, activation=tf.nn.relu)
        self.dense_3 = Dense(1024, activation=tf.nn.relu)
        self.dense_4 = Dense(28 * 28)
        self.reshape = Reshape((28, 28, 1))

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        return self.reshape(x)
    
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

class AC_Generator(BaseModel):
    def __init__(self,config):
        super(AC_Generator, self).__init__(config=config)
        self.dense_1 = Dense(256, activation=tf.nn.relu)
        self.dense_2 = Dense(512, activation=tf.nn.relu)
        self.dense_3 = Dense(1024, activation=tf.nn.relu)
        self.dense_4 = Dense(28 * 28)
        self.reshape = Reshape((28, 28, 1))

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        return self.reshape(x)