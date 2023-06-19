import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
class ExampleModel(Model):
    def __init__(self, config):
        super(ExampleModel, self).__init__(config)
        self.config = config
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()
 
        self.build_model()
        self.init_saver()



    def call(self, x):
        x = self.conv1(tf.expand_dims(x, axis=-1))
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)



    def build_model(self):
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)
    

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

