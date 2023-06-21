from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import os
import tensorflow as tf
from tensorflow.keras import Model

class Trainer(BaseTrain):
    def __init__(self, data, discriminator, generator, latent_dim, config):#,logger):
        super(Trainer, self).__init__(data, discriminator, generator, config)#,logger)
        self.latent_dim = latent_dim
        self.seed = tf.random.normal([self.config.num_examples_to_generate, self.latent_dim])

        self.disc_optimizer = tf.keras.optimizers.legacy.Adam(self.config.learning_rate)
        self.gen_optimizer = tf.keras.optimizers.legacy.Adam(self.config.learning_rate)

        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)


    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)

        cur_it = self.model.global_step_tensor.numpy()
        summaries_dict = {
            'loss': loss,
            'acc': acc,
        }
        print("loss::::",loss)
        print("acc::::",acc)
        # self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save()

    # def train_step(self):
    #     batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
    #     feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        
    #     _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy],
    #                                  feed_dict=feed_dict)
    #     return loss, acc
    
    @tf.function
    def train_step(self):
        images, labels = next(self.data.next_batch(self.config.batch_size))
    
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = loss_object(labels, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return self.train_loss(loss), self.train_accuracy(labels, predictions)
