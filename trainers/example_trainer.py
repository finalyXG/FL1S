from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf

class ExampleTrainer(BaseTrain):
    def __init__(self, model, data, config):#,logger):
        super(ExampleTrainer, self).__init__( model, data, config)#,logger)
        self.optimizer = tf.keras.optimizers.Adam()
                
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

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
