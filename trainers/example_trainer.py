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

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
        fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.loss_fn(tf.ones_like(fake_output), fake_output)

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
    def generate_and_save_images(self,model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig('./generate_img/image_at_epoch_{:04d}.png'.format(epoch))
        plt.close()
        # plt.show()
    
    # @tf.function: The below function is completely Tensor Code
    # Good for optimization
    @tf.function
    # Modify Train step for GAN
    def train_step(self):
        images, _ = next(self.data.next_batch(self.config.batch_size))
        noise = tf.random.normal([self.config.batch_size, self.latent_dim])
        # Define the loss function
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        # Calculate Gradient
        grad_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        grad_gen = gen_tape.gradient(gen_loss, self.generator.trainable_variables)

        # Optimization Step: Update Weights & Learning Rate
        self.disc_optimizer.apply_gradients(zip(grad_disc, self.discriminator.trainable_variables))
        self.gen_optimizer.apply_gradients(zip(grad_gen, self.generator.trainable_variables))
        
        return gen_loss, disc_loss