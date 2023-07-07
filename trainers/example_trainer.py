from tqdm import tqdm
import datetime
import io
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import os
import tensorflow as tf
from tensorflow.keras import Model
class Trainer:
    def __init__(self, data, discriminator, generator, latent_dim, config):
        self.discriminator = discriminator
        self.generator = generator
        self.config = config
        self.data = data

        self.img_save_path = "./generate_img"
        self.latent_dim = latent_dim
        self.seed = tf.random.normal([self.config.num_examples_to_generate, self.latent_dim])
        
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)     
        self.disc_optimizer = tf.keras.optimizers.legacy.Adam(self.config.learning_rate)
        self.gen_optimizer = tf.keras.optimizers.legacy.Adam(self.config.learning_rate)

        self.disc_train_loss = tf.keras.metrics.Mean(name='disc_train_loss')
        self.gen_train_loss = tf.keras.metrics.Mean(name='gen_train_loss')  

        self.disc_test_loss = tf.keras.metrics.Mean(name='disc_test_loss')
        self.gen_test_loss = tf.keras.metrics.Mean(name='gen_test_loss')

        self.disc_train_acc =  tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.disc_test_acc =  tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
        logdir = "logs/train_data/" + current_time
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        self.file_writer = tf.summary.create_file_writer(logdir)

    def __call__(self):
        if self.init is None:
            self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        return self.init
    
    def train(self):
        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.disc_optimizer,
                                        discriminator_optimizer=self.gen_optimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator)

        for cur_epoch in range(self.generator.cur_epoch_tensor.numpy(), self.config.num_epochs + 1, 1):
            for _ in range(self.config.num_iter_per_epoch):
                self.train_step()
            with self.train_summary_writer.as_default():
                tf.summary.scalar('disc_loss', self.disc_train_loss.result(), step=cur_epoch)
                tf.summary.scalar('gen_loss', self.gen_train_loss.result(), step=cur_epoch)
                tf.summary.scalar('disc_acc', self.disc_train_acc.result(), step=cur_epoch)

            self.generator.cur_epoch_tensor.assign_add(1)
            self.discriminator.cur_epoch_tensor.assign_add(1)
            display.clear_output(wait=True)
            print('cur_epoch',cur_epoch+1)

            img = self.generate_and_save_images(self.generator,
                            cur_epoch + 1,
                            self.seed)
            # Convert to image and log
            with self.file_writer.as_default():
                tf.summary.image("Generate data", img, step=cur_epoch)
            
            for(X_test, Y_test) in self.data.test_batched:
                self.test_step(X_test, Y_test)
            with self.test_summary_writer.as_default():
                tf.summary.scalar('disc_loss', self.disc_test_loss.result(), step=cur_epoch)
                tf.summary.scalar('gen_loss', self.gen_test_loss.result(), step=cur_epoch)
                tf.summary.scalar('disc_acc', self.disc_test_acc.result(), step=cur_epoch)
            template = 'Epoch {}, Generator Loss: {}, Discriminator Loss: {}, Discriminator Accuracy: {}, Generator Test Loss: {}, Discriminator Test Loss: {}, Discriminator Test Accuracy: {}'
            print (template.format(cur_epoch+1,
                                    self.gen_train_loss.result(), 
                                    self.disc_train_loss.result(), 
                                    self.disc_train_acc.result()*100,
                                    self.gen_test_loss.result(), 
                                    self.disc_test_loss.result(), 
                                    self.disc_test_acc.result()*100))

            # Reset metrics every epoch
            self.gen_train_loss.reset_states()
            self.disc_train_loss.reset_states()
            self.gen_test_loss.reset_states()
            self.disc_test_loss.reset_states()
            self.disc_train_acc.reset_states()
            self.disc_test_acc.reset_states()
            # Save the model every 15 epochs
            # if (cur_epoch + 1) % 15 == 0:
            #     checkpoint.save(file_prefix = checkpoint_prefix)

         # Generate after the final epoch
        display.clear_output(wait=True)
        img = self.generate_and_save_images(self.generator,
                                self.config.num_epochs + 1,
                                self.seed)
        
        with self.file_writer.as_default():
            tf.summary.image("Generate data", img, step=self.config.num_epochs + 1)
        
    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
        fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.loss_fn(tf.ones_like(fake_output), fake_output)


    def generate_and_save_images(self,model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image
    
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
        
        #produce disc labels and output
        labels = tf.concat([tf.ones_like(real_output), tf.zeros_like(fake_output)], 0)
        output = tf.concat([real_output ,fake_output],0)
   
        return self.gen_train_loss(gen_loss), self.disc_train_loss(disc_loss),self.disc_train_acc(labels,output)

    @tf.function
    def test_step(self,images, labels):
        noise = tf.random.normal([self.config.batch_size, self.latent_dim])

        generated_images = self.generator(noise, training=False)
        real_output = self.discriminator(images, training=False)
        fake_output = self.discriminator(generated_images, training=False)
        
        gen_loss = self.generator_loss(fake_output)
        disc_loss = self.discriminator_loss(real_output, fake_output)

        #produce disc labels and output
        labels = tf.concat([tf.ones_like(real_output), tf.zeros_like(fake_output)], 0)
        output = tf.concat([real_output ,fake_output],0)

        return self.gen_test_loss(gen_loss), self.disc_test_loss(disc_loss),self.disc_test_acc(labels,output)
