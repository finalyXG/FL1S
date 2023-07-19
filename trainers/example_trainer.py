from tqdm import tqdm
import datetime
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import display
import os
import tensorflow as tf
import seaborn as sns
from sklearn.manifold import TSNE

class Trainer:
    def __init__(self, data, discriminator, generator, config):
        self.discriminator = discriminator
        self.generator = generator
        self.config = config
        self.data = data
        self.image_size = config.image_size
        self.gp_weight = config.gp_weight
        self.num_classes = config.num_classes
        self.discriminator_extra_steps = config.discriminator_extra_steps
        self.img_save_path = "./generate_img"
        self.latent_dim = config.latent_dim
        noise = tf.random.normal([self.config.num_classes, self.latent_dim])
        seed_labels = tf.keras.utils.to_categorical(range(self.config.num_classes),self.config.num_classes)
        self.seed = tf.concat(
            [noise, seed_labels], axis=1
            )
        self.disc_optimizer = tf.keras.optimizers.legacy.Adam(self.config.learning_rate)
        self.gen_optimizer = tf.keras.optimizers.legacy.Adam(self.config.learning_rate)

        self.disc_train_loss = tf.keras.metrics.Mean(name='disc_train_loss')
        self.gen_train_loss = tf.keras.metrics.Mean(name='gen_train_loss')  

        self.disc_test_loss = tf.keras.metrics.Mean(name='disc_test_loss')
        self.gen_test_loss = tf.keras.metrics.Mean(name='gen_test_loss')

        self.disc_train_acc =  tf.keras.metrics.CategoricalAccuracy(name='train_accuracy') # for 1 pair 1
        self.disc_test_acc =  tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
        
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
        generator_img_logdir = "logs/train_data/" + current_time

        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        self.generator_img_writer = tf.summary.create_file_writer(generator_img_logdir)
        self.compare_fake_real_img_writer = tf.summary.create_file_writer( "logs/compare_fake_real_img/" + current_time)
   
    def __call__(self):
        if self.init is None:
            self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        return self.init
    
    def gradient_penalty(self, real_images, fake_images):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([self.config.batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train(self):
        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.disc_optimizer,
                                        discriminator_optimizer=self.gen_optimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator)
        #read latest_checkpoint
        # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)) 

        for cur_epoch in range(self.generator.cur_epoch_tensor.numpy(), self.config.num_epochs + 1, 1):
            # train 
            for _ in range(self.config.num_iter_per_epoch):
                self.train_step()
            # recoed training result on tensorboard
            with self.train_summary_writer.as_default():
                tf.summary.scalar('disc_loss', self.disc_train_loss.result(), step=cur_epoch)
                tf.summary.scalar('gen_loss', self.gen_train_loss.result(), step=cur_epoch)
                tf.summary.scalar('disc_acc', self.disc_train_acc.result(), step=cur_epoch)

            self.generator.cur_epoch_tensor.assign_add(1)
            self.discriminator.cur_epoch_tensor.assign_add(1)
            display.clear_output(wait=True)
            print('cur_epoch',cur_epoch+1)

            #get generator output img
            img = self.generate_and_save_images(self.generator,
                            cur_epoch + 1,
                            self.seed)
            
            # show img on tensorboard
            with self.generator_img_writer.as_default():
                tf.summary.image("Generate data", img, step=cur_epoch)

            fake_img = self.generator(self.tsne_seed)
            compare_input = tf.concat([self.data.test_x[:self.config.test_sample_num],fake_img],0)
            fake_real_img_flattened = tf.reshape(compare_input,[2*self.config.test_sample_num,-1])
            with self.compare_fake_real_img_writer.as_default():
                tf.summary.image("compare fake real img", self.generate_tsne_images(fake_real_img_flattened), step=cur_epoch)

            #test
            for(X_test, Y_test) in self.data.test_batched:
                self.test_step(X_test,Y_test)
            # recoed test result on tensorboard
            with self.test_summary_writer.as_default():
                tf.summary.scalar('disc_loss', self.disc_test_loss.result(), step=cur_epoch)
                tf.summary.scalar('gen_loss', self.gen_test_loss.result(), step=cur_epoch)
                tf.summary.scalar('disc_acc', self.disc_test_acc.result(), step=cur_epoch)
            #print
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
            #Save the model every 50 epochs
            if (cur_epoch + 1) % 50 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
        # Generate after the final epoch
        display.clear_output(wait=True)
        img = self.generate_and_save_images(self.generator,
                                self.config.num_epochs + 1,
                                self.seed)
        
        with self.generator_img_writer.as_default():
            tf.summary.image("Generate data", img, step=self.config.num_epochs + 1)
    
    def generate_tsne_images(self,data,generator=False):
        tsne = TSNE(n_components=2, verbose=1, random_state=123)
        z = tsne.fit_transform(data)

        df = pd.DataFrame()
        df["comp-1"] = z[:,0]
        df["comp-2"] = z[:,1]
        buf = io.BytesIO()
        if generator: #show generator output
            sns.scatterplot(x="comp-1", y="comp-2",data=df)
        else: #show discriminator output
            df.loc[:self.config.test_sample_num,'y'] = 1
            df.loc[self.config.test_sample_num:,'y'] = 0
            sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                            palette=sns.color_palette("hls", 2),
                            data=df)
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image
    
    def discriminator_loss(self, real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss

    def generator_loss(self, fake_img):
        return -tf.reduce_mean(fake_img)

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
        # Train the discriminator first.
        images, labels = next(self.data.next_batch(self.config.batch_size))
        for i in range(self.discriminator_extra_steps):
            noise = tf.random.normal([self.config.batch_size, self.latent_dim])
            with tf.GradientTape() as disc_tape:
                generated_images = self.generator(noise, training=True)
                real_output = self.discriminator(images, training=True)
                fake_output = self.discriminator(generated_images, training=True)
                disc_cost = self.discriminator_loss(real_output, fake_output)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(images, generated_images)     
                # Add the gradient penalty to the original discriminator loss
                disc_loss = disc_cost + gp * self.gp_weight  

            # Calculate Gradient
            grad_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            
            self.disc_optimizer.apply_gradients(zip(grad_disc, self.discriminator.trainable_variables))
        # Train the generator
        # Get the latent vector
        noise = tf.random.normal([self.config.batch_size, self.latent_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = self.generator_loss(fake_output)
           
        grad_gen = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(grad_gen, self.generator.trainable_variables))
        
        #produce disc labels and output
        labels = tf.concat([tf.ones_like(real_output), tf.zeros_like(fake_output)], 0)
        output = tf.concat([real_output ,fake_output],0)
   
        return self.gen_train_loss.update_state(gen_loss), self.disc_train_loss.update_state(disc_loss),self.disc_train_acc.update_state(labels,output)

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

        return self.gen_test_loss.update_state(gen_loss), self.disc_test_loss.update_state(disc_loss),self.disc_test_acc.update_state(labels,output)
