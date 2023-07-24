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
from absl import app, flags
from tensorboard.plugins.hparams import api as hp
import shutil

class Trainer:
    def __init__(self, data, discriminator, generator, config, hparams):
        self.cls = cls
        self.discriminator = discriminator
        self.generator = generator
        self.config = config
        self.data = data
        hparams = {h.name: hparams[h] for h in hparams}
        self.version = hparams['gan_version']
        self.batch_size = hparams['batch_size']
        self.cls_batch_size = config.cls_batch_size
        self.learning_rate = hparams['learning_rate']
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
        self.disc_optimizer = tf.keras.optimizers.legacy.Adam(self.learning_rate)
        self.gen_optimizer = tf.keras.optimizers.legacy.Adam(self.learning_rate)
        self.loss_fn_binary = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.loss_fn_categorical = tf.keras.losses.CategoricalCrossentropy(from_logits=True) #int
        self.loss_fn_cls = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        self.disc_train_loss = tf.keras.metrics.Mean(name='disc_train_loss')
        self.gen_train_loss = tf.keras.metrics.Mean(name='gen_train_loss')  

        self.disc_test_loss = tf.keras.metrics.Mean(name='disc_test_loss')
        self.gen_test_loss = tf.keras.metrics.Mean(name='gen_test_loss')
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
        alpha = tf.random.normal([self.batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff


        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            if self.version == "ACGAN":
                pred, pred_class = self.discriminator(interpolated, training=True)
            else:
                pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def trainGAN(self, real_features):
        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.disc_optimizer,
                                        discriminator_optimizer=self.gen_optimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator)
        #read latest_checkpoint
        # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)) 

        for cur_epoch in range(self.generator.cur_epoch_tensor.numpy(), self.config.GAN_num_epochs + 1, 1):
            # train 
            for _ in range(self.config.num_iter_per_epoch):
                self.trainGAN_step()
            # recoed training result on tensorboard
            with self.train_summary_writer.as_default():
                tf.summary.scalar('disc_loss', self.disc_train_loss.result(), step=cur_epoch)
                tf.summary.scalar('gen_loss', self.gen_train_loss.result(), step=cur_epoch)

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
            
            with self.compare_fake_real_img_writer.as_default():
                tf.summary.image("compare fake real img", self.generate_tsne_images(), step=cur_epoch)

            #test
            for(X_test, Y_test) in self.data.test_batched:
                self.testGAN_step(X_test,Y_test)
            # recoed test result on tensorboard
            with self.test_summary_writer.as_default():
                tf.summary.scalar('disc_loss', self.disc_test_loss.result(), step=cur_epoch)
                tf.summary.scalar('gen_loss', self.gen_test_loss.result(), step=cur_epoch)
            
            #print
            template = 'Epoch {}, Generator Loss: {}, Discriminator Loss: {}, Generator Test Loss: {}, Discriminator Test Loss: {}'
            print (template.format(cur_epoch+1,
                                    self.gen_train_loss.result(), 
                                    self.disc_train_loss.result(), 
                                    self.gen_test_loss.result(), 
                                    self.disc_test_loss.result()))
            if cur_epoch != self.config.GAN_num_epochs:
                # Reset metrics every epoch
                self.gen_train_loss.reset_states()
                self.disc_train_loss.reset_states()
                self.gen_test_loss.reset_states()
                self.disc_test_loss.reset_states()

            #Save the model every 50 epochs
            if (cur_epoch + 1) % 50 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
        
        self.trainGAN_step()
        # Generate after the final epoch
        display.clear_output(wait=True)
        img = self.generate_and_save_images(self.generator,
                                self.config.GAN_num_epochs + 1,
                                self.seed)
        
        with self.generator_img_writer.as_default():
            tf.summary.image("Generate data", img, step=self.config.GAN_num_epochs + 1)
        
        return self.disc_test_loss.result(), self.gen_test_loss.result()

    def generate_tsne_images(self):
        noise = tf.random.normal([self.config.test_sample_num, self.latent_dim])
        seed = tf.concat(
            [noise, self.data.test_y[:self.config.test_sample_num]], axis=1
            )
        fake_img = self.generator(seed)
        compare_input = tf.concat([self.data.test_x[:self.config.test_sample_num],fake_img],0)
        data = tf.reshape(compare_input,[2*self.config.test_sample_num,-1])

        tsne = TSNE(n_components=2, verbose=1, random_state=123)
        z = tsne.fit_transform(data)

        df = pd.DataFrame()
        df["comp-1"] = z[:,0]
        df["comp-2"] = z[:,1]
        buf = io.BytesIO()
        #show discriminator output
        df.loc[:self.config.test_sample_num-1,'y'] = "real"
        df.loc[self.config.test_sample_num:,'y'] = "fake"

        df.loc[:self.config.test_sample_num-1,"classes"] = self.data._test_y[:self.config.test_sample_num] #_test_y using 0-9 lable
        df.loc[self.config.test_sample_num:,"classes"] = self.data._test_y[:self.config.test_sample_num]
        sns.scatterplot(x="comp-1", y="comp-2", hue=df.classes.tolist(), style=df.y.tolist(),
                        palette=sns.color_palette("hls", 10),
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

        fig = plt.figure(figsize=(3, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(3, 4, i+1)
            plt.imshow(predictions[i, :, :, 0]  * 255, cmap='gray')#* 127.5 + 127.5, cmap='gray')
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
    def trainGAN_step(self):
        # Train the discriminator first.
        real_images, one_hot_labels = next(self.data.next_batch(self.batch_size))
        # Add dummy dimensions to the labels so that they can be concatenated with
        # the images. This is for the discriminator.
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[self.image_size * self.image_size]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, self.image_size, self.image_size, self.num_classes)
        )
        if self.version == "CGAN":
            real_images = tf.concat([real_images, image_one_hot_labels], -1)

        for i in range(self.discriminator_extra_steps):
            noise = tf.random.normal([self.batch_size, self.latent_dim])
            random_vector_labels = tf.concat(
            [noise, one_hot_labels], axis=1
            )
            with tf.GradientTape() as disc_tape:
                generated_images = self.generator(random_vector_labels, training=True)
                # Combine them with real images. Note that we are concatenating the labels
                # with these images here.
                if self.version == "CGAN":
                    generated_images = tf.concat([generated_images, image_one_hot_labels], -1)
                    
                combined_images = tf.concat(
                    [real_images, generated_images], axis=0
                )

                # Assemble labels discriminating real from fake images.
                labels = tf.concat(  #set real img classify to zero
                    [tf.ones((self.batch_size, 1)), tf.zeros((self.batch_size, 1))], axis=0
                )
                if self.version == "ACGAN":
                    double_one_hot_labels = tf.concat(  
                        [one_hot_labels, one_hot_labels], axis=0
                    )
                    predictions, class_labels = self.discriminator(combined_images, training=True)
                    disc_binary_cost = self.loss_fn_binary(labels,predictions)
                    disc_categorical_cost = self.loss_fn_categorical(double_one_hot_labels, class_labels)
                    # Calculate the gradient penalty
                    gp = self.gradient_penalty(real_images, generated_images)    
                    # Add the gradient penalty to the original discriminator loss
                    disc_loss = disc_binary_cost +  disc_categorical_cost  +  gp * self.gp_weight
                else: #CGAN
                    predictions = self.discriminator(combined_images, training=True)
                    disc_cost = self.loss_fn_binary(labels, predictions)
                    gp = self.gradient_penalty(real_images, generated_images)     
                    disc_loss = disc_cost + gp * self.gp_weight  

            # if not tf.reduce_any(tf.math.is_nan(disc_loss)):
            # Calculate Gradient
            grad_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(grad_disc, self.discriminator.trainable_variables))

        # Train the generator
        # Get the latent vector
        noise = tf.random.normal([self.batch_size, self.latent_dim])
        random_vector_labels = tf.concat(
            [noise, one_hot_labels], axis=1
            )
        # Assemble labels that say "all real images". 
        misleading_labels = tf.ones((self.batch_size, 1))
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(random_vector_labels, training=True)
            
            if self.version == "ACGAN":
                fake_predictions, fake_class_labels = self.discriminator(generated_images, training=True)
                gen_binary_loss = self.loss_fn_binary(misleading_labels, fake_predictions) #lead classify to 
                gen_categorical_loss  = self.loss_fn_categorical(one_hot_labels, fake_class_labels) 
                gen_loss = gen_binary_loss + gen_categorical_loss
            else:
                fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
                fake_output = self.discriminator(fake_image_and_labels, training=True)
                gen_loss = self.loss_fn_binary(misleading_labels, fake_output)
        grad_gen = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(grad_gen, self.generator.trainable_variables))
        return self.gen_train_loss.update_state(gen_loss), self.disc_train_loss.update_state(disc_loss)

    @tf.function
    def testGAN_step(self,real_images, one_hot_labels):
        noise = tf.random.normal([self.batch_size, self.latent_dim])
        random_vector_labels = tf.concat(
            [noise, one_hot_labels], axis=1
            )
        # Add dummy dimensions to the labels so that they can be concatenated with
        # the images. This is for the discriminator.
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[self.image_size * self.image_size]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, self.image_size, self.image_size, self.num_classes)
        )

        generated_images = self.generator(random_vector_labels, training=False)
        if self.version == 'CGAN':
            generated_images = tf.concat([generated_images, image_one_hot_labels], -1)
            real_images = tf.concat([real_images, image_one_hot_labels], -1)
        
        combined_images = tf.concat(
            [real_images, generated_images], axis=0
        )

        # Assemble labels discriminating real from fake images.
        labels = tf.concat(  #set real img classify to zero
            [tf.ones((self.batch_size, 1)), tf.zeros((self.batch_size, 1))], axis=0
        )
        misleading_labels = tf.ones((self.batch_size, 1))

        if self.version == 'ACGAN':
            double_one_hot_labels = tf.concat(  
                        [one_hot_labels, one_hot_labels], axis=0
                    )
            predictions, class_labels  = self.discriminator(combined_images, training=False)
            #using fake img's output to count generator loss
            gen_binary_loss = self.loss_fn_binary(misleading_labels,predictions[self.batch_size:])
            gen_categorical_loss  = self.loss_fn_categorical(one_hot_labels, class_labels[self.batch_size:]) 
            gen_loss = gen_binary_loss + gen_categorical_loss
            
            disc_binary_cost = self.loss_fn_binary(labels, predictions)
            disc_categorical_cost = self.loss_fn_categorical(double_one_hot_labels, class_labels)
            # Calculate the gradient penalty
            gp = self.gradient_penalty(real_images, generated_images)     
            # Add the gradient penalty to the original discriminator loss
            disc_loss = disc_binary_cost + disc_categorical_cost + gp * self.gp_weight  
        else: #cgan
            predictions_combined_images = self.discriminator(combined_images, training=False)
            disc_cost = self.loss_fn_binary(labels, predictions_combined_images)
            gp = self.gradient_penalty(real_images, generated_images)     
            disc_loss = disc_cost + gp * self.gp_weight 
            
            fake_output = self.discriminator(generated_images, training=False)
            gen_loss = self.loss_fn_binary(misleading_labels,fake_output)
            
        return self.gen_test_loss.update_state(gen_loss), self.disc_test_loss.update_state(disc_loss)
