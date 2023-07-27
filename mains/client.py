import tensorflow as tf
from models.example_model import Classifier, C_Discriminator,C_Generator,AC_Discriminator,AC_Generator
from trainers.example_trainer import Trainer
from tensorboard.plugins.hparams import api as hp
import os
import time
def clients_main(config,client_data, all_test_x,all_test_y,client_name):
    HP_BATCH_SIZE = hp.HParam("batch_size", hp.Discrete([64]))
    HP_LEARNING_RATE = hp.HParam("learning_rate", hp.Discrete([0.0005]))#hp.RealInterval(0.001, 0.1))
    # HP_GAN_VERSION = hp.HParam("gan_version", hp.Discrete(['ACGAN']))
    HPARAMS = [
        HP_BATCH_SIZE,
        HP_LEARNING_RATE,
        # HP_GAN_VERSION,
    ]
    METRICS = [
        hp.Metric(
            "generator_test_loss",
            display_name="generator_test_loss",
        ),
        hp.Metric(
            "discriminator_test_loss",
            display_name="discriminator_test_loss",
        ),
    ]
    with tf.summary.create_file_writer(os.path.join(config.logdir,client_name)).as_default():
        hp.hparams_config(hparams=HPARAMS, metrics=METRICS)

    def run(run_dir, hparams):
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            # create your data generator
            # if hparams[HP_GAN_VERSION] == "CGAN":
            #     generator = C_Generator(config)
            #     discriminator = C_Discriminator(config)
            # else:
            #     generator = AC_Generator(config)
            #     discriminator = AC_Discriminator(config)
            cls = Classifier(config)
            generator = AC_Generator(config)
            discriminator = AC_Discriminator(config)

            trainer = Trainer( client_name, client_data,all_test_x,all_test_y, cls, discriminator, generator,config,hparams)
            local_acc, global_acc = trainer.train_cls()
            print("after train cls")
            disc_test_loss, gen_test_loss, fake_features = trainer.trainGAN()

            tf.summary.scalar("discriminator_test_loss", disc_test_loss, step=1)
            tf.summary.scalar("generator_test_loss", gen_test_loss, step=1)
        return local_acc, global_acc, fake_features
    
    session_num = 0
    # create an instance of the model
    # for model_version in HP_GAN_VERSION.domain.values:
    for batch_size in HP_BATCH_SIZE.domain.values:
        for learning_rate in HP_LEARNING_RATE.domain.values:
            hparams = {
                # HP_GAN_VERSION: model_version,
                HP_BATCH_SIZE: batch_size,
                HP_LEARNING_RATE: learning_rate,
            }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            local_acc, global_acc, fake_features = run(os.path.join(config.logdir,client_name,run_name), hparams)
            session_num += 1
    return local_acc, global_acc, fake_features
