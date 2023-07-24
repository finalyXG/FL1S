import tensorflow as tf
from data_loader.data_generator import DataGenerator
from models.example_model import Classifier, C_Discriminator,C_Generator,AC_Discriminator,AC_Generator
from trainers.example_trainer import Trainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from tensorboard.plugins.hparams import api as hp

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    
    HP_DISCRIMINATOR_EXTRA_STEPS = hp.Hparam("discriminator_extra_steps",hp.Discrete([1,3,5]))
    HP_BATCH_SIZE = hp.HParam("batch_size", hp.Discrete([64]))
    HP_LEARNING_RATE = hp.HParam("learning_rate", hp.Discrete([0.0005]))#hp.RealInterval(0.001, 0.1))
    HP_GAN_VERSION = hp.HParam("gan_version", hp.Discrete(['ACGAN']))

    HPARAMS = [
        HP_DISCRIMINATOR_EXTRA_STEPS,
        HP_BATCH_SIZE,
        HP_LEARNING_RATE,
        HP_GAN_VERSION,
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
    with tf.summary.create_file_writer(config.logdir).as_default():
        hp.hparams_config(hparams=HPARAMS, metrics=METRICS)

    def run(run_dir, hparams):
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            # create your data generator
            data = DataGenerator(config, hparams[HP_BATCH_SIZE])

            if hparams[HP_GAN_VERSION] == "CGAN":
                generator = C_Generator(config)
                discriminator = C_Discriminator(config)
            else:
                generator = AC_Generator(config)
                discriminator = AC_Discriminator(config)

            trainer = Trainer( data, discriminator, generator, config,hparams)
            trainer.train_cls()
            real_features = trainer.get_real_features()

            # here you train your model
            disc_test_loss, gen_test_loss = trainer.trainGAN(real_features)
            tf.summary.scalar("discriminator_test_loss", disc_test_loss, step=1)
            tf.summary.scalar("generator_test_loss", gen_test_loss, step=1)
    
    session_num = 0
    # create an instance of the model
    for model_version in HP_GAN_VERSION.domain.values:
        for batch_size in HP_BATCH_SIZE.domain.values:
            for learning_rate in HP_LEARNING_RATE.domain.values:
                hparams = {
                    HP_GAN_VERSION: model_version,
                    HP_BATCH_SIZE: batch_size,
                    HP_LEARNING_RATE: learning_rate,
                }

                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                run('logs/hparam_tuning/' + run_name, hparams)
                session_num += 1


if __name__ == '__main__':
    main()
