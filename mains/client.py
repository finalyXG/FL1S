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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # General command line arguments for all models
    parser.add_argument(
        "--clients_name",
        type=str,
        help="Name of client",
        default="clients_1",
    )
    parser.add_argument("--use_features_central", type=bool, default=False)
    parser.add_argument("--features_central_version", type=str, default="0")
    parser.add_argument("--cls_num_epochs", type=int, default=20)
    parser.add_argument("--features_ouput_layer", help="The index of features output Dense layer",type=int, default=2)
    parser.add_argument("--GAN_num_epochs", type=int, default=1)
    parser.add_argument("--num_iter_per_epoch", type=int, default=10)
    parser.add_argument("--test_feature_num", type=int, default=500)
    parser.add_argument("--test_sample_num", help="The number of real features and fake features in tsne img", type=int, default=500) 
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--gp_weight", type=int, default=10.0)
    parser.add_argument("--discriminator_extra_steps", type=int, default=3)
    parser.add_argument("--num_examples_to_generate", type=int, default=16)
    parser.add_argument("--buffer_size", type=int, default=5000)
    parser.add_argument("--latent_dim", type=int, default=16)
    # parser.add_argument("--feature_dim", type=int, default=16)
    parser.add_argument("--max_to_keep", type=int, default=5)
    parser.add_argument("--num_classes", type=int, default=10)

    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--client_train_num", type=int, default=5000)
    parser.add_argument("--client_test_num", type=int, default=5000)
    parser.add_argument("--random_seed", type=int, default=10)

    parser.add_argument("--exp_name", type=str, default="example")
    parser.add_argument("--logdir", type=str, default="logs/hparam_tuning")

    args = parser.parse_args()
    args.use_features_central = bool(args.use_features_central)

    print("client:", args.clients_name)
    print("Whether use features central:", args.use_features_central)
    print("features_central_version:", args.features_central_version)
    print("client_train_num:", args.client_train_num)
    print("client_test_num:", args.client_test_num)
    clients_main(args)