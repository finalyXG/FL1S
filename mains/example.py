import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.example_model import C_Discriminator,C_Generator,AC_Discriminator,AC_Generator
from trainers.example_trainer import Trainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args


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

    # create your data generator
    data = DataGenerator(config)

    # create an instance of the model you want
    generator = C_Generator(config)
    discriminator = C_Discriminator(config)

    # create trainer and pass all the previous components to it
    trainer = Trainer( data, discriminator, generator, config)
 
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
