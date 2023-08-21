import tensorflow as tf
from models.example_model import Classifier, C_Discriminator,C_Generator, AC_Discriminator, AC_Generator
from trainers.example_trainer import Trainer
from data_loader.data_generator import DataGenerator
from tensorboard.plugins.hparams import api as hp
import numpy as np
import os
import argparse
import pickle
import random
import openpyxl

def clients_main(config):
    client_name = config.clients_name
    data = DataGenerator(config)
    if config.use_features_central:
        suffix = f"_with_{config.features_central_version}" #indicate clients_1 version features center
        with open(f'tmp/clients_1/{config.features_central_version}/features_central.pkl','rb') as fp: 
            pre_features_central = pickle.load(fp) #load features_central pre-saved
    else:#for clients_1
        suffix = ""
        pre_features_central = None

    if not os.path.exists(f"tmp/{client_name}"):
        version_num = 0
        #create new exccel to record metrics in cls best local acc
        workbook = openpyxl.Workbook() 
        worksheet = workbook.create_sheet("0", 0)
        for col_num,col_index in enumerate(["version_num",'w','Train_acc',"Test_local acc", "Test_global acc","Cos_loss"]):
            worksheet.cell(row=1, column=col_num+1, value = col_index) 
    else:
        file_list = next(os.walk(f"./tmp/{client_name}"))[1]   #get all dir in path
        file_list = [int(i.split("_")[0]) for i in file_list] 
        file_list.sort()
        version_num = file_list[-1]+1  #get latest version num + 1
        workbook = openpyxl.load_workbook(f'./tmp/{client_name}/metrics_record.xlsx')
        worksheet = workbook['0'] 

    w_list = [0.0,1.0,5.0, 10.0] 

    all_test_x,all_test_y = data.test_x, data.test_y
    client_data = data.clients[client_name]
    HP_BATCH_SIZE = hp.HParam("batch_size", hp.Discrete([64]))
    HP_DISTANCE_LOSS_WEIGHT = hp.HParam("distance_loss_weight", hp.Discrete(w_list))
    HP_LEARNING_RATE = hp.HParam("learning_rate", hp.Discrete([0.001])) #hp.RealInterval(0.001, 0.1))
    # HP_GAN_VERSION = hp.HParam("gan_version", hp.Discrete(['ACGAN']))
    HPARAMS = [
        HP_BATCH_SIZE,
        HP_LEARNING_RATE,
        HP_DISTANCE_LOSS_WEIGHT,
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

    # create an instance of the model
    # for model_version in HP_GAN_VERSION.domain.values:
    for batch_size in HP_BATCH_SIZE.domain.values:
        for learning_rate in HP_LEARNING_RATE.domain.values:
            for distance_loss_weight in HP_DISTANCE_LOSS_WEIGHT.domain.values: #batch_size={batch_size}_learning_rate={learning_rate}
                # reset random seed for each client
                tf.random.set_seed(args.random_seed)
                np.random.seed(args.random_seed)
                random.seed(args.random_seed)

                # print("tf.random",tf.random.normal([2,1]))
                # print("np.seed",np.random.rand(2,1))
                # print("random.seed",random.choice('123456789'))
                # run_name = "run-%d" % version_num
                hparams = {
                    # HP_GAN_VERSION: model_version,
                    HP_BATCH_SIZE: batch_size,
                    HP_LEARNING_RATE: learning_rate,
                    HP_DISTANCE_LOSS_WEIGHT: distance_loss_weight
                }
                print({h.name: hparams[h] for h in hparams})
                hparams = {h.name: hparams[h] for h in hparams}
                os.makedirs(f"tmp/{client_name}/{version_num}{suffix}")

                # record hparams and config value in this version
                record_hparams_file = open(f"./tmp/{client_name}/{version_num}{suffix}/hparams_record.txt", "wt")
                for key,value in hparams.items():
                    record_hparams_file.write(f"{key}: {value}")
                    record_hparams_file.write("\n")
                for key,value in vars(config).items():
                    record_hparams_file.write(f"{key}: {value}")
                    record_hparams_file.write("\n")
                record_hparams_file.close()

                print('--- Starting trial: %s' % version_num)
                with tf.summary.create_file_writer(os.path.join(config.logdir,client_name,str(version_num))).as_default():
                    hp.hparams(hparams)  # record the values used in this trial
        
                    cls = Classifier(config)
                    generator = AC_Generator(config)
                    discriminator = AC_Discriminator(config)
                    trainer = Trainer(client_name, version_num, client_data, all_test_x, all_test_y, pre_features_central, cls, discriminator, generator,config,hparams)
                    cur_features_central, real_features = trainer.train_cls(worksheet,suffix)
                    features_label = trainer.get_features_label()
                    ### GAN
                    # print("after train cls")
                    # disc_test_loss, gen_test_loss, fake_features = trainer.trainGAN()
                    # tf.summary.scalar("discriminator_test_loss", disc_test_loss, step=1)
                    # tf.summary.scalar("generator_test_loss", gen_test_loss, step=1)
                    if client_name == "clients_1":
                        with open(f"tmp/clients_1/{version_num}{suffix}/features_central.pkl","wb") as fp:
                            pickle.dump(cur_features_central, fp)
                    np.save(f"tmp/{client_name}/{version_num}{suffix}/real_features",real_features)
                    np.save(f"tmp/{client_name}/{version_num}{suffix}/features_label",features_label)
                
                version_num += 1
                if client_name == "clients_1": 
                    #client 1 do not need to loop distance_loss_weight hapram
                    break
                
    workbook.save(f'./tmp/{client_name}/metrics_record.xlsx')

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
    parser.add_argument("--data_random_seed", type=int, default=1693)

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