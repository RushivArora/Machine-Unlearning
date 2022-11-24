import os
from os import path
import pickle
import logging
import shutil

import config

from utils.load_data import LoadData

class DataStore:
    def __init__(self, args):
        self.logger = logging.getLogger("DataStore")
        self.args = args
        self.determine_data_path()

    def create_basic_folders(self):
        folder_list = [config.SPLIT_INDICES_PATH, config.SHADOW_MODEL_PATH, config.TARGET_MODEL_PATH,
                       config.ATTACK_DATA_PATH, config.ATTACK_MODEL_PATH]
        for folder in folder_list:
            self.create_folder(folder)

    def determine_data_path(self):
        self.save_name = "_".join((self.args.unlearning_method, self.args.dataset_name,
                                   self.args.original_label, self.args.original_model,
                                   str(self.args.shadow_set_num),
                                   str(self.args.target_set_num),
                                   str(self.args.shadow_set_size),
                                   str(self.args.target_set_size),
                                   str(self.args.shadow_unlearning_size),
                                   str(self.args.target_unlearning_size),
                                   str(self.args.shadow_unlearning_num),
                                   str(self.args.target_unlearning_num),
                                   str(self.args.target_num_shard),
                                   str(self.args.shadow_num_shard)
                                   ))
        if self.args.is_dp_defense:
            self.save_name += "_DP"

        self.target_model_name = config.TARGET_MODEL_PATH + self.save_name
        self.shadow_model_name = config.SHADOW_MODEL_PATH + self.save_name

        self.attack_train_data = config.SHADOW_MODEL_PATH + "posterior" + self.save_name
        self.attack_test_data = config.TARGET_MODEL_PATH + "posterior" + self.save_name

    def load_raw_data(self):
        load = LoadData()
        num_classes = {
            "adult": 2,
            "accident": 3,
            "location": 9,
            "cifar10": 10,
            "mnist": 10,
            "stl10": 10
        }
        self.num_classes = num_classes[self.args.dataset_name]
        if self.args.dataset_name == "cifar10":
            self.df = load.load_cifar10_data()
            self.num_records = self.df.data.shape[0]
        elif self.args.dataset_name == "stl10":
            self.df = load.load_stl10_data()
            self.num_records = self.df.data.shape[0]
        elif self.args.dataset_name == "mnist":
            self.df = load.load_mnist_data()
            self.num_records = self.df.data.shape[0]
        # Uncomment this to test categorical dataset on DNN model
        # elif self.args['dataset_name'] in ["adult", "accident", "location"]:
        #     self.df = load.loader_cat_data(self.args['dataset_name'], self.args['original_label'], batch_size=32)
        #     self.num_records = self.df.tensors[0].data.shape[0]
        elif self.args.dataset_name == "adult":
            self.df = load.load_adult(self.args.original_label)
            self.num_records = self.df.shape[0]
        elif self.args.dataset_name == "accident":
            self.df = load.load_accident(self.args.original_label)
            self.num_records = self.df.shape[0]
        elif self.args.dataset_name == "location":
            self.df = load.load_location(self.args.original_label)
            self.num_records = self.df.shape[0]
        else:
            raise Exception("invalid dataset name")

        return self.df, self.num_records, self.num_classes

    def save_raw_data(self):
        pass

    def save_record_split(self, record_split):
        pickle.dump(record_split, open(config.SPLIT_INDICES_PATH + self.save_name, 'wb'))

    def load_record_split(self):
        record_split = pickle.load(open(config.SPLIT_INDICES_PATH + self.save_name, 'rb'))
        return record_split

    def save_attack_train_data(self, attack_train_data):
        pickle.dump((attack_train_data), open(self.attack_train_data, 'wb'))

    def load_attack_train_data(self):
        attack_train_data = pickle.load(open(self.attack_train_data, 'rb'))
        return attack_train_data

    def save_attack_test_data(self, attack_test_data):
        pickle.dump((attack_test_data), open(self.attack_test_data, 'wb'))

    def load_attack_test_data(self):
        attack_test_data = pickle.load(open(self.attack_test_data, 'rb'))
        return attack_test_data

    def create_folder(self, folder):
        if not path.exists(folder):
            try:
                self.logger.info("checking directory %s", folder)
                os.mkdir(folder)
                self.logger.info("new directory %s created", folder)
            except OSError as error:
                self.logger.info("deleting old and creating new empty %s", folder)
                # os.rmdir(folder)
                shutil.rmtree(folder)
                os.mkdir(folder)
                self.logger.info("new empty directory %s created", folder)
        else:
            self.logger.info("folder %s exists, do not need to create again.", folder)
