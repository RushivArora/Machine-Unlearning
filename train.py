import argparse
import pandas as pd
import numpy as np

import logging

from lib_unlearning.record_split import RecordSplit

from models.decision_tree import DT
from models.logistic_regression import LR
from models.MLP import MLP
from models.random_forest import RF
from models.dnn import DNN

from utils.data_store import DataStore
from multiprocessing import Pool

import torch
from torch.utils.data import Subset, DataLoader

ORIGINAL_DATASET_PATH = "temp_data/dataset/"
PROCESSED_DATASET_PATH = "temp_data/processed_dataset/"
SHADOW_MODEL_PATH = "temp_data/shadow_models/"
TARGET_MODEL_PATH = "temp_data/target_models/"
SPLIT_INDICES_PATH = "temp_data/split_indices/"
ATTACK_MODEL_PATH = "temp_data/attack_models/"
ATTACK_DATA_PATH = "temp_data/attack_data/"


class Train:
    def __init__(self, args):
        self.dataset_name = args.dataset_name
        self.dataset_type = args.dataset_type
        self.args = args
        assert self.args.shadow_set_size >= self.args.shadow_unlearning_size
        assert self.args.target_set_size >= self.args.target_unlearning_size

        self.load_data()

    def load_data(self):
        print("print loading data")
        self.data_store = DataStore(self.args)
        self.save_name = self.data_store.save_name
        self.df, self.num_records, self.num_classes = self.data_store.load_raw_data()
        self.data_store.create_basic_folders()
        print("print data loaded")

    def get_model(self):
        if self.args.original_model == "LR":
            return LR()
        elif self.args.original_model == "DT":
            return DT()
        elif self.args.original_model == "RF":
            return RF()
        elif self.args.original_model == "MLP":
            return MLP()
        elif self.args.original_model == 'LRTorch':
            return DNN(net_name='logistic', num_classes=self.num_classes, args=self.args)
        elif self.args.original_model == 'scnn':
            return DNN(net_name='simple_cnn', num_classes=self.num_classes, args=self.args)
        elif self.args.original_model == 'resnet50':
            return DNN(net_name='resnet50', num_classes=self.num_classes, args=self.args)
        elif self.args.original_model == 'densenet':
            return DNN(net_name='densenet', num_classes=self.num_classes, args=self.args)
        elif self.args.original_model == 'MLPTorch':
            return DNN(net_name='mlp', num_classes=self.num_classes, args=self.args)


class Train_Scratch_Model(Train):
    def __init__(self, args):
        super(Train_Scratch_Model, self).__init__(args)
        self.args = args

        if self.args.is_sample:
            self.split_records()

        self.get_model()
        self.train_shadow_model()
        self.train_target_model()

    def train_all_models(self, num_sample, num_shard, save_path, model_type):
        if not self.args.is_sample:
            self.record_split = pickle.load(open(config.SPLIT_INDICES_PATH + self.save_name, 'rb'))

        #  data split
        self.record_split.generate_sample(model_type)

        if self.args.is_train_multiprocess:
            p = Pool(50, maxtasksperchild=1)

        """
        import psutil
        ps = psutil.Process()
        cores = ps.cpu_affinity()
        ps.cpu_affinity(cores[0:50])
        """

        for sample_index in range(num_sample):
            sample_set = self.record_split.sample_set[sample_index]
            sample_indices = sample_set["set_indices"]
            unlearning_set = sample_set["unlearning_set"]

            save_name_original = save_path + "original_S" + str(sample_index)
            self.__train_model_single(sample_indices, save_name_original, sample_index, j=0)

            for unlearning_set_index, unlearning_indices in unlearning_set.items():
                print("training %s model: sample set %s | unlearning set %s" % (model_type, sample_index, unlearning_set_index))

                # case = "deletion"
                unlearning_train_indices = np.setdiff1d(sample_indices, unlearning_indices)
                # case = "online_learning"
                if self.args.samples_to_evaluate == "online_learning":
                    replace_indices = np.random.choice(self.record_split.replace_indices, size=unlearning_indices.shape[0], replace=False)
                    unlearning_train_indices = np.append(unlearning_train_indices, replace_indices)

                save_name_unlearning = save_path + "_".join(
                    ("unlearning_S" + str(sample_index), str(unlearning_set_index)))

                self.__train_model_single(unlearning_train_indices, save_name_unlearning, sample_index, unlearning_set_index)


    def train_shadow_model(self):
        path = SHADOW_MODEL_PATH + self.save_name + "/"
        self.data_store.create_folder(path)
        self.train_all_models(self.args.shadow_set_num, self.args.shadow_num_shard, path, "shadow")
        print("Shadow model trained")

    def train_target_model(self):
        path = TARGET_MODEL_PATH + self.save_name + "/"
        self.data_store.create_folder(path)
        self.train_all_models(self.args.target_set_num, self.args.target_num_shard, path, "target")
        print("target model trained")

    def __train_model_single(self, sample_set_indices, save_name, i, j):
        original_model = self.get_model()

        if self.dataset_type == "categorical":
            train_x = self.df.iloc[sample_set_indices, :-1].values
            train_y = self.df.iloc[sample_set_indices, -1].values

            if self.args.unlearning_method == "sisa" and self.dataset_name == "location":
                train_x = np.concatenate((train_x, np.zeros((9, train_x.shape[1]))))
                train_y = np.concatenate((train_y, np.arange(9)))

            #print(train_x, train_y)
            original_model.train_model(train_x, train_y, save_name=save_name)

        elif self.dataset_name in ["mnist", "cifar10", 'stl10']:
            train_dataset = Subset(self.df, sample_set_indices)
            train_loader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size, shuffle=True)
            test_dataset = Subset(self.df, self.record_split.target_set[0]["set_indices"])
            test_loader = DataLoader(dataset=test_dataset, batch_size=self.args.batch_size, shuffle=True)
            original_model.train_model(train_loader, test_loader=test_loader, save_name=save_name)

        print("print model trained")

    def split_records(self):
        split_para = self.num_records
        self.record_split = RecordSplit(split_para, args=self.args)
        self.record_split.split_shadow_target()
        self.record_split.sample_records(self.args.unlearning_method)
        self.data_store.save_record_split(self.record_split)



class Train_Sisa_Model(Train):
    def __init__(self, args):
        super(Train_Sisa_Model, self).__init__(args)
        self.args = args

        print("Model train sisa")

        if self.args.is_sample:
            self.split_records()

        self.get_model()
        self.train_shadow_model()
        self.train_target_model()

    def train_all_models(self, num_sample, num_shard, save_path, model_type):
        if not self.args.is_sample:
            self.record_split = pickle.load(open(config.SPLIT_INDICES_PATH + self.save_name, 'rb'))

        self.record_split.generate_sample(model_type)

        if self.args.is_train_multiprocess:
            p = Pool(20, maxtasksperchild=1)
        #p = Pool(20, maxtasksperchild=1)
        """
        import psutil
        ps = psutil.Process()
        cores = ps.cpu_affinity()
        ps.cpu_affinity(cores[0:int(len(cores)/2)])
        """
        for i in range(num_sample):
            sample_set = self.record_split.sample_set[i]
            shard_set = sample_set["shard_set"]
            unlearning_indices = sample_set["unlearning_indices"]
            unlearning_shard_mapping = sample_set["unlearning_shard_mapping"]

            # train original model
            for j in range(num_shard):
                save_name = save_path + "original_S%s_M%s" % (i, j)
                self.__train_model_single(shard_set[j], save_name, i, j)

            # train unlearning models
            for j in unlearning_indices:
                print("training %s model set %s unlearning %s" % (model_type, i, j))

                shard_index = unlearning_shard_mapping[j]
                shard_indices = shard_set[shard_index]
                indices = np.delete(shard_indices, np.where(shard_indices == j)[0])
                save_name_unlearning = save_path + "unlearning_S%s_M%s" % (i, shard_index) + "_" + str(j)
                #p.apply_async(self.__train_model_single, args=(indices, save_name_unlearning, i, j))
                # sleep(0.1)
                self.__train_model_single(indices, save_name_unlearning, i, j)
        #p.close()
        #p.join()

    def train_shadow_model(self):
        path = SHADOW_MODEL_PATH + self.save_name + "/"
        self.data_store.create_folder(path)
        self.train_all_models(self.args.shadow_set_num, self.args.shadow_num_shard, path, "shadow")
        print("Shadow model trained")

    def train_target_model(self):
        path = TARGET_MODEL_PATH + self.save_name + "/"
        self.data_store.create_folder(path)
        self.train_all_models(self.args.target_set_num, self.args.target_num_shard, path, "target")
        print("target model trained")

    def __train_model_single(self, sample_set_indices, save_name, i, j):
        original_model = self.get_model()

        if self.dataset_type == "categorical":
            train_x = self.df.iloc[sample_set_indices, :-1].values
            train_y = self.df.iloc[sample_set_indices, -1].values

            if self.args.unlearning_method == "sisa" and self.dataset_name == "location":
                train_x = np.concatenate((train_x, np.zeros((9, train_x.shape[1]))))
                train_y = np.concatenate((train_y, np.arange(9)))

            original_model.train_model(train_x, train_y, save_name=save_name)

        elif self.dataset_name in ["mnist", "cifar10", 'stl10']:
            train_dataset = Subset(self.df, sample_set_indices)
            train_loader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size, shuffle=True)
            test_dataset = Subset(self.df, self.record_split.target_set[0]["set_indices"])
            test_loader = DataLoader(dataset=test_dataset, batch_size=self.args.batch_size, shuffle=True)
            original_model.train_model(train_loader, test_loader=test_loader, save_name=save_name)

        print("print model trained")

    def split_records(self):
        split_para = self.num_records
        self.record_split = RecordSplit(split_para, args=self.args)
        self.record_split.split_shadow_target()
        self.record_split.sample_records(self.args.unlearning_method)
        self.data_store.save_record_split(self.record_split)
