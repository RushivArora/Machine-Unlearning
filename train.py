import argparse
import pandas as pd
import numpy as np

from models.decision_tree import DT
from models.logistic_regression import LR
from models.MLP import MLP
from models.random_forest import RF
from utils.data_store import DataStore

class Train:
    def __init__(args):
        self.dataset_name = args.dataset_name
        self.dataset_type = args.dataset_type
        assert self.args['shadow_set_size'] >= self.args['shadow_unlearning_size']
        assert self.args['target_set_size'] >= self.args['target_unlearning_size']

        if args.original_model == "LR":
            self.original_model = LR()
        elif args.original_model == "DT":
            self.original_model = DT()
        elif args.original_model == "RF":
            self.original_model = RF()
        elif args.original_model == "MLP":
            self.original_model = MLP()

    self.load_data()

    def load_data(self):
        print("print loading data")
        self.data_store = DataStore(self.args)
        self.save_name = self.data_store.save_name
        self.df, self.num_records, self.num_classes = self.data_store.load_raw_data()
        self.data_store.create_basic_folders()
        print("print data loaded")


class Train_Scratch_Model():
    def __init__(self, args):
        super(Train_Scratch_Model).__init__(args)



    def train_shadow_model(self):

    def train_target_model(self):
