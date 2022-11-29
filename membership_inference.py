import argparse
import pandas as pd
import numpy as np
import pickle

import logging

from lib_unlearning.record_split import RecordSplit
from lib_unlearning.attack import Attack
from lib_unlearning.construct_feature import ConstructFeature

from models.decision_tree import DT
from models.logistic_regression import LR
from models.MLP import MLP
from models.random_forest import RF
from models.dnn import DNN

from utils.data_store import DataStore
from multiprocessing import Pool

import config

ORIGINAL_DATASET_PATH = "temp_data/dataset/"
PROCESSED_DATASET_PATH = "temp_data/processed_dataset/"
SHADOW_MODEL_PATH = "temp_data/shadow_models/"
TARGET_MODEL_PATH = "temp_data/target_models/"
SPLIT_INDICES_PATH = "temp_data/split_indices/"
ATTACK_MODEL_PATH = "temp_data/attack_models/"
ATTACK_DATA_PATH = "temp_data/attack_data/"

class Base:
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

class MemInfBase(Base):
    def __init__(self, args):
        super(MemInfBase, self).__init__(args)
        self.args = args

    def load_split_record(self):
        self.record_split = self.data_store.load_record_split()

    def obtain_shadow_posterior(self):
        self.logger.info("obtaining shadow posterior")
        print("obtaining shadow posterior")

        path = config.SHADOW_MODEL_PATH + self.save_name + "/"
        self.shadow_posterior_df = self._obtain_posterior(self.args.shadow_set_num, self.args.shadow_num_shard, "shadow", path)
        self.construct_feature(self.shadow_posterior_df)
        self._save_posterior(self.shadow_posterior_df, config.SHADOW_MODEL_PATH)
        self.logger.info("obtained shadow posterior")
        print("obtained shadow posterior")

    def obtain_target_posterior(self):
        self.logger.info("obtaining target posterior")
        print("obtaining target posterior")

        path = config.TARGET_MODEL_PATH + self.save_name + "/"
        self.target_posterior_df = self._obtain_posterior(self.args.target_set_num, self.args.target_num_shard, "target", path)
        self.construct_feature(self.target_posterior_df)
        self._save_posterior(self.target_posterior_df, config.TARGET_MODEL_PATH)
        self.logger.info("obtained target posterior")
        print("obtained target posterior")

    def construct_feature(self, posterior_df):
        self.logger.info("constructing feature")
        print("constructing feature")
        feature = ConstructFeature(posterior_df)

        if self.args.is_defense and self.args.top_k == 0:
            posterior_df = feature.launch_label_defense(posterior_df)
        elif self.args.is_defense and self.args.top_k != 0:
            posterior_df = feature.launch_topk_defense(posterior_df, top_k=self.args.top_k)

        for method in ["direct_diff", "sorted_diff", 'direct_concat', 'sorted_concat', 'l2_distance', 'basic_mia']:
            feature.obtain_feature(method, posterior_df)

    def launch_attack(self):
        self.logger.info("launching attack")
        print("launching attack")

        save_name = "_".join((self.save_name, self.args.attack_model))
        path = config.ATTACK_MODEL_PATH + save_name + "/"
        self.data_store.create_folder(path)

        if not self.args.is_obtain_posterior:
            self.shadow_posterior_df = self._load_posterior(config.SHADOW_MODEL_PATH)
            self.target_posterior_df = self._load_posterior(config.TARGET_MODEL_PATH)

        self.attack_posterior_train = pd.DataFrame(data=self.shadow_posterior_df["label"], columns=["label"])
        self.attack_posterior_test = pd.DataFrame(data=self.target_posterior_df["label"], columns=["label"])

        upload_data = {}
        for method in ["direct_diff", "sorted_diff", 'direct_concat', 'sorted_concat', 'l2_distance', 'basic_mia']:
            attack = Attack(self.args.attack_model, self.shadow_posterior_df, self.target_posterior_df)

            upload_data['train_acc'], upload_data['train_auc'] = attack.train_attack_model(method, path)
            upload_data['test_acc'], upload_data['test_auc'] = attack.test_attack_model(method)
            upload_data['attack_feature'] = method

            attack.obtain_attack_posterior(self.attack_posterior_train, self.attack_posterior_test, method)

        upload_data = {}
        for method in ["direct_diff", "sorted_diff", 'direct_concat', 'sorted_concat', 'l2_distance', 'basic_mia']:
            upload_data['conf_improve_rate_mean_train'], upload_data['conf_better_rate_train'] = \
                Attack.calculate_comparison_metrics(self.attack_posterior_train, method)
            upload_data['conf_improve_rate_mean_test'], upload_data['conf_better_rate_test'] = \
                Attack.calculate_comparison_metrics(self.attack_posterior_test, method)
            upload_data['attack_feature'] = method


    def calculate_overfitting(self):
        overfitting_min = 1.0
        overfitting_max = 0.0
        model_path = config.TARGET_MODEL_PATH + self.save_name + "/"
        test_indices = self.record_split.shadow_set[0]["set_indices"]

        for i in range(self.args.num_target_set):
            train_indices = self.record_split.target_set[i]["set_indices"]

            if self.args.unlearning_method == "scratch":
                train_accuracy = self._calculate_scratch_acc(model_path, i, train_indices)
                test_accuracy = self._calculate_scratch_acc(model_path, i, test_indices)
            elif self.args.unlearning_method == "sisa":
                train_accuracy = self._calculate_sisa_acc(model_path, i, train_indices)
                test_accuracy = self._calculate_sisa_acc(model_path, i, test_indices)
            else:
                raise Exception("invalid unlearning method")

            overfitting = train_accuracy - test_accuracy

            if overfitting <= overfitting_min:
                overfitting_min = overfitting
            if overfitting >= overfitting_max:
                overfitting_max = overfitting

            self.logger.info("%s model: train_accuracy: %s | test_accuracy: %s ｜ overfitting: %s | "
                             "overfitting_min: %s ""| overfitting_max: %s"
                             % (i, train_accuracy, test_accuracy, overfitting, overfitting_min, overfitting_max))
            print("%s model: train_accuracy: %s | test_accuracy: %s ｜ overfitting: %s | "
                "overfitting_min: %s ""| overfitting_max: %s"
                % (i, train_accuracy, test_accuracy, overfitting, overfitting_min, overfitting_max))

        return round(overfitting_min, 4), round(overfitting_max, 4)

    def _obtain_posterior(self, num_sample, num_shard, sample_name, save_path):
        pass

    def _save_posterior(self, posterior_df, save_path):
        pickle.dump(posterior_df, open(save_path + "_".join(("posterior", self.save_name)), 'wb'))

    def _load_posterior(self, save_path):
        return pickle.load(open(save_path + "_".join(("posterior", self.save_name)), 'rb'))

    def _generate_test_case(self, index):
        # Uncomment this to test categorical datasets on DNN models
        # if self.dataset_name in ["adult", "accident", "location"]:
        #     labels = self.df.tensors[1]
        #     num_one = np.count_nonzero(labels)
        #     one_ratio = num_one / len(labels)
        #     case = self.df[index]
        #     label = case[1]
        #     feat = case[0].view(1, case[0].shape[1])
        #     return feat
        if self.dataset_name == "adult":
            return self.df.values[index, :14].reshape([1, 14])
        elif self.dataset_name == "accident":
            return self.df.values[index, :29].reshape([1, 29])
        elif self.dataset_name == "location":
            return self.df.values[index, :168].reshape([index.size, 168])
        elif self.dataset_name == "spotify":
            return self.df.values[index, :17].reshape([1, 17])
        elif self.dataset_name in ["mnist", 'stl10', 'cifar10']:
            #print("Index is ", index)
            #print("DF", self.df)
            case = self.df[index[0]]
            #for val in index:
            #    return self.df[val]
            #print(case)
            return case[0].unsqueeze(0)
            #return case.unsqueeze(0)
        else:
            raise Exception("invalid test dataset")

    def _calculate_scratch_acc(self, model_path, sample_index, indices):
        model = self.get_model()
        model.load_model(model_path + "original_S" + str(sample_index))

        dataset = torch.utils.data.Subset(self.df, indices)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)
        accuracy = model.test_model_acc(data_loader)

        return accuracy

    def _calculate_sisa_acc(self, model_path, sample_index, indices):
        post_dict = {}

        # calculate posterior
        for shard_index in range(self.args.shadow_num_shard):
            save_name = model_path + "original_S%s_M%s" % (sample_index, shard_index)
            model = self.get_model()
            model.load_model(save_name)

            post_dict[shard_index] = model.predict_proba(self.df[indices])

        # extract true label
        if self.dataset_name in ["adult", "accident", "location", "spotify"]:
            true_labels = self.df.values[indices, -1]
        elif self.dataset_name in ["mnist", "stl10", "cifar10"]:
            true_labels = self.df.class_to_idx[indices]

        # calculate predict label
        posterior = post_dict[0]
        for shard_index in range(1, self.args.shadow_num_shard):
            posterior += post_dict[shard_index]

        posterior /= self.args.shadow_num_shard
        pred_labels = np.argmax(posterior, axis=1)

        accuracy = np.count_nonzero(true_labels == pred_labels) / true_labels.size

        return accuracy



class MemInfScratch(MemInfBase):
    def __init__(self, args):
        super(MemInfScratch, self).__init__(args)

        self.logger = logging.getLogger("exp_mem_inf_scratch")
        self.args = args
        self.load_split_record()
        self.get_model()

        if self.args.is_obtain_posterior:
            self.obtain_shadow_posterior()
            self.obtain_target_posterior()

        self.launch_attack()

    def _obtain_posterior(self, num_sample, num_shard, model_type, save_path):
        self.record_split.generate_sample(model_type)
        posterior_df = pd.DataFrame(columns=["original", "unlearning", "label"])

        for sample_index in range(num_sample):
            sample_set = self.record_split.sample_set[sample_index]
            unlearning_set = sample_set["unlearning_set"]

            save_name_original = save_path + "original_S" + str(sample_index)
            model_original = self.get_model()
            model_original.load_model(save_name_original)

            pos_case = self.args.samples_to_evaluate

            for unlearning_set_index, unlearning_indices in unlearning_set.items():
                self.logger.debug("obtain posterior for %s model: sample set %s | unlearning set %s | unlearning"
                                  % (model_type, sample_index, unlearning_set_index))
                print("obtain posterior for %s model: sample set %s | unlearning set %s | unlearning"
                                  % (model_type, sample_index, unlearning_set_index))
                save_name_unlearning = save_path + "unlearning_S" + str(sample_index) + "_" + str(unlearning_set_index)
                model_unlearning = self.get_model()
                model_unlearning.load_model(save_name_unlearning)

                if pos_case == "in_out":
                    pass
                elif pos_case == "in_in":
                    temp = np.setdiff1d(sample_set["set_indices"], unlearning_indices)
                    unlearning_indices = np.random.choice(temp, size=1, replace=False)
                elif pos_case == "in_out_multi_version":
                    unlearning_indices = np.random.choice(unlearning_indices, size=1, replace=False)
                else:
                    raise Exception("Unsupported positive cases.")

                test_pos_case = self._generate_test_case(unlearning_indices)
                post_before_pos = model_original.predict_proba(test_pos_case)
                post_after_pos = model_unlearning.predict_proba(test_pos_case)

                df = pd.DataFrame(columns=["original", "unlearning", "label"])
                for index in range(post_before_pos.shape[0]):
                    df.loc[len(df)] = [post_before_pos[index].reshape([1, -1]), post_after_pos[index].reshape([1, -1]), 1]

                neg_index = np.random.choice(self.record_split.negative_indices, size=unlearning_indices.size)

                test_neg_case = self._generate_test_case(neg_index)
                post_before_neg = model_original.predict_proba(test_neg_case)
                post_after_neg = model_unlearning.predict_proba(test_neg_case)
                for index in range(post_before_neg.shape[0]):
                    df.loc[len(df)] = [post_before_neg[index].reshape([1, -1]), post_after_neg[index].reshape([1, -1]), 0]

                posterior_df = posterior_df.append(df, ignore_index=True)

        return posterior_df


class MemInfSISA(MemInfBase):
    def __init__(self, args):
        super(MemInfSISA, self).__init__(args)

        self.logger = logging.getLogger("exp_mem_inf_sisa")
        self.get_model()

        if self.args.is_obtain_posterior:
            self.obtain_shadow_posterior()
            self.obtain_target_posterior()

        self.launch_attack()

    def _obtain_posterior(self, num_sample, num_shard, model_name, save_path):
        posterior_df = pd.DataFrame(columns=["original", "unlearning", "label"])
        self.load_split_record()
        self.record_split.generate_sample(model_name)

        for i in range(num_sample):
            sample_set = self.record_split.sample_set[i]
            unlearning_indices = sample_set["unlearning_indices"]
            unlearning_shard_mapping = sample_set["unlearning_shard_mapping"]
            neg_indices = np.random.choice(self.record_split.negative_indices, unlearning_indices.size, replace=False)

            pos_posterior_original_dict = {}
            neg_posterior_original_dict = {}

            for shard_index in range(num_shard):
                save_name = save_path + "original_S%s_M%s" % (i, shard_index)
                model = self.get_model()
                model.load_model(save_name)
                pos_posterior_original_dict[shard_index] = model.predict_proba(
                    self._generate_test_cases(unlearning_indices))
                neg_posterior_original_dict[shard_index] = model.predict_proba(self._generate_test_cases(neg_indices))

            pos_posterior_original = pos_posterior_original_dict[0]
            neg_posterior_original = neg_posterior_original_dict[0]

            for shard_index in range(1, num_shard):
                print(pos_posterior_original_dict[shard_index].shape)
                pos_posterior_original += pos_posterior_original_dict[shard_index]
                neg_posterior_original += neg_posterior_original_dict[shard_index]

            #  Shard posterior is the average posterior of total.
            pos_posterior_original /= num_shard
            neg_posterior_original /= num_shard

            for j, pos_index in enumerate(unlearning_indices):
                self.logger.debug("obtain posterior for %s model set %s unlearning %s" % (model_name, i, pos_index))
                print("obtain posterior for %s model set %s unlearning %s" % (model_name, i, pos_index))

                shard_index = unlearning_shard_mapping[pos_index]
                save_name_unlearning = save_path + "unlearning_S%s_M%s" % (i, shard_index) + "_" + str(pos_index)
                model = self.get_model()
                model.load_model(save_name_unlearning)

                test_pos_case = self._generate_test_case(pos_index)
                pos_posterior_unlearning = model.predict_proba(test_pos_case)
                test_neg_case = self._generate_test_case(neg_indices[j])
                neg_posterior_unlearning = model.predict_proba(test_neg_case)

                for index in range(num_shard):
                    if index != shard_index:
                        pos_posterior_unlearning += pos_posterior_original_dict[index][j]
                        neg_posterior_unlearning += neg_posterior_original_dict[index][j]

                pos_posterior_unlearning /= num_shard
                neg_posterior_unlearning /= num_shard

                posterior_df.loc[len(posterior_df)] = [pos_posterior_original[j].reshape((1, -1)),
                                                       pos_posterior_unlearning, 1]
                posterior_df.loc[len(posterior_df)] = [neg_posterior_original[j].reshape((1, -1)),
                                                       neg_posterior_unlearning, 0]

        return posterior_df

    def _generate_test_cases(self, indices):
        if self.dataset_name == "adult":
            return self.df.values[indices, :14]
        elif self.dataset_name == "accident":
            return self.df.values[indices, :29]
        elif self.dataset_name == "location":
            return self.df.values[indices, :168]
        elif self.dataset_name == "spotify":
            return self.df.values[indices, :17]
        elif self.dataset_name in ["mnist", "cifar10"]:
            return self.df[indices]
        else:
            raise Exception("Unsupported dataset!")
