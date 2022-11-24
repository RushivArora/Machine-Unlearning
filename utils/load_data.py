import pickle
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10, MNIST, STL10
import torchvision.transforms as transforms

import config


class LoadData:
    def __init__(self):
        self.logger = logging.getLogger("load_data")

    @staticmethod
    def load_location(original_label):
        if original_label == "NY":
            df = pickle.load(open(config.PROCESSED_DATASET_PATH + "Insta_ny", 'rb'))
        elif original_label == "LA":
            df = pickle.load(open(config.PROCESSED_DATASET_PATH + "Insta_la", 'rb'))
        else:
            raise Exception("invalid location city name")
        return df

    @staticmethod
    def load_adult(original_label):
        df = pickle.load(open(config.PROCESSED_DATASET_PATH + "adult", 'rb'))
        if original_label == 'income':
            df = df[['age', 'workclass', 'fnlwgt', 'education', 'educational-num',
                     'occupation', 'relationship', 'marital-status', 'race', 'gender', 'capital-gain',
                     'capital-loss', 'hours-per-week', 'native-country', 'income']]
        return df

    @staticmethod
    def load_accident(original_label):
        df = pickle.load(open(config.PROCESSED_DATASET_PATH + "accident", 'rb'))
        # 3-class balanced
        if original_label == 'severity':
            df = df[['Source', 'TMC', 'Start_Lat', 'Start_Lng', 'Distance(mi)',
                     'Side', 'County', 'State', 'Timezone', 'Airport_Code', 'Temperature(F)',
                     'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
                     'Wind_Direction', 'Wind_Speed(mph)', 'Precipitation(in)',
                     'Weather_Condition', 'Amenity', 'Crossing', 'Junction', 'Railway',
                     'Station', 'Traffic_Signal', 'Sunrise_Sunset', 'Civil_Twilight',
                     'Nautical_Twilight', 'Astronomical_Twilight', 'Severity']]
            df['Severity'] = df['Severity'].replace(2, 1)
            df['Severity'] = df['Severity'].replace(4, 2)
            df['Severity'] = df['Severity'].replace(3, 2)
        return df

    def load_mnist_data(self):
        trainloader, testloader, trainset, testset = LoadData.load_mnist()
        return trainset

    def load_cifar10_data(self):
        train_loader, test_loader, train_set, test_set = LoadData.load_cifar10()
        return train_set

    def load_stl10_data(self):
        train_loader, test_loader, train_set, test_set = LoadData.load_stl10()
        return train_set

    @staticmethod
    def loader_cat_data(dataset, original_label, batch_size):
        if dataset == 'adult':
            df = LoadData.load_adult("income")
        elif dataset == 'accident':
            df = LoadData.load_accident(original_label='severity')
            train_size = df.shape[0]
            data = df.iloc[:, :-1].to_numpy()
            labels = df.iloc[:, -1].to_numpy()
            zero_indices = np.where(labels == 2)
            labels[zero_indices] = 0
            train_x = torch.tensor(torch.from_numpy(np.array(data[:train_size, :], dtype=np.float32)))
            train_y = torch.tensor(np.int64(labels[:train_size]))
            train_dset = TensorDataset(train_x, train_y)
            return train_dset
        elif dataset == 'location':
            df = LoadData.load_location(original_label)
        else:
            raise Exception("invalid dataset name")

        train_size = df.shape[0]
        data = df.iloc[:, :-1].to_numpy()
        labels = df.iloc[:, -1].to_numpy()
        train_x = torch.tensor(data[:train_size, :]).float()
        train_y = torch.tensor(np.int64(labels[:train_size]))
        train_dset = TensorDataset(train_x, train_y)

        return train_dset

    @staticmethod
    def load_mnist(batch_size=32):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = MNIST(root=config.ORIGINAL_DATASET_PATH + 'mnist', train=True, transform=transform,
                          download=True)
        test_set = MNIST(root=config.ORIGINAL_DATASET_PATH + 'mnist', train=False, transform=transform)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader, train_set, test_set

    @staticmethod
    def load_cifar10(batch_size=32, num_workers=1):
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = CIFAR10(root=config.ORIGINAL_DATASET_PATH + 'cifar10', train=True, download=True, transform=transform_train)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_set = CIFAR10(root=config.ORIGINAL_DATASET_PATH + 'cifar10', train=False, download=True, transform=transform_test)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, test_loader, train_set, test_set

    @staticmethod
    def load_stl10(batch_size=32, num_workers=1):
        train_set = STL10(root=config.ORIGINAL_DATASET_PATH + 'stl10', split='train', download=True,
                          transform=transforms.Compose([
                              transforms.Pad(4),
                              transforms.Resize(32),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]))
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        test_set = STL10(root=config.ORIGINAL_DATASET_PATH + 'stl10', split='test', download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Resize(32),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         ]))
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, test_loader, train_set, test_set

    @staticmethod
    def load_image(dataset_name):
        load_data = LoadData()
        if dataset_name == 'mnist':
            return load_data.load_mnist()
        elif dataset_name == 'stl10':
            return load_data.load_stl10()
        elif dataset_name == 'cifar10':
            return load_data.load_cifar10()
