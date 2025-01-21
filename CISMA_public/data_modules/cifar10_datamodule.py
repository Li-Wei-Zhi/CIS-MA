import os
import sys
path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   #os.path.dirname通俗的讲是返回上一级文件夹绝对路径的意思，多套一层就多返回一层
sys.path.append(path) 
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, transforms

# from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from data_modules.grouping_wrapper import GroupingWrapper
import os
import random
from data_modules.colored_mnist_dataloader import Unified_Dataloader
import matplotlib.pyplot as plt
class CIFAR10DataModule(Dataset):
    def __init__(self, config, train = True) -> None:
        super().__init__()
        self.config = config
        self.train = train
        data_folder = config['data'][config['data']['dataset']]['root']
        assert config['data']['dataset'] == 'cifar'
        if not os.path.isdir(data_folder):
            os.makedirs(data_folder)
        self.data = datasets.CIFAR10(
            root=data_folder, train=train, download=True, transform=ToTensor()
        )
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.counter = 0
        self.pair_seed = 1
        self.train_num = self.config['data'][self.config['data']['dataset']]['train_val_split'][0]
        if self.train:
            if self.config['data'][self.config['data']['dataset']]['pair_mode'] == "random":
                self.create_train_pairs()
                self.create_val_pairs()
            elif self.config['data'][self.config['data']['dataset']]['pair_mode'] == "correlated":
                self.create_cor_pairs()
        else:
            self.create_test_pairs()
    def create_train_pairs(self):
        # 设置seed
        random.seed(self.pair_seed)
        self.pair_train_data_index = [i for i in range(self.config['data'][self.config['data']['dataset']]['train_val_split'][0])]
        random.shuffle(self.pair_train_data_index)
    def create_val_pairs(self):
        # 设置seed
        random.seed(self.config['data']['val_seed'])
        self.pair_val_data_index = [i for i in range(self.config['data'][self.config['data']['dataset']]['train_val_split'][1])]
        random.shuffle(self.pair_val_data_index)
    def create_test_pairs(self):
        # 设置seed
        random.seed(self.config['data']['val_seed'])
        self.pair_test_data_index = [i for i in range(len(self.data))]
        random.shuffle(self.pair_test_data_index)
    def create_cor_pairs(self):
        random.seed(self.config['data']['color_seed'])
        self.pair_data = []
        targets_list_tensor = torch.from_numpy(np.array(self.data.targets)).unsqueeze(-1)
        all_targets = targets_list_tensor.repeat(1, len(self.data.targets))
        all_targets_r = all_targets - targets_list_tensor.squeeze()
        for idx in range(len(self.data)):
            corr_index_set = (all_targets_r[idx]==0).nonzero().numpy().reshape(-1).tolist()
            # target = self.data[idx][1]
            # corr_index_set = [i for i, x in enumerate(self.data) if x[1] == target]
            # corr_index_set = np.argwhere(np.array(self.data.targets) == target).reshape(-1).tolist()
            corr_index_set.remove(idx)
            pair_idx = random.choice(corr_index_set)
            self.pair_data.append(self.data[pair_idx])
    def __getitem__(self, index):
        if self.config['data'][self.config['data']['dataset']]['pair_mode'] == "correlated":
            image, digit_label_img = self.data[index]
            cor_image, digit_label_corimg = self.pair_data[index]
        
            label_img = torch.tensor(digit_label_img, dtype=torch.float32)
            label_cor_img = torch.tensor(digit_label_corimg, dtype=torch.float32)
        else:
            if self.train:
                if index < self.config['data'][self.config['data']['dataset']]['train_val_split'][0]:
                    image, digit_label_img = self.data[index]
                    cor_image, digit_label_corimg = self.data[self.pair_train_data_index[index]]
            
                    label_img = torch.tensor(digit_label_img, dtype=torch.float32)
                    label_cor_img = torch.tensor(digit_label_corimg, dtype=torch.float32)
                    if self.config['orthogonal']:
                        self.counter += 1
                        if self.counter == self.config['data'][self.config['data']['dataset']]['train_val_split'][0]:
                            self.counter = 0
                            self.pair_seed += 1
                            self.create_train_pairs()
                else:
                    image, digit_label_img = self.data[index]
                    cor_image, digit_label_corimg = self.data[self.pair_val_data_index[index-self.train_num]]
            
                    label_img = torch.tensor(digit_label_img, dtype=torch.float32)
                    label_cor_img = torch.tensor(digit_label_corimg, dtype=torch.float32)
            else:
                image, digit_label_img = self.data[index]
                cor_image, digit_label_corimg = self.data[self.pair_test_data_index[index]]
        
                label_img = torch.tensor(digit_label_img, dtype=torch.float32)
                label_cor_img = torch.tensor(digit_label_corimg, dtype=torch.float32)
        return image, cor_image, label_img, label_cor_img, index
    def __len__(self):
        return len(self.data)

