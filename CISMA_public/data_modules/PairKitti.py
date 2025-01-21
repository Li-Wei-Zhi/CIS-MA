import random
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from data_modules.colored_mnist_dataloader import Unified_Dataloader
class PairKitti(Dataset):

    def __init__(self, path, set_type, config, resize=(128, 256)):
        super(Dataset, self).__init__()
        self.resize = resize
        self.config = config
        self.ar = []
        idx_path = '/home/lwz/lwzproj/MDMA-NOMA/data_modules/data_paths/KITTI_stereo_' + set_type + '.txt'
        with open(idx_path) as f:
            content = f.readlines()

        for i in range(0, len(content), 2):
            left_id = content[i].strip()
            right_id = content[i + 1].strip()
            self.ar.append((path + '/' + left_id, path + '/' + right_id))

        if set_type == 'train':
            self.transform = self.train_deterministic_cropping
        elif set_type == 'test' or set_type == 'val':
            self.transform = self.test_val_deterministic_cropping
        self.type = set_type
        if self.type == "train":
            self.counter = 0
            self.pair_seed = 1
            self.create_pairs()
    def create_pairs(self):
        random.seed(self.pair_seed)
        self.pair_data_index = [i for i in range(len(self.ar))]
        random.shuffle(self.pair_data_index)
    def train_deterministic_cropping(self, img, side_img):
        # Center Crop
        img = TF.center_crop(img, (370, 740))
        side_img = TF.center_crop(side_img, (370, 740))

        # Resize
        img = TF.resize(img, self.resize)
        side_img = TF.resize(side_img, self.resize)

        # Random Horizontal Flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            side_img = TF.hflip(side_img)

        # Convert to Tensor
        img = transforms.ToTensor()(img)
        side_img = transforms.ToTensor()(side_img)

        return img, side_img

    def test_val_deterministic_cropping(self, img, side_img):
        # Center Crop
        img = TF.center_crop(img, (370, 740))
        side_img = TF.center_crop(side_img, (370, 740))

        # Resize
        img = TF.resize(img, self.resize)
        side_img = TF.resize(side_img, self.resize)

        # Convert to Tensor
        img = transforms.ToTensor()(img)
        side_img = transforms.ToTensor()(side_img)

        return img, side_img
    def give_cor_pairs(self, index):
        left_path, right_path = self.ar[index]

        img = Image.open(left_path)
        side_img = Image.open(right_path)
        image_pair = self.transform(img, side_img)

        return image_pair[0], image_pair[1], left_path, right_path, index
    def give_uncor_pairs(self, index):
        left_path, _ = self.ar[index]
        _, right_path = self.ar[self.pair_data_index[index]]
        img = Image.open(left_path)
        side_img = Image.open(right_path)
        image_pair = self.transform(img, side_img)
        self.counter += 1
        if self.counter == len(self.ar):
            self.counter = 0
            self.pair_seed += 1
            self.create_pairs()
        return image_pair[0], image_pair[1], left_path, right_path, index
    def __getitem__(self, index):
        if self.type == "train":
            if self.config['orthogonal']:
                img, side_img, left_path, right_path, index = self.give_uncor_pairs(index)
            else:
                img, side_img, left_path, right_path, index = self.give_cor_pairs(index)
        else:
            img, side_img, left_path, right_path, index = self.give_cor_pairs(index)
    
        return img, side_img, left_path, right_path, index

    def __len__(self):
        return len(self.ar)

    def __str__(self):
        return 'KITTI_stereo'


