import os
import sys
path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   #os.path.dirname通俗的讲是返回上一级文件夹绝对路径的意思，多套一层就多返回一层
sys.path.append(path)    #将路径添加到python的搜索路径中
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import random
import numpy as np
import torch.nn.functional as F
# from src.utils.custom_typing import ColoredMNISTData
import os
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from data_modules.grouping_wrapper import GroupingWrapper
def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True

class ColoredMNISTDataset(Dataset):
    @staticmethod
    # 静态方法，随机生成颜色
    def get_random_colors(color_seed, data_len):
        # RGB颜色列表
        rgb_code_list = np.array([
            (255.0, 0.0, 0.0),
            (255.0, 128.0, 0.0),
            (255.0, 255.0, 0.0),
            (128.0, 255.0, 0.0),
            (0.0, 255.0, 0.0),
            (0.0, 255.0, 128.0),
            (0.0, 255.0, 255.0),
            (0.0, 128.0, 255.0),
            (0.0, 0.0, 255.0),
            (128.0, 0.0, 255.0),
            (255.0, 0.0, 255.0),
            (255.0, 0.0, 128.0),
        ])
        # 设置seed
        random.seed(color_seed)
        # 颜色列表的长度16
        lenght = len(rgb_code_list)
        # bg的颜色index
        bg_index = np.array([random.randint(0, lenght - 1) for _ in range(data_len)])
        # fg的颜色index
        fg_index = np.array([random.randint(0, lenght - 1) for _ in range(data_len)])
        # bg的颜色
        color_bg = rgb_code_list[bg_index]
        # fg的颜色
        color_fg = rgb_code_list[fg_index]

        return color_bg, color_fg, bg_index, fg_index

    @staticmethod
    # 产生颜色对
    def create_colored_pairs(image, rgb_color_bg, rgb_color_fg):
        """
        Get an MNIST image an generate two nex images by changing the background and foreground of the image

        :param image: Array whose values are in the range of [0.0, 1.0]
        """
        # 一张图像，像素值的区间为0到1之间，像素值小于0.5的部分可以被视为背景，大于0.5的部分可以被视为前景
        # .long()代表将其转为长整型
        index_background = (image < 0.5).long()
        index_foreground = (image >= 0.5).long()
        # 用背景mask乘以图像，得到图像中的背景部分
        keep_background = index_background * image
        # 用前景mask乘以图像，得到图像中的前景部分
        keep_foreground = index_foreground * image

        index_background = index_background - keep_background
        index_foreground = keep_foreground
        # 保留前景(数字)，改变背景的颜色， 拼接三个颜色通道
        colored_background = torch.stack(
            [
                rgb_color_bg[0] * index_background + keep_foreground * 255.0,
                rgb_color_bg[1] * index_background + keep_foreground * 255.0,
                rgb_color_bg[2] * index_background + keep_foreground * 255.0,
            ],
            axis=2,
        )
        # 保留背景颜色，改变数字的颜色，拼接三个通道
        colored_foreground = torch.stack(
            [
                rgb_color_fg[0] * index_foreground + keep_background * 255.0,
                rgb_color_fg[1] * index_foreground + keep_background * 255.0,
                rgb_color_fg[2] * index_foreground + keep_background * 255.0,
            ],
            axis=2,
        )
        return colored_background.permute(2, 0, 1), colored_foreground.permute(2, 0, 1)
    
   

    def __init__(self, config, rho_req, train = True, test_rho = False) -> None:
        super().__init__()
        self.config = config
        self.train = train
        self.test_rho = test_rho
        self.rho_req = rho_req
        data_folder = config['data'][config['data']['dataset']]['root']
        assert config['data']['dataset'] == 'mnist'
        if not os.path.isdir(data_folder):
            os.makedirs(data_folder)
        self.data = datasets.MNIST(
            root=data_folder, train=train, download=True, transform=ToTensor()
        )
        self.color_bg_list, self.color_fg_list, self.bg_index_list, self.fg_index_list = self.get_random_colors(self.config['data']['color_seed'], len(self.data))
        self.counter = 0
        self.pair_seed = 1
        if self.train:
            self.create_pairs()
    def create_pairs(self):
        random.seed(self.pair_seed)
        self.pair_data_index = [i for i in range(self.config['data'][self.config['data']['dataset']]['train_val_split'][0])]
        random.shuffle(self.pair_data_index)
    def give_cor_pairs_rho_req(self, index):
        image, digit_label = self.data[index]
        # image /= 255
        # rgb_color_bg, rgb_color_fg, bg_label, fg_label = self.get_random_colors(self.config['data']['color_seed'])
        rgb_color_bg, rgb_color_fg, bg_label, fg_label = self.color_bg_list[index], self.color_fg_list[index], self.bg_index_list[index], self.fg_index_list[index]
        bg_digit, fg_digit = self.create_colored_pairs(
            image=image.squeeze(0), rgb_color_bg=rgb_color_bg, rgb_color_fg=rgb_color_fg
        )
        fg_digit_trans = self.give_alter_image(fg_digit)
        fg_digit_trans /= 255
        bg_digit /= 255
        fg_label = torch.tensor(fg_label, dtype=torch.float32)
        bg_label = torch.tensor(bg_label, dtype=torch.float32)
        digit_label = torch.tensor(digit_label, dtype=torch.float32)
        return bg_digit, fg_digit_trans, fg_label, bg_label, digit_label
    def give_alter_image(self, fg_digit):
        transformed_image = fg_digit.clone()
        # 调整新图像以达到目标余弦相似度
        while True:
            # 计算新图像与原始图像的余弦相似度
            cosine_similarity = F.cosine_similarity(fg_digit.reshape(-1), transformed_image.reshape(-1), dim=0)

            # 如果余弦相似度达到目标值，则跳出循环
            if np.abs(cosine_similarity.item() - self.rho_req)<=0.05:
                break
            # if cosine_similarity<=target_cosine_similarity:
            #     break
            # 调整新图像的像素值
            # transformed_image = cv2.warpAffine(original_image, np.float32([[1, 0.1, 0], [0.2, 1, 0]]), (original_image.shape[1], original_image.shape[0]))
            transformed_image = transformed_image +  10*torch.rand_like(fg_digit)
            transformed_image = torch.clip(transformed_image, 0, 255)
        return transformed_image
    def give_uncor_pairs(self, index):
        image_l, digit_label_l = self.data[index]
        # np.random.seed(self.pair_seed)
        # random_index = np.random.randint(low=0, high=len(self.data))
        image_r, digit_label_r = self.data[self.pair_data_index[index]]
        # image /= 255
        # rgb_color_bg, rgb_color_fg, bg_label, fg_label = self.get_random_colors(self.config['data']['color_seed'])
        rgb_color_bg, rgb_color_fg, bg_label, fg_label = self.color_bg_list[index], self.color_fg_list[self.pair_data_index[index]], self.bg_index_list[index], self.fg_index_list[self.pair_data_index[index]]
        bg_digit, _ = self.create_colored_pairs(
            image=image_l.squeeze(0), rgb_color_bg=rgb_color_bg, rgb_color_fg=rgb_color_fg
        )
        _, fg_digit = self.create_colored_pairs(
            image=image_r.squeeze(0), rgb_color_bg=rgb_color_bg, rgb_color_fg=rgb_color_fg
        )
        fg_digit /= 255
        bg_digit /= 255
        fg_label = torch.tensor(fg_label, dtype=torch.float32)
        bg_label = torch.tensor(bg_label, dtype=torch.float32)
        digit_label_l = torch.tensor(digit_label_l, dtype=torch.float32)
        digit_label_r = torch.tensor(digit_label_r, dtype=torch.float32)
        self.counter += 1
        if self.counter == self.config['data'][self.config['data']['dataset']]['train_val_split'][0]:
            self.counter = 0
            self.pair_seed += 1
            self.create_pairs()
        # self.pair_seed = self.pair_seed % 12345
        return bg_digit, fg_digit, fg_label, bg_label, digit_label_l
    def give_cor_pairs(self, index):
        image, digit_label = self.data[index]
        # image /= 255
        # rgb_color_bg, rgb_color_fg, bg_label, fg_label = self.get_random_colors(self.config['data']['color_seed'])
        rgb_color_bg, rgb_color_fg, bg_label, fg_label = self.color_bg_list[index], self.color_fg_list[index], self.bg_index_list[index], self.fg_index_list[index]
        bg_digit, fg_digit = self.create_colored_pairs(
            image=image.squeeze(0), rgb_color_bg=rgb_color_bg, rgb_color_fg=rgb_color_fg
        )
        
        fg_digit /= 255
        bg_digit /= 255
        fg_label = torch.tensor(fg_label, dtype=torch.float32)
        bg_label = torch.tensor(bg_label, dtype=torch.float32)
        digit_label = torch.tensor(digit_label, dtype=torch.float32)
        return bg_digit, fg_digit, fg_label, bg_label, digit_label
    def set_type(self, type):
        self.type = type
    def __getitem__(self, index):
        if self.train:
            if not self.config['orthogonal']:
                bg_digit, fg_digit, fg_label, bg_label, digit_label = self.give_cor_pairs(index)
            else:
                if index >= self.config['data'][self.config['data']['dataset']]['train_val_split'][0]:
                    self.type = "val"
                    bg_digit, fg_digit, fg_label, bg_label, digit_label = self.give_cor_pairs(index)
                else:
                    self.type = "train"
                    bg_digit, fg_digit, fg_label, bg_label, digit_label = self.give_uncor_pairs(index)
        else:
            if not self.test_rho:
                bg_digit, fg_digit, fg_label, bg_label, digit_label = self.give_cor_pairs(index)
            else:
                bg_digit, fg_digit, fg_label, bg_label, digit_label = self.give_cor_pairs_rho_req(index)
        return bg_digit, fg_digit, fg_label, bg_label, digit_label

    def __len__(self):
        return len(self.data)
class Unified_Dataloader():
    def __init__(self, config):
        self.counter = 1
        self.config = config
        self.transforms = transforms.Compose([transforms.ToTensor()])
    def generate_noise_vec(self, 
                           num_samples, 
                           min_snr=None, 
                           max_snr=None, 
                           min_snr_interval = None,
                           max_snr_interval = None,
                           seed=1):
        csi_near = torch.empty(num_samples, 1).uniform_(
            min_snr,
            max_snr,
            generator=torch.Generator().manual_seed(seed),
        )   
        # csi_interval = torch.randint(low = min_snr_interval, high = max_snr_interval+1, size=csi_near.shape, generator=torch.Generator().manual_seed(seed),)
        # csi_far = csi_near - csi_interval
        csi_far = torch.empty(num_samples, 1).uniform_(
            min_snr,
            max_snr,
            generator=torch.Generator().manual_seed(seed*2),
        )   
        return csi_near, csi_far
    def train_val_loader(self, traindata, epoch):
        # self.data_train_single, self.data_val = random_split(
        #     dataset=traindata,
        #     lengths=self.config['data'][self.config['data']['dataset']]['train_val_split'],
        #     generator=torch.Generator().manual_seed(1),
        # )
        train_data_indices = torch.randperm(self.config['data'][self.config['data']['dataset']]['train_val_split'][0], generator=torch.Generator().manual_seed(1)).tolist()
        val_data_indices = torch.randperm(self.config['data'][self.config['data']['dataset']]['train_val_split'][1], generator=torch.Generator().manual_seed(1)).tolist()
        val_data_indices = [index+self.config['data'][self.config['data']['dataset']]['train_val_split'][0] for index in val_data_indices]
        self.data_train_single = torch.utils.data.Subset(traindata, train_data_indices)
        self.data_val = torch.utils.data.Subset(traindata, val_data_indices)
        num_train_samples = len(self.data_train_single)
        num_val_samples = len(self.data_val)
        csi_near_train, csi_far_train = self.generate_noise_vec(
            num_train_samples, 
            min_snr = self.config['data']['min_snr'],
            max_snr = self.config['data']['max_snr'],
            min_snr_interval = self.config['data']['train_min_snr_interval'],
            max_snr_interval = self.config['data']['train_max_snr_interval'],
            seed = self.counter)        
        csi_near_val, csi_far_val = self.generate_noise_vec(
            num_val_samples,
            min_snr = self.config['data']['min_snr'],
            max_snr = self.config['data']['max_snr'],
            min_snr_interval = self.config['data']['train_min_snr_interval'],
            max_snr_interval = self.config['data']['train_max_snr_interval'],
            seed = self.config['data']["val_seed"]
        )
        if self.config['data']['regroup_every_training_epoch']:
            self.counter += 1
        self.data_train = GroupingWrapper(
                self.data_train_single,
                csi_near_train,
                csi_far_train
            )
        self.data_val = GroupingWrapper(
                self.data_val,
                csi_near_val,
                csi_far_val
            )
        # setup_seed(42)
        train_loader = DataLoader(
            dataset=self.data_train,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            generator=torch.Generator().manual_seed(epoch),
        )
        val_loader = DataLoader(
            dataset=self.data_val,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
        )
        return train_loader, val_loader
    def train_val_loader_for_large(self, traindata, valdata, epoch):
        # self.data_train_single, self.data_val = random_split(
        #     dataset=traindata,
        #     lengths=self.config['data'][self.config['data']['dataset']]['train_val_split'],
        #     generator=torch.Generator().manual_seed(1),
        # )
        self.data_train_single = traindata
        self.data_val = valdata
        num_train_samples = len(self.data_train_single)
        num_val_samples = len(self.data_val)
        csi_near_train, csi_far_train = self.generate_noise_vec(
            num_train_samples, 
            min_snr = self.config['data']['min_snr'],
            max_snr = self.config['data']['max_snr'],
            min_snr_interval = self.config['data']['train_min_snr_interval'],
            max_snr_interval = self.config['data']['train_max_snr_interval'],
            seed = self.counter)        
        csi_near_val, csi_far_val = self.generate_noise_vec(
            num_val_samples,
            min_snr = self.config['data']['min_snr'],
            max_snr = self.config['data']['max_snr'],
            min_snr_interval = self.config['data']['train_min_snr_interval'],
            max_snr_interval = self.config['data']['train_max_snr_interval'],
            seed = self.config['data']["val_seed"]
        )
        if self.config['data']['regroup_every_training_epoch']:
            self.counter += 1
        self.data_train = GroupingWrapper(
                self.data_train_single,
                csi_near_train,
                csi_far_train
            )
        self.data_val = GroupingWrapper(
                self.data_val,
                csi_near_val,
                csi_far_val
            )
        
        train_loader = DataLoader(
            dataset=self.data_train,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            generator=torch.Generator().manual_seed(epoch),
        )
        val_loader = DataLoader(
            dataset=self.data_val,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
        )
        return train_loader, val_loader
    def test_loader(self, testdata):
        self.data_test = []
        for i, snr in enumerate(torch.arange(self.config['data']['min_snr'], self.config['data']['max_snr'] + 1.0, 1.0)):
            csi_near_test, csi_far_test = self.generate_noise_vec(
                len(testdata), 
                min_snr=snr, 
                max_snr=snr, 
                min_snr_interval=self.config['data']['test_snr_interval'],
                max_snr_interval=self.config['data']['test_snr_interval'],
                seed=i)
            self.data_test.append(GroupingWrapper(
                    testdata,
                    csi_near_test,
                    csi_far_test
                ))
        self.test_dataloader = [
            DataLoader(
                dataset=dt,
                batch_size=self.config['data']['batch_size'],
                shuffle=False,
            )
            for dt in self.data_test
        ]
        return self.test_dataloader
    def test_loader_for_pow_allo(self, testdata):
        self.data_test = []
        for i, snr_interval in enumerate(torch.arange(self.config['testing']['power_allo_test']['test_min_snr_interval'], 
                                                      self.config['testing']['power_allo_test']['test_max_snr_interval'] + 1.0, 1.0)):
            csi_near_test, csi_far_test = self.generate_noise_vec(
                len(testdata), 
                min_snr=self.config['testing']['power_allo_test']['csi_near'], 
                max_snr=self.config['testing']['power_allo_test']['csi_near'], 
                min_snr_interval=int(snr_interval.numpy()),
                max_snr_interval=int(snr_interval.numpy()),
                seed=i)
            self.data_test.append(GroupingWrapper(
                    testdata,
                    csi_near_test,
                    csi_far_test
                ))
        self.test_dataloader = [
            DataLoader(
                dataset=dt,
                batch_size=self.config['data']['batch_size'],
                shuffle=False,
            )
            for dt in self.data_test
        ]
        return self.test_dataloader
