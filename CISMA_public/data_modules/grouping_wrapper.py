import math

import torch


class GroupingWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, csi_near, csi_far):
        super().__init__()
        # 图片数据集
        self.dataset = dataset
        # CSI数据
        self.csi_near = csi_near
        self.csi_far = csi_far
    def __getitem__(self, idx):
        """if self.with_replacement:"""
        
        img, cor_img, _, _, _ = self.dataset.__getitem__(idx)
        
        return (img, cor_img, self.csi_near[idx], self.csi_far[idx])

    def __len__(self):

        return len(self.dataset)
