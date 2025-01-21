import torch.nn as nn
import torch
import numpy as np
class MINE(nn.Module):
    def __init__(self, input_channels):
        super(MINE, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
            in_channels=input_channels, out_channels=512, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1),
            nn.ReLU()
        )
        # 特征图大小为14乘14
        self.layers2 = nn.Sequential(nn.Linear(14*14, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 1))

    def forward(self, x, y):
        batch_size = x.size(0)
        tiled_x = torch.cat([x, x, ], dim=0)
        idx = torch.randperm(batch_size)

        shuffled_y = y[idx]
        concat_y = torch.cat([y, shuffled_y], dim=0)
        inputs = torch.cat([tiled_x, concat_y], dim=1)
        logits = self.layers(inputs)
        logits_1d = logits.reshape(2*batch_size, -1)

        pred_xy = logits_1d[:batch_size]
        pred_x_y = logits_1d[batch_size:]
        loss = - np.log2(np.exp(1)) * (torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y))))
        # compute loss, you'd better scale exp to bit
        return loss

class LocalStatisticsNetwork(nn.Module):
    def __init__(self, img_feature_channels: int):
        """Local statistique nerwork

        Args:
            img_feature_channels (int): [Number of input channels]
        """

        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=img_feature_channels, out_channels=1024, kernel_size=1, stride=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=1024, out_channels=1024, kernel_size=1, stride=1
        )
        self.conv3 = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, concat_feature: torch.Tensor) -> torch.Tensor:
        x = self.conv1(concat_feature)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        local_statistics = self.conv3(x)
        return local_statistics
    
class GlobalStatisticsNetwork(nn.Module):
    """Global statistics network

    Args:
        feature_map_size (int): Size of input feature maps 特征图大小
        feature_map_channels (int): Number of channels in the input feature maps 特征图通道数
        latent_dim (int): Dimension of the representationss 表征的维度
    """
    
    def __init__(
        self, feature_map_size: tuple, feature_map_channels: int, latent_dim: int
    ):

        super().__init__()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(
            in_features=(feature_map_size[0] * feature_map_size[1] * feature_map_channels) + latent_dim,
            out_features=512,
        )
        self.dense2 = nn.Linear(in_features=512, out_features=512)
        self.dense3 = nn.Linear(in_features=512, out_features=1)
        self.relu = nn.ReLU()

    def forward(
        self, feature_map: torch.Tensor, representation: torch.Tensor
    ) -> torch.Tensor:
        feature_map = self.flatten(feature_map)
        representation = self.flatten(representation)
        x = torch.cat([feature_map, representation], dim=1)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        global_statistics = self.dense3(x)

        return global_statistics

