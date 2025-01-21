import os
import sys
path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   #os.path.dirname通俗的讲是返回上一级文件夹绝对路径的意思，多套一层就多返回一层
sys.path.append(path)
import torch
import torch.nn as nn
from compressai.entropy_models import GaussianConditional, EntropyBottleneck
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
)
from model.afmodule import AFModule, AFModule_Side_Info
import os
import sys
path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   #os.path.dirname通俗的讲是返回上一级文件夹绝对路径的意思，多套一层就多返回一层
sys.path.append(path)    #将路径添加到python的搜索路径中
import tqdm
from utils import BLN2BCHW, BCHW2BLN
from model.attention_block import CrossAttention
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        return x

class Semantic_Encoder_small_no_csi(nn.Module):

    def __init__(self, N, M, C=3, **kwargs):
        super().__init__()

        self.g_a = nn.ModuleList([
            ResidualBlockWithStride(
                in_ch=C,
                out_ch=N,
                stride=2),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            AttentionBlock(N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=N,
                stride=2),
            AttentionBlock(N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlock(
                in_ch=N,
                out_ch=M),
            AttentionBlock(M),
        ])
    

    def forward(self, x):

        if isinstance(x, tuple):
            x, snr = x
        else:
            snr = None

        # for layer in self.g_a:
        #     if isinstance(layer, AFModule):
        #         x = layer((x, snr))
        #     else:
        #         x = layer(x)

        # return x
        output_list = []
        for layer in self.g_a:            
            if isinstance(layer, ResidualBlockWithStride):
                output_list.append(x)
                x = layer(x)
            else:
                x = layer(x)
        output_list.append(x)
        return x, output_list

class Semantic_Decoder_small_no_csi(nn.Module):

    def __init__(self, N, M, C=3, **kwargs):
        super().__init__()

        self.g_s = nn.ModuleList([
            AttentionBlock(M),
            ResidualBlock(
                in_ch=M,
                out_ch=N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            AttentionBlock(N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=C,
                upsample=2),
        ])


    def forward(self, x):

        if isinstance(x, tuple):
            x, snr = x
        else:
            snr = None

        for layer in self.g_s:
            if isinstance(layer, AFModule):
                x = layer((x, snr))
            else:
                x = layer(x)

        return x
class Semantic_Encoder_large_no_csi(nn.Module):

    def __init__(self, N, M, C=3, **kwargs):
        super().__init__()

        self.g_a = nn.ModuleList([
            ResidualBlockWithStride(
                in_ch=C,
                out_ch=N,
                stride=2),
            # AFModule(N=N, num_dim=1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=N,
                stride=2),
            # AFModule(N=N, num_dim=1),
            AttentionBlock(N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=N,
                stride=2),
            # AFModule(N=N, num_dim=1),
            AttentionBlock(N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=M,
                stride=2),
            # AFModule(N=M, num_dim=1),
            AttentionBlock(M),
        ])
    

    def forward(self, x):

        if isinstance(x, tuple):
            x, snr = x
        else:
            snr = None

        # for layer in self.g_a:
        #     if isinstance(layer, AFModule):
        #         x = layer((x, snr))
        #     else:
        #         x = layer(x)

        # return x
        output_list = []
        count = 1
        for layer in self.g_a:            
            if isinstance(layer, ResidualBlockWithStride):
                if not count % 2 == 0:
                    output_list.append(x)
                count += 1  
            if isinstance(layer, AFModule):
                x = layer((x, snr))
            else:
                x = layer(x)
        output_list.append(x)
        return x, output_list

class Semantic_Decoder_large_no_csi(nn.Module):

    def __init__(self, N, M, C=3, **kwargs):
        super().__init__()

        self.g_s = nn.ModuleList([
            AttentionBlock(M),
            ResidualBlock(
                in_ch=M,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            # AFModule(N=N, num_dim=1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            # AFModule(N=N, num_dim=1),
            AttentionBlock(N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            # AFModule(N=N, num_dim=1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=C,
                upsample=2),
            # AFModule(N=C, num_dim=1),
        ])


    def forward(self, x):

        if isinstance(x, tuple):
            x, snr = x
        else:
            snr = None

        for layer in self.g_s:
            if isinstance(layer, AFModule):
                x = layer((x, snr))
            else:
                x = layer(x)

        return x


# class Common_info_Encoder_small(nn.Module):

#     def __init__(self, N, M, C=3, **kwargs):
#         super().__init__()

        
#         self.g_a = nn.ModuleList([
#             ResidualBlockWithStride(
#                 in_ch=C,
#                 out_ch=N,
#                 stride=2),
#             AFModule(N=N, num_dim=1),
#             ResidualBlock(
#                 in_ch=N,
#                 out_ch=N),
#             ResidualBlock(
#                 in_ch=N,
#                 out_ch=N),
#             AFModule(N=N, num_dim=1),
#             AttentionBlock(N),
#             ResidualBlock(
#                 in_ch=N,
#                 out_ch=N),
#             ResidualBlockWithStride(
#                 in_ch=N,
#                 out_ch=N,
#                 stride=2),
#             AFModule(N=N, num_dim=1),
#             AttentionBlock(N),
#             ResidualBlock(
#                 in_ch=N,
#                 out_ch=N),
#             ResidualBlock(
#                 in_ch=N,
#                 out_ch=M),
#             AFModule(N=M, num_dim=1),
#             AttentionBlock(M),
#         ])
    

#     def forward(self, x):
#         if isinstance(x, tuple):
#             x, snr = x
#         else:
#             snr = None
#         output_list = []
#         for layer in self.g_a:            
#             if isinstance(layer, ResidualBlockWithStride):
#                 output_list.append(x)
#             if isinstance(layer, AFModule):
#                 x = layer((x, snr))
#             else:
#                 x = layer(x)
#         output_list.append(x)
#         return output_list
# class Semantic_Decoder_with_side_info_small(nn.Module):
#     def __init__(self, N, M, M_cor, C=3, image_size = (28, 28), **kwargs):
#         super().__init__()

#         self.g_s = nn.ModuleList([
#             AttentionBlock(M+M_cor),
#             ResidualBlock(
#                 in_ch=M+M_cor,
#                 out_ch=N),
#             ResidualBlock(
#                 in_ch=N,
#                 out_ch=N),
#             AFModule(N, num_dim=1),
#             ResidualBlock(
#                 in_ch=N,
#                 out_ch=N),
#             ResidualBlockUpsample(
#                 in_ch=N,
#                 out_ch=N,
#                 upsample=2),
#             AFModule(2*N, num_dim=1),
#             AttentionBlock(2*N),
#             ResidualBlock(
#                 in_ch=2*N,
#                 out_ch=N),
#             ResidualBlock(
#                 in_ch=N,
#                 out_ch=N),
#             AFModule(N, num_dim=1),
#             ResidualBlock(
#                 in_ch=N,
#                 out_ch=N),
#             ResidualBlockUpsample(
#                 in_ch=N,
#                 out_ch=C,
#                 upsample=2),
#             AFModule(N=2*C, num_dim=1),           
#             ResidualBlock(
#                 in_ch=2*C,
#                 out_ch=C),
#         ])


#     def forward(self, x, side_info_list):

#         if isinstance(x, tuple):
#             x, snr = x
#         else:
#             snr = None
#         count = 2
#         x = torch.cat([x, side_info_list[-1]], dim = 1)
#         for layer in self.g_s:

#             if isinstance(layer, AFModule):
#                 x = layer((x, snr))
#             else:
#                 x = layer(x)
#             if isinstance(layer, ResidualBlockUpsample):
#                 x = torch.cat([x, side_info_list[len(side_info_list)-count]], dim = 1)
#                 count += 1
#         return x
    

    

