import os
import sys
path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   #os.path.dirname通俗的讲是返回上一级文件夹绝对路径的意思，多套一层就多返回一层
sys.path.append(path)
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from model.attention_block import CrossAttention, EfficientAttention
from model.matchtransformer import MatchAttentionBlock
from model.statistic_network import LocalStatisticsNetwork, GlobalStatisticsNetwork
class Semantic_Encoder_large(nn.Module):

    def __init__(self, N, M, C=3, **kwargs):
        super().__init__()

        self.g_a = nn.ModuleList([
            ResidualBlockWithStride(
                in_ch=C,
                out_ch=N,
                stride=2),
            AFModule(N=N, num_dim=1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=N,
                stride=2),
            AFModule(N=N, num_dim=1),
            AttentionBlock(N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=N,
                stride=2),
            AFModule(N=N, num_dim=1),
            AttentionBlock(N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=M,
                stride=2),
            AFModule(N=M, num_dim=1),
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

class Semantic_Decoder_large(nn.Module):

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
            AFModule(N=N, num_dim=1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            AFModule(N=N, num_dim=1),
            AttentionBlock(N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            AFModule(N=N, num_dim=1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=C,
                upsample=2),
            AFModule(N=C, num_dim=1),
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
class Common_info_Encoder_large(nn.Module):

    def __init__(self, N, M, C=3, **kwargs):
        super().__init__()

        
        self.g_a = nn.ModuleList([
            ResidualBlockWithStride(
                in_ch=C,
                out_ch=N,
                stride=2),
            AFModule(N=N, num_dim=1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=N,
                stride=2),
            AFModule(N=N, num_dim=1),
            AttentionBlock(N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=N,
                stride=2),
            AFModule(N=N, num_dim=1),
            AttentionBlock(N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=M,
                stride=2),
            AFModule(N=M, num_dim=1),
            AttentionBlock(M),
        ])
    

    def forward(self, x):
        if isinstance(x, tuple):
            x, snr = x
        else:
            snr = None
        output_list = []
        count = 1
        for layer in self.g_a:            
            if isinstance(layer, ResidualBlockWithStride):
                if not count % 2 == 0:
                    output_list.append(x)
                # output_list.append(x)
                count += 1  
            if isinstance(layer, AFModule):
                x = layer((x, snr))
            else:
                x = layer(x)
        output_list.append(x)
        return output_list
class Common_info_Encoder_large_mi_esti(nn.Module):

    def __init__(self, N, M, C=3, **kwargs):
        super().__init__()
        self.block = nn.ModuleList([
            ResidualBlockWithStride(
                in_ch=C,
                out_ch=N,
                stride=2),
            AFModule(N=N, num_dim=1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlock(
                in_ch=N,
                out_ch=N)
        ])
        
        self.g_a = nn.ModuleList([
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=N,
                stride=2),
            AFModule(N=N, num_dim=1),
            AttentionBlock(N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=N,
                stride=2),
            AFModule(N=N, num_dim=1),
            AttentionBlock(N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=M,
                stride=2),
            AFModule(N=M, num_dim=1),
            AttentionBlock(M),
        ])
    

    def forward(self, x):
        if isinstance(x, tuple):
            x, snr = x
        else:
            snr = None
        output_list = []
        count = 1
        for layer in self.block:            
            if isinstance(layer, ResidualBlockWithStride):
                if not count % 2 == 0:
                    output_list.append(x)
                count += 1  
            if isinstance(layer, AFModule):
                x = layer((x, snr))
            else:
                x = layer(x)
        feature = x
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
        return output_list, feature
# class Semantic_Decoder_with_side_info_large(nn.Module):
#     def __init__(self, N, M, C=3, image_size = (128, 256), **kwargs):
#         super().__init__()

#         self.g_s = nn.ModuleList([
#             AttentionBlock(2*M),
#             ResidualBlock(
#                 in_ch=2*M,
#                 out_ch=N),
#             ResidualBlockUpsample(
#                 in_ch=N,
#                 out_ch=N,
#                 upsample=2),
#             AFModule(2*N, num_dim=1),
#             ResidualBlock(
#                 in_ch=2*N,
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
#             ResidualBlockUpsample(
#                 in_ch=N,
#                 out_ch=N,
#                 upsample=2),
#             AFModule(2*N, num_dim=1),
#             ResidualBlock(
#                 in_ch=2*N,
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
#                 # if (count-1) % 2 == 0:
#                 x = torch.cat([x, side_info_list[len(side_info_list)-count]], dim = 1)
#                 count += 1
#         return x
class Semantic_Decoder_with_side_info_large(nn.Module):
    def __init__(self, N, M, M_cor, C=3, image_size = (128, 256), **kwargs):
        super().__init__()

        self.g_s = nn.ModuleList([
            AttentionBlock(M+M_cor),
            ResidualBlock(
                in_ch=M+M_cor,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            AFModule(N, num_dim=1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            AFModule(2*N, num_dim=1),
            AttentionBlock(2*N),
            ResidualBlock(
                in_ch=2*N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            AFModule(N, num_dim=1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=C,
                upsample=2),
            AFModule(N=2*C, num_dim=1),           
            ResidualBlock(
                in_ch=2*C,
                out_ch=C),
        ])


    def forward(self, x, side_info_list):

        if isinstance(x, tuple):
            x, snr = x
        else:
            snr = None
        count = 2
        x = torch.cat([x, side_info_list[-1]], dim = 1)
        for layer in self.g_s:

            if isinstance(layer, AFModule):
                x = layer((x, snr))
            else:
                x = layer(x)
            if isinstance(layer, ResidualBlockUpsample):
                if (count-1) % 2 == 0:
                    x = torch.cat([x, side_info_list[len(side_info_list)-(count+1)//2]], dim = 1)
                # x = torch.cat([x, side_info_list[len(side_info_list)-(count)]], dim = 1)
                count += 1
        return x
class Semantic_Decoder_with_side_info_large_attention(nn.Module):
    def __init__(self, N, M, M_cor, embed_dim, num_heads, image_size, patches, C=3, **kwargs):
        super().__init__()

        self.g_s = nn.ModuleList([
            CrossAttention(input_size=(image_size[0] // 16, image_size[1] // 16), num_filters=M_cor,
                                  dim = embed_dim, num_patches=patches[0], heads=num_heads, dropout=0.1),
            AttentionBlock(M+M_cor),
            ResidualBlock(
                in_ch=M+M_cor,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            # CrossAttention(input_size=(image_size[0] // 8, image_size[1] // 8), num_filters=N,
            #                 dim = embed_dim, num_patches=patches[1], heads=num_heads, dropout=0.1),
            AFModule(N, num_dim=1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            CrossAttention(input_size=(image_size[0] // 4, image_size[1] // 4), num_filters=N,
                                dim = embed_dim, num_patches=patches[1], heads=num_heads, dropout=0.1),
            AFModule(2*N, num_dim=1),
            AttentionBlock(2*N),
            ResidualBlock(
                in_ch=2*N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            # CrossAttention(input_size=(image_size[0] // 2, image_size[1] // 2), num_filters=N,
            #         dim = embed_dim, num_patches=patches[3], heads=num_heads, dropout=0.1),
            AFModule(N, num_dim=1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=C,
                upsample=2),
            CrossAttention(input_size=(image_size[0] , image_size[1] ), num_filters=C,
                              dim = embed_dim, num_patches=patches[2], heads=num_heads, dropout=0.1),
            AFModule(N=2*C, num_dim=1),           
            ResidualBlock(
                in_ch=2*C,
                out_ch=C),
        ])


    def forward(self, x, side_info_list):

        if isinstance(x, tuple):
            x, snr = x
        else:
            snr = None
        count = 1
        # x = torch.cat([x, side_info_list[-1]], dim = 1)
        for layer in self.g_s:
            if isinstance(layer, CrossAttention):
                input = (x, side_info_list[len(side_info_list)-count])
                x = layer(input)
                # x = torch.cat([x, y], dim = 1)
                count += 1
            elif isinstance(layer, AFModule):
                x = layer((x, snr))
            else:
                x = layer(x)
                
  
        return x
# class Semantic_Decoder_with_side_info_large_attention(nn.Module):
#     def __init__(self, N, M, M_cor, embed_dim, num_heads, image_size, patches, C=3, **kwargs):
#         super().__init__()

#         self.g_s = nn.ModuleList([
#             MatchAttentionBlock(img_size=(image_size[0] // 16, image_size[1] // 16), in_chans=M, embed_dims=128, patch_size=patches[0], 
#                                 num_heads=2, mlp_ratios=2, sr_ratios=2, stride=1, depths=1),
#             AttentionBlock(M+M_cor),
#             ResidualBlock(
#                 in_ch=M+M_cor,
#                 out_ch=N),
#             ResidualBlockUpsample(
#                 in_ch=N,
#                 out_ch=N,
#                 upsample=2),
#             # CrossAttention(input_size=(image_size[0] // 8, image_size[1] // 8), num_filters=N,
#             #                 dim = embed_dim, num_patches=patches[1], heads=num_heads, dropout=0.1),
#             AFModule(N, num_dim=1),
#             ResidualBlock(
#                 in_ch=N,
#                 out_ch=N),
#             ResidualBlockUpsample(
#                 in_ch=N,
#                 out_ch=N,
#                 upsample=2),
#             # CrossAttention(input_size=(image_size[0] // 4, image_size[1] // 4), num_filters=N,
#             #                     dim = embed_dim, num_patches=patches[2], heads=num_heads, dropout=0.1),
#             MatchAttentionBlock(img_size=(image_size[0] // 4, image_size[1] // 4), in_chans=N, embed_dims=128, patch_size=patches[2], 
#                     num_heads=2, mlp_ratios=2, sr_ratios=2, stride=1, depths=1),
#             AFModule(2*N, num_dim=1),
#             AttentionBlock(2*N),
#             ResidualBlock(
#                 in_ch=2*N,
#                 out_ch=N),
#             ResidualBlockUpsample(
#                 in_ch=N,
#                 out_ch=N,
#                 upsample=2),
#             # CrossAttention(input_size=(image_size[0] // 2, image_size[1] // 2), num_filters=N,
#             #         dim = embed_dim, num_patches=patches[3], heads=num_heads, dropout=0.1),
#             AFModule(N, num_dim=1),
#             ResidualBlock(
#                 in_ch=N,
#                 out_ch=N),
#             ResidualBlockUpsample(
#                 in_ch=N,
#                 out_ch=C,
#                 upsample=2),
#             # CrossAttention(input_size=(image_size[0] , image_size[1] ), num_filters=C,
#             #                   dim = embed_dim, num_patches=patches[4], heads=num_heads, dropout=0.1),
#             MatchAttentionBlock(img_size=(image_size[0], image_size[1]), in_chans=C, embed_dims=128, patch_size=patches[4], 
#                     num_heads=2, mlp_ratios=2, sr_ratios=2, stride=1, depths=1),
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
#         count = 1
#         # x = torch.cat([x, side_info_list[-1]], dim = 1)
#         for layer in self.g_s:
#             if isinstance(layer, MatchAttentionBlock):
#                 # input = (x, side_info_list[len(side_info_list)-count])
#                 x = layer(x, side_info_list[len(side_info_list)-count])
#                 # x = torch.cat([x, y], dim = 1)
#                 count += 2
#             elif isinstance(layer, AFModule):
#                 x = layer((x, snr))
#             else:
#                 x = layer(x)
                
  
#         return x
# class Semantic_Decoder_with_side_info_large_attention(nn.Module):
#     def __init__(self, N, M, M_cor, embed_dim, num_heads, image_size, patches, C=3, **kwargs):
#         super().__init__()

#         self.g_s = nn.ModuleList([
#             EfficientAttention(key_in_channels=M, query_in_channels=M, key_channels=M//4, 
#             head_count=2, value_channels=M//4),
#             AttentionBlock(M+M_cor),
#             ResidualBlock(
#                 in_ch=M+M_cor,
#                 out_ch=N),
#             ResidualBlockUpsample(
#                 in_ch=N,
#                 out_ch=N,
#                 upsample=2),
#             AFModule(N, num_dim=1),
#             ResidualBlock(
#                 in_ch=N,
#                 out_ch=N),
#             ResidualBlockUpsample(
#                 in_ch=N,
#                 out_ch=N,
#                 upsample=2),
#             EfficientAttention(key_in_channels=N, query_in_channels=N, key_channels=N//8, 
#             head_count=2, value_channels=N//4),
#             AFModule(2*N, num_dim=1),
#             AttentionBlock(2*N),
#             ResidualBlock(
#                 in_ch=2*N,
#                 out_ch=N),
#             ResidualBlockUpsample(
#                 in_ch=N,
#                 out_ch=N,
#                 upsample=2),
#             # CrossAttention(input_size=(image_size[0] // 2, image_size[1] // 2), num_filters=N,
#             #         dim = embed_dim, num_patches=patches[3], heads=num_heads, dropout=0.1),
#             AFModule(N, num_dim=1),
#             ResidualBlock(
#                 in_ch=N,
#                 out_ch=N),
#             ResidualBlockUpsample(
#                 in_ch=N,
#                 out_ch=C,
#                 upsample=2),
#             EfficientAttention(key_in_channels=C, query_in_channels=C, key_channels=2, 
#             head_count=2, value_channels=2),
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
#         count = 1
#         # x = torch.cat([x, side_info_list[-1]], dim = 1)
#         for layer in self.g_s:
#             if isinstance(layer, EfficientAttention):
#                 side_info = layer(x, side_info_list[len(side_info_list)-count])
#                 # residual connection
#                 side_info += side_info_list[len(side_info_list)-count]
#                 x = torch.cat([x, side_info], dim = 1)
#                 count += 2
#             elif isinstance(layer, AFModule):
#                 x = layer((x, snr))
#             else:
#                 x = layer(x)
                
  
#         return x
class symbol_detection_large(nn.Module):
    
    def __init__(self, N, M, config, forusr, atusr, C=3, **kwargs):
        super().__init__()
        self.config = config
        if self.config['scheme_rx'] == "sic_ml" and atusr == "N":
            M_last = 2 * M
        else:
            M_last = M
        if self.config['scheme_rx'] == "defusion_sic" and forusr == "N" and atusr == "N":
            M_first = 2 * M
        else:
            M_first = M
        self.g_s = nn.ModuleList([
            AttentionBlock(M_first),
            ResidualBlock(
                in_ch=M_first,
                out_ch=N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            AFModule(N=N, num_dim=1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            AFModule(N=N, num_dim=1),
            AttentionBlock(N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlock(
                in_ch=N,
                out_ch=M_last),
            AFModule(N=M_last, num_dim=1),
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
