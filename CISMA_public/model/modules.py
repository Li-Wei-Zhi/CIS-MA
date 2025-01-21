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

class Semantic_Encoder_small(nn.Module):

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
            ResidualBlock(
                in_ch=N,
                out_ch=N),
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
            ResidualBlock(
                in_ch=N,
                out_ch=M),
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
        for layer in self.g_a:            
            if isinstance(layer, ResidualBlockWithStride):
                output_list.append(x)
            if isinstance(layer, AFModule):
                x = layer((x, snr))
            else:
                x = layer(x)
        output_list.append(x)
        return x, output_list

class Semantic_Decoder_small(nn.Module):

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
            ResidualBlock(
                in_ch=N,
                out_ch=N),
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

class symbol_detection_small(nn.Module):
    
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

class Common_info_Encoder_small(nn.Module):

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
            ResidualBlock(
                in_ch=N,
                out_ch=N),
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
            ResidualBlock(
                in_ch=N,
                out_ch=M),
            AFModule(N=M, num_dim=1),
            AttentionBlock(M),
        ])
    

    def forward(self, x):
        if isinstance(x, tuple):
            x, snr = x
        else:
            snr = None
        output_list = []
        for layer in self.g_a:            
            if isinstance(layer, ResidualBlockWithStride):
                output_list.append(x)
            if isinstance(layer, AFModule):
                x = layer((x, snr))
            else:
                x = layer(x)
        output_list.append(x)
        return output_list
class Semantic_Decoder_with_side_info_small(nn.Module):
    def __init__(self, N, M, M_cor, C=3, image_size = (28, 28), **kwargs):
        super().__init__()

        self.g_s = nn.ModuleList([
            AttentionBlock(M+M_cor),
            ResidualBlock(
                in_ch=M+M_cor,
                out_ch=N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
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
            ResidualBlock(
                in_ch=N,
                out_ch=N),
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
            # if count < 3:
            if isinstance(layer, ResidualBlockUpsample):
                x = torch.cat([x, side_info_list[len(side_info_list)-count]], dim = 1)
                count += 1
        return x
class Semantic_Decoder_with_side_info_small_attention(nn.Module):
    def __init__(self, N, M, M_cor, embed_dim, num_heads, image_size, patches, C=3, **kwargs):
        super().__init__()

        self.g_s = nn.ModuleList([
            # CrossAttention(input_size=(image_size[0] // 4, image_size[1] // 4), num_filters=M_cor,
            #                       dim = embed_dim, num_patches=patches[0], heads=num_heads, dropout=0.1),
            # AttentionBlock(M+M_cor),
            # ResidualBlock(
            #     in_ch=M+M_cor,
            #     out_ch=N),
            AttentionBlock(M),
            ResidualBlock(
                in_ch=M,
                out_ch=N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            AFModule(N, num_dim=1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            # CrossAttention(input_size=(image_size[0] // 2, image_size[1] // 2), num_filters=N,
            #                     dim = embed_dim, num_patches=patches[1], heads=num_heads, dropout=0.1),
            # AFModule(2*N, num_dim=1),
            # AttentionBlock(2*N),
            # ResidualBlock(
            #     in_ch=2*N,
            #     out_ch=N),
            AFModule(N, num_dim=1),
            AttentionBlock(N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            AFModule(N, num_dim=1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=C,
                upsample=2),
            # CrossAttention(input_size=(image_size[0] , image_size[1] ), num_filters=C,
            #                   dim = embed_dim, num_patches=patches[2], heads=num_heads, dropout=0.1),
            # AFModule(N=2*C, num_dim=1),           
            # ResidualBlock(
            #     in_ch=2*C,
            #     out_ch=C),
            AFModule(N=C, num_dim=1),           
            ResidualBlock(
                in_ch=C,
                out_ch=C),
        ])

    def forward(self, x, side_info_list):

        if isinstance(x, tuple):
            x, snr = x
        else:
            snr = None
        count = 3
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
# class Semantic_Decoder_with_side_info_small_attention(nn.Module):
#     def __init__(self, N, M, M_cor, embed_dim, num_heads, image_size, patches, C=3, **kwargs):
#         super().__init__()

#         self.g_s = nn.ModuleList([
#             ResidualBlock(
#                 in_ch=M,
#                 out_ch=M),
#             CrossAttention(input_size=(image_size[0] // 4, image_size[1] // 4), num_filters=M_cor,
#                                   dim = embed_dim, num_patches=patches[0], heads=num_heads, dropout=0.1),
#             ResidualBlock(
#                 in_ch=M+M_cor,
#                 out_ch=M),
#             ResidualBlock(
#                 in_ch=M,
#                 out_ch=N),
#             AttentionBlock(N),
#             ResidualBlock(
#                 in_ch=N,
#                 out_ch=N),
#             AFModule(N, num_dim=1),
#             # ResidualBlock(
#             #     in_ch=N,
#             #     out_ch=N),
#             ResidualBlockUpsample(
#                 in_ch=N,
#                 out_ch=N,
#                 upsample=2),
#             ResidualBlock(
#                 in_ch=N,
#                 out_ch=N),
#             CrossAttention(input_size=(image_size[0] // 2, image_size[1] // 2), num_filters=N,
#                                 dim = embed_dim, num_patches=patches[1], heads=num_heads, dropout=0.1),
#             ResidualBlock(
#                 in_ch=2*N,
#                 out_ch=N),
#             AFModule(N, num_dim=1),
#             AttentionBlock(N),
#             ResidualBlock(
#                 in_ch=N,
#                 out_ch=N),
#             # ResidualBlock(
#             #     in_ch=N,
#             #     out_ch=N),
#             AFModule(N, num_dim=1),
#             ResidualBlockUpsample(
#                 in_ch=N,
#                 out_ch=C,
#                 upsample=2),          
#             ResidualBlock(
#                 in_ch=C,
#                 out_ch=C),
#             CrossAttention(input_size=(image_size[0] , image_size[1] ), num_filters=C,
#                               dim = embed_dim, num_patches=patches[2], heads=num_heads, dropout=0.1),
#             ResidualBlock(
#                 in_ch=2*C,
#                 out_ch=C),
#             AFModule(N=C, num_dim=1),           
#             ResidualBlock(
#                 in_ch=C,
#                 out_ch=C),
#         ])

#     def forward(self, x, side_info_list):
#         if isinstance(x, tuple):
#             x, snr = x
#         else:
#             snr = None
#         count = 1
#         # x = torch.cat([x, side_info_list[-1]], dim = 1)
#         for idx in range(len(self.g_s)):
#             layer = self.g_s[idx]
#             if idx <= len(self.g_s)-2:
#                 if isinstance(self.g_s[idx+1], CrossAttention):
#                     identity = x
#             if isinstance(layer, CrossAttention):
#                 input = (x, side_info_list[len(side_info_list)-count])
#                 x = layer(input)
#                 # x = torch.cat([x, y], dim = 1)
#                 count += 1
#             elif isinstance(layer, AFModule):
#                 x = layer((x, snr))
#             else:
#                 x = layer(x)
#             if idx >= 1:
#                 if isinstance(self.g_s[idx-1], CrossAttention):
#                     x += identity
                
  
#         return x
    
class fusion_net(nn.Module):
    def __init__(self, embed_dim, h , w,  **kwargs):
        super().__init__()
        self.len = embed_dim * h * w
        self.MLP = Mlp(self.len * 2 + 2, self.len * 4, self.len)
    def forward(self, x_n, x_f, csi_n, csi_f):
        shape = x_n.shape
        x_n_r = x_n.reshape(shape[0], self.len)
        x_f_r = x_f.reshape(shape[0], self.len)
        # csi的shape (B, 1)
        csi_n_embed = csi_n
        csi_f_embed = csi_f
        input = torch.cat([x_n_r, csi_n_embed, x_f_r, csi_f_embed], dim=1)
        y = self.MLP(input)
        # pow = self.dense(y)
        # pow = torch.sigmoid(pow)
        y = y.reshape(shape)
        return y
class power_factor_net(nn.Module):
    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.MLP = Mlp(embed_dim * 2, embed_dim * 4, 1)
        # self.dense = nn.Linear(2*h*w+2, 1)
    def forward(self, csi_n, csi_f):
        # x_n_embed = BCHW2BLN(x_n)
        # x_f_embed = BCHW2BLN(x_f)
        # csi的shape (B, 1)
        csi_n_embed = csi_n.repeat(1, self.embed_dim)
        csi_f_embed = csi_f.repeat(1, self.embed_dim)
        input = torch.cat([csi_n_embed, csi_f_embed], dim=1)
        y = self.MLP(input)
        # pow = self.dense(y)
        pow = torch.sigmoid(y)
        return pow, 1 - pow

# 叠加编码函数
def superimposed_coding(x_n, x_f, rho_n, rho_f, power_norm):
    input_shape_n = x_n.shape
    input_shape_f = x_f.shape
    x_n = x_n.reshape(input_shape_n[0], -1)
    x_f = x_f.reshape(input_shape_f[0], -1)
    
    if power_norm:
       power_n = torch.mean(x_n ** 2, dim = 1).unsqueeze(-1)
       power_f = torch.mean(x_f ** 2, dim = 1).unsqueeze(-1)
       
       channel_tx_n = x_n / torch.sqrt(power_n)
       channel_tx_f = x_f / torch.sqrt(power_f)
       
    else:
       channel_tx_n = x_n
       channel_tx_f = x_f
    # 叠加编码结果
    sc_signal = torch.mul(torch.sqrt(rho_n), channel_tx_n) + torch.mul(torch.sqrt(rho_f), channel_tx_f)
    return sc_signal, input_shape_n, input_shape_f, sc_signal.shape

# 接收端SIC过程
def symbol_sic_process(rx_signal_n, rx_signal_f, decoder_n, decoder_f, ic_net, rho_n, rho_f, snr_n, snr_f, BCHW_shape):
    rx_signal_f_r = rx_signal_f.reshape(BCHW_shape)
    rx_signal_n_r = rx_signal_n.reshape(BCHW_shape)
    
    # f-user, decode s_f directly
    s_f_hat = decoder_f((rx_signal_f_r, snr_f))
    # n-user, estimate x_f first
    x_f_hat = ic_net((rx_signal_n_r, snr_n))
    pow = torch.mean(x_f_hat.reshape(BCHW_shape[0], -1) ** 2, dim = 1).unsqueeze(-1)
    x_f_hat_norm = x_f_hat.reshape(BCHW_shape[0], -1) / torch.sqrt(pow)
    recon_x_f = torch.mul(torch.sqrt(rho_f), x_f_hat_norm)
    residual = (rx_signal_n - recon_x_f) / torch.sqrt(rho_n)
    s_n_hat = decoder_n((residual.reshape(BCHW_shape), snr_n))
    return s_f_hat, x_f_hat, s_n_hat
# 提取共性信息，直传一份共性信息，两份个性信息
def PEARSON(dataA, dataB):
    #[B, C, H, W]
    shape_A = dataA.shape
    shape_B = dataB.shape
    # [B, C, 1, H*W]
    dataA_F = dataA.reshape(shape_A[0], shape_A[1], 1, -1)
    dataB_F = dataB.reshape(shape_B[0], shape_B[1], 1, -1)
    dataA_FM = dataA_F.mean(axis=3, keepdims=True)
    # torch.Size([8, 64, 1, 1])
    dataA_FM_1 = dataA_FM.expand(dataA_F.size())
    dataB_FM = dataB_F.mean(axis=3, keepdims=True)
    dataB_FM_1 = dataB_FM.expand(dataB_F.size())
    # 向量与均值向量的差向量[B, C, 1, H*W]
    dataA_FM_D = dataA_F - dataA_FM_1
    dataB_FM_D = dataB_F - dataB_FM_1
    # 两个差向量的点积求和
    # [B, C, 1, H*W]
    dataAB_FM_DM = torch.mul(dataA_FM_D, dataB_FM_D)
    # [B, C, 1]
    dataA_FM_DS = torch.sum(dataAB_FM_DM, dim=3)
    # 差向量的平方和的平方根的点积torch.Size([8, 64, 1])
    dataA_FM_DN = torch.norm(dataA_FM_D, p=2, dim=3)
    #torch.Size([8, 64, 1])
    dataB_FM_DN = torch.norm(dataB_FM_D, p=2, dim=3)
    #torch.Size([8, 64, 1])
    dataAB_FM_DNM = torch.mul(dataA_FM_DN, dataB_FM_DN)
    # 皮尔逊系数#torch.Size([8, 64, 1])
    COEF = torch.abs(torch.div(dataA_FM_DS, dataAB_FM_DNM))
    # [B, C]
    COEF1 = COEF.reshape(shape_A[0], shape_A[1]).clone()
    [coef_sort, sort_index] = torch.sort(COEF1, dim=1, descending=False)
    # coef_sort_numpy = coef_sort.cpu().detach().numpy()
    # COEF1[COEF1 >= cf_percent/100] = 0  # 高于相似度阈值的置零，留下低相似度
    # person_index = torch.nonzero(COEF1[0])  # 个性特征对应的索引
    [Sort_index_P, index] = torch.sort(sort_index[:, 0:(shape_A[1] // 2)], dim=1, descending=False)
    [Sort_index_S, index] = torch.sort(sort_index[:, (shape_A[1] // 2): shape_A[1]], dim=1, descending=False)
    # print(sort_index)
    # print(Sort_index)
    Data_a = []
    Data_b = []
    Data_ab = []
    for i in range(shape_A[0]):
        data_a = dataA[i, Sort_index_P[i, :], :, :]  # dataA相对于dataB的个性数据
        data_b = dataB[i, Sort_index_P[i, :], :, :]  # dataB相对于dataA的个性数据
        data_ab = (dataB[i, Sort_index_S[i, :], :, :] + dataA[i, Sort_index_S[i, :], :, :]) / 2  # dataB中处理dataB与dataA的共享数据
        # dataB[i, Sort_index_S[i, :], :, :] = (dataB[i, Sort_index_S[i, :], :, :] + dataA[i, Sort_index_S[i, :], :, :]) / 2  # dataB中处理dataB与dataA的共享数据
        # dataA[i, Sort_index_S[i, :], :, :] = 0  # dataA中处理dataB与dataA的共享数据
        Data_a.append(data_a)
        Data_b.append(data_b)
        Data_ab.append(data_ab)
    DataA_P = torch.stack(Data_a)
    DataB_P = torch.stack(Data_b)
    DataAB_S = torch.stack(Data_ab)
    DataAB = torch.cat((DataA_P, DataAB_S, DataB_P), dim=1)
    # DataAB = torch.cat((DataA_P, dataB), dim=1)
    return DataAB, Sort_index_S, Sort_index_P

