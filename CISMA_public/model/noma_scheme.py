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
from model.afmodule import AFModule
import os
import sys
from model.modules import * 
from channel import Channel
from utils import *
class eq_pow_sccoding_direct_decoding(nn.Module):
    def __init__(self, config, channel, decoder_n, decoder_f, distortion):
        super().__init__()
        self.config = config
        self.channel = channel
        self.decoder_n = decoder_n
        self.decoder_f = decoder_f
        self.distortion = distortion

    def forward(self, x_n, x_f, csi_n, csi_f):
        x_n_norm = power_norm(x_n, type ='real', keep_shape = True)
        x_f_norm = power_norm(x_f, type ='real', keep_shape = True)
        sc_coding = torch.sqrt(1/2) * x_n_norm + torch.sqrt(1/2) * x_f_norm
        # pass through channel
        y_n, channel_use_n = self.channel(sc_coding, csi_n, keep_shape = True)
        y_f, channel_use_f = self.channel(sc_coding, csi_f, keep_shape = True)
        # directly decoding
        s_n_hat = self.decoder_n(y_n)
        s_f_hat = self.decoder_f(y_f)
        
        return s_n_hat, s_f_hat, channel_use_n, channel_use_f
    
class eq_pow_sccoding_sic_decoding(nn.Module):
    def __init__(self, config, channel, decoder_n, decoder_f, sd_net_n, sd_net_f):
        super().__init__()
        self.config = config
        self.channel = channel
        self.decoder_n = decoder_n
        self.decoder_f = decoder_f
        self.sd_net_n = sd_net_n
        self.sd_net_f = sd_net_f
    def forward(self, x_n, x_f, csi_n, csi_f):
        x_n_norm = power_norm(x_n, type ='real', keep_shape = True)
        x_f_norm = power_norm(x_f, type ='real', keep_shape = True)
        sc_coding = torch.sqrt(1/2) * x_n_norm + torch.sqrt(1/2) * x_f_norm
        # pass through channel
        y_n, channel_use_n = self.channel(sc_coding, csi_n, keep_shape = True)
        y_f, channel_use_f = self.channel(sc_coding, csi_f, keep_shape = True)
        # sic 解码过程
        # 先对远用户进行解码
        # x_f_hat = self.sd_net_f(y_f)
        s_f_hat = self.decoder_f(y_f)
        # 近用户先对远用户的symbol进行estimation
        x_f_hat_n = self.sd_net_f(y_n)
        
        # 重新power norm
        x_f_hat_norm = power_norm(x_f_hat_n, type ='real', keep_shape = True)
        residual = (y_n - torch.sqrt(1/2) * x_f_hat_norm) / torch.sqrt(1/2)
        s_n_hat = self.decoder_n(residual)
        
        return s_n_hat, s_f_hat, x_f_hat_n, channel_use_n, channel_use_f
    
class eq_pow_sccoding_symbol_detection_decoding_no_sic(nn.Module):
    def __init__(self, config, channel, decoder_n, decoder_f, sd_net_n, sd_net_f):
        super().__init__()
        self.config = config
        self.channel = channel
        self.decoder_n = decoder_n
        self.decoder_f = decoder_f
        self.sd_net_n = sd_net_n
        self.sd_net_f = sd_net_f
    def forward(self, x_n, x_f, csi_n, csi_f):
        x_n_norm = power_norm(x_n, type ='real', keep_shape = True)
        x_f_norm = power_norm(x_f, type ='real', keep_shape = True)
        sc_coding = torch.sqrt(1/2) * x_n_norm + torch.sqrt(1/2) * x_f_norm
        # pass through channel
        y_n, channel_use_n = self.channel(sc_coding, csi_n, keep_shape = True)
        y_f, channel_use_f = self.channel(sc_coding, csi_f, keep_shape = True)
        x_n_hat = self.sd_net_n(y_n)
        x_f_hat = self.sd_net_f(y_f)
        s_n_hat = self.decoder_n(x_n_hat)
        s_f_hat = self.decoder_f(x_f_hat)
        
        return s_n_hat, s_f_hat, x_f_hat, x_n_hat, channel_use_n, channel_use_f  
    
class exp_pow_learning_pow_sccoding_sic(nn.Module):
    
    def __init__(self, config, channel, decoder_n, decoder_f, sd_net_n, sd_net_f, pow_factor_net):
        super().__init__()
        self.config = config
        self.channel = channel
        self.decoder_n = decoder_n
        self.decoder_f = decoder_f
        self.sd_net_n = sd_net_n
        self.sd_net_f = sd_net_f
        self.power_factor_net = pow_factor_net
    def forward(self, x_n, x_f, csi_n, csi_f):
        pass

# 发射端scheme
class Tx_scheme(nn.Module):
    def __init__(self, config, power_factor_net, fusion_net):
        super().__init__()
        self.config = config
        self.power_factor_net = power_factor_net
        self.fusion_net = fusion_net
    def eq_pow_sc(self, x_n, x_f, rho_n, rho_f, csi_n, csi_f):
        self.rho_n = 0.5 * torch.ones_like(self.rho_n)
        self.rho_f = 0.5 * torch.ones_like(self.rho_f)
        x_n_norm = power_norm(x_n, type ='real', keep_shape = True)
        x_f_norm = power_norm(x_f, type ='real', keep_shape = True)
        sc_coding = np.sqrt(1/2) * x_n_norm + np.sqrt(1/2) * x_f_norm
        return sc_coding

    def exp_pow_learning_sc(self, x_n, x_f, rho_n, rho_f, csi_n, csi_f):
        input_shape = x_n.shape
        x_n_norm = power_norm(x_n, type ='real', keep_shape = False)
        x_f_norm = power_norm(x_f, type ='real', keep_shape = False)
        sc_signal = torch.mul(torch.sqrt(rho_n), x_n_norm) + torch.mul(torch.sqrt(rho_f), x_f_norm)
        sc_signal = sc_signal.reshape(input_shape)
        return sc_signal

    def inexp_pow_learning_sc(self, x_n, x_f, rho_n, rho_f, csi_n, csi_f):
        sc_signal = x_n + x_f
        sc_signal_norm = power_norm(sc_signal, type ='real', keep_shape = True)
        return sc_signal_norm

    def fusion_scheme(self, x_n, x_f, rho_n, rho_f, csi_n, csi_f):
        sc_signal = self.fusion_net(x_n, x_f, csi_n, csi_f)
        sc_signal_norm = power_norm(sc_signal, type ='real', keep_shape = True)
        return sc_signal_norm
    
    def forward(self, x_n, x_f, csi_n, csi_f):
        self.rho_n, self.rho_f = torch.randn((x_n.shape[0], 1)), torch.randn((x_n.shape[0], 1))
        # self.rho_n, self.rho_f = self.power_factor_net(csi_n, csi_f)
        sc_coding = getattr(self, self.config['scheme_tx'])(x_n, x_f, self.rho_n, self.rho_f, csi_n, csi_f)
        return sc_coding, self.rho_n, self.rho_f
    
# 接收端scheme
class Rx_scheme(nn.Module):
    def __init__(self, config, decoder_n, decoder_f, aux_decoder_f, symbol_n_esti_n, 
                 symbol_f_esti_n, symbol_f_esti_f, common_info_enc_n, common_info_enc_f, decoder_n_with_si, decoder_f_with_si, encoder_n):
        super().__init__()
        self.config = config
        # decode s_n
        self.decoder_n = decoder_n
        # decode s_f
        self.decoder_f = decoder_f
        self.aux_decoder_f = aux_decoder_f
        # detect x_n
        self.symbol_n_esti_n = symbol_n_esti_n
        # detect x_f
        self.symbol_f_esti_n = symbol_f_esti_n
        self.symbol_f_esti_f = symbol_f_esti_f
        # 共性信息提取
        self.common_info_enc_n = common_info_enc_n
        self.common_info_enc_f = common_info_enc_f
        
        # 带有边信息的解码器
        self.decoder_n_with_si = decoder_n_with_si
        self.decoder_f_with_si = decoder_f_with_si
        
        self.encoder_n = encoder_n
        # for SIC-ML
        # self.sic_ml_net = sic_ml_net
        
    def direct_decoding(self, y_n, y_f, csi_n, csi_f, rho_n, rho_f):
        # suitable for 
        # eq_pow_sc, 
        # exp_pow_learning_sc, 
        # inexp_pow_learning_sc, 
        # fusion
        assert self.config['scheme_rx'] == "direct_decoding"
        assert self.config['use_side_info'] == False
        if self.config['detect_symbol_n'] and self.config['detect_symbol_f']:
            x_n_hat_n = self.symbol_n_esti_n((y_n, csi_n))
            x_f_hat_f = self.symbol_f_esti_f((y_f, csi_f))
            s_n_hat = self.decoder_n((x_n_hat_n, csi_n))
            s_f_hat = self.decoder_f((x_f_hat_f, csi_f))
        else:
            s_n_hat = self.decoder_n((y_n, csi_n))
            s_f_hat = self.decoder_f((y_f, csi_f))
            x_n_hat_n = None
            x_f_hat_f = None
        x_f_hat_n = None    
        aux_x_f_n = None
        return s_n_hat, s_f_hat, aux_x_f_n, x_n_hat_n, x_f_hat_f, x_f_hat_n

    def sic_ml(self, y_n ,y_f, csi_n, csi_f, rho_n, rho_f):    
        # suitable for 
        # eq_pow_sc, 
        # exp_pow_learning_sc, 
        # inexp_pow_learning_sc, 
        # fusion
        
        assert self.config['scheme_rx'] == "sic_ml"
        # for f user
        self.sic_ml_net = self.symbol_n_esti_n
        if self.config['detect_symbol_f']:
            x_f_hat_f = self.symbol_f_esti_f((y_f, csi_f))
            s_f_hat = self.decoder_f((x_f_hat_f, csi_f))
        else:
            s_f_hat = self.decoder_f((y_f, csi_f))
            x_f_hat_f = None
        # for n-user
        x_n_hat_n, x_f_hat_n = self.sic_ml_net((y_n, csi_n))[:, 1::2, :, :], self.sic_ml_net((y_n, csi_n))[:, ::2, :, :]
        aux_x_f_n = self.aux_decoder_f((x_f_hat_n, csi_n))
        
        s_n_hat = self.decoder_n((x_n_hat_n, csi_n))

        return s_n_hat, s_f_hat, aux_x_f_n, x_n_hat_n, x_f_hat_f, x_f_hat_n
    
    def defusion_sic(self, y_n, y_f, csi_n, csi_f, rho_n, rho_f):
        # suitable for 
        # inexp_pow_learning_sc, 
        # fusion
        assert self.config['scheme_tx'] != "ep_pow_sc"
        assert self.config['scheme_tx'] != "exp_pow_learning_sc"
        
        # for F user
        if self.config['detect_symbol_f']:
            x_f_hat_f = self.symbol_f_esti_f((y_f, csi_f))
            s_f_hat = self.decoder_f((x_f_hat_f, csi_f))
        else:
            s_f_hat = self.decoder_f((y_f, csi_f))
            x_f_hat_f = None
        # for N user
        x_f_hat_n = self.symbol_f_esti_n((y_n, csi_n))
        aux_x_f_n = self.aux_decoder_f((x_f_hat_n, csi_n))
        
        # defusion 
        x_n_hat_n = self.symbol_n_esti_n((torch.cat([y_n, x_f_hat_n], dim = 1), csi_n))
        s_n_hat = self.decoder_n((x_n_hat_n, csi_n))
        return s_n_hat, s_f_hat, aux_x_f_n, x_n_hat_n, x_f_hat_f, x_f_hat_n
    
    def sic_sccoding(self, y_n, y_f, csi_n, csi_f, rho_n, rho_f):
        # suitable for 
        # ep_pow_sc
        # exp_pow_learning_sc
        assert self.config['scheme_tx'] != "fusion"
        assert self.config['scheme_tx'] != "inexp_pow_learning_sc"
        if self.config['scheme_tx'] == "ep_pow_sc":
            assert torch.mean(rho_n).item() == 0.5
            assert torch.mean(rho_f).item() == 0.5
        if self.config['detect_symbol_f']:
            x_f_hat_f = self.symbol_f_esti_f((y_f, csi_f))
            s_f_hat = self.decoder_f((x_f_hat_f, csi_f))
        else:
            s_f_hat = self.decoder_f((y_f, csi_f))
            x_f_hat_f = None
        # SIC at user_N
        x_f_hat_n = self.symbol_f_esti_n((y_n, csi_n))
        aux_s_f_n = self.aux_decoder_f((x_f_hat_n, csi_n))
        # power norm
        shape = x_f_hat_n.shape
        x_f_hat_n_norm = power_norm(x_f_hat_n, type ='real', keep_shape = False)
        residual = (y_n.reshape(shape[0], -1) - torch.sqrt(rho_f) * x_f_hat_n_norm) / torch.sqrt(rho_n)
        s_n_hat = self.decoder_n((residual.reshape(shape), csi_n))
        if self.config['detect_symbol_n']:
            x_n_hat_n = residual.reshape(shape)
        else:
            x_n_hat_n = None
        return s_n_hat, s_f_hat, aux_s_f_n, x_n_hat_n, x_f_hat_f, x_f_hat_n
    
    def sic_sccoding_unique_cancellation(self, y_n, y_f, csi_n, csi_f, rho_n, rho_f):
        # suitable for 
        # ep_pow_sc
        # exp_pow_learning_sc
        assert self.config['scheme_tx'] != "fusion"
        assert self.config['scheme_tx'] != "inexp_pow_learning_sc"
        if self.config['scheme_tx'] == "ep_pow_sc":
            assert torch.mean(rho_n).item() == 0.5
            assert torch.mean(rho_f).item() == 0.5
        if self.config['detect_symbol_f']:
            x_f_hat_f = self.symbol_f_esti_f((y_f, csi_f))
            s_f_hat = self.decoder_f((x_f_hat_f, csi_f))
        else:
            s_f_hat = self.decoder_f((y_f, csi_f))
            x_f_hat_f = None
        # SIC at user_N
        # detect exclusive info of user-F
        exclu_f = self.symbol_f_esti_n((y_n, csi_n))
        aux_s_f_n = self.aux_decoder_f((exclu_f, csi_n))
        # power norm
        residual = y_n  - exclu_f
        s_n_hat = self.decoder_n((residual, csi_n))

        x_n_hat_n = None
        # aux_s_f_n = None
        x_f_hat_n = None
        return s_n_hat, s_f_hat, aux_s_f_n, x_n_hat_n, x_f_hat_f, x_f_hat_n
    
    def direct_decoding_with_side_info(self, y_n, y_f, csi_n, csi_f, rho_n, rho_f):
        assert self.config['use_side_info'] == True
        if self.config['detect_symbol_f']:
            x_f_hat_f = self.symbol_f_esti_f((y_f, csi_f))
            s_f_hat = self.decoder_f((x_f_hat_f, csi_f))
        else:
            s_f_hat = self.decoder_f((y_f, csi_f))
            x_f_hat_f = None
        # for N user
        # s_f_n = self.aux_decoder_f((y_n, csi_n))
        if self.config['detect_symbol_f']:
            with torch.no_grad():
                x_f_hat_n = self.symbol_f_esti_f((y_n, csi_n))
                s_f_hat_n = self.decoder_f((x_f_hat_n, csi_n))
        else:
            with torch.no_grad():
                s_f_hat_n = self.decoder_f((y_n, csi_n))
            x_f_hat_n = None
            
        side_info_list = self.common_info_enc((s_f_hat_n, csi_n))
        if self.config['detect_symbol_n']:
            x_n_hat_n = self.symbol_n_esti_n((y_n, csi_n))
        # with torch.no_grad():
        #     _, side_info_list = self.encoder_n((s_f_n, csi_n))
            s_n_hat = self.decoder_n_with_si((x_n_hat_n, csi_n), side_info_list)
        else:
            s_n_hat = self.decoder_n_with_si((y_n, csi_n), side_info_list)
            x_n_hat_n = None
            
        # x_f_hat_n = None
        # x_n_hat_n = None  
        s_f_hat_n = None  
        return s_n_hat, s_f_hat, s_f_hat_n, x_n_hat_n, x_f_hat_f, x_f_hat_n
    
    def forward(self, y_n, y_f, csi_n, csi_f, rho_n, rho_f):
        # rx_func = globals()[self.config['scheme_rx']]
        s_n_hat, s_f_hat, aux_x_f_n, x_n_hat_n, x_f_hat_f, x_f_hat_n = getattr(self, self.config['scheme_rx'])(y_n, y_f, csi_n, csi_f, rho_n, rho_f)
        return s_n_hat, s_f_hat, aux_x_f_n, x_n_hat_n, x_f_hat_f, x_f_hat_n
        
        
        
            
            
            
        
    