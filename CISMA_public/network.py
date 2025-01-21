import os
import sys
path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   #os.path.dirname通俗的讲是返回上一级文件夹绝对路径的意思，多套一层就多返回一层
sys.path.append(path)    #将路径添加到python的搜索路径中
import torch
import torch.nn as nn 
from model.modules import *
from channel import Channel
from distortion import Distortion, DJSLoss
from omegaconf import OmegaConf
from data_modules.colored_mnist_dataloader import ColoredMNISTDataset, Unified_Dataloader
from model.noma_scheme import *
from model.modules_large import *
from model.modules_wo_af import *
import cv2
from model.statistic_network import LocalStatisticsNetwork, MINE
class MDMA_NOMA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config['data']['dataset'] == "mnist" or self.config['data']['dataset'] == "cifar":
            size = 'small'
            self.h = self.config['data'][self.config['data']['dataset']]['H'] // 4
            self.w = self.config['data'][self.config['data']['dataset']]['W'] // 4
        elif self.config['data']['dataset'] == "kitti" or self.config['data']['dataset'] == "cityscape":
            size = 'large'
            self.h = self.config['data'][self.config['data']['dataset']]['H'] // 16
            self.w = self.config['data'][self.config['data']['dataset']]['W'] // 16
        if self.config['data']['dataset'] == "mnist":
            num_patches = (1, 2, 4)
        elif self.config['data']['dataset'] == "cifar":
            num_patches = (2, 2, 4)
        elif self.config['data']['dataset'] == "kitti" or self.config['data']['dataset'] == "cityscape":
            num_patches = (2, 2, 2)
        if self.config['use_af_module']:
            self.encoder_n = globals()["Semantic_Encoder_"+ size](self.config['model']['N'], self.config['model']['M'])
            self.encoder_f = globals()["Semantic_Encoder_"+ size](self.config['model']['N'], self.config['model']['M'])
            self.decoder_n = globals()["Semantic_Decoder_"+ size](self.config['model']['N'], self.config['model']['M'])
            self.decoder_f = globals()["Semantic_Decoder_"+ size](self.config['model']['N'], self.config['model']['M'])
        else:
            self.encoder_n = globals()["Semantic_Encoder_"+ size + "_no_csi"](self.config['model']['N'], self.config['model']['M'])
            self.encoder_f = globals()["Semantic_Encoder_"+ size + "_no_csi"](self.config['model']['N'], self.config['model']['M'])
            self.decoder_n = globals()["Semantic_Decoder_"+ size + "_no_csi"](self.config['model']['N'], self.config['model']['M'])
            self.decoder_f = globals()["Semantic_Decoder_"+ size + "_no_csi"](self.config['model']['N'], self.config['model']['M'])
        # self.aux_decoder_f_n = globals()["Semantic_Decoder_"+ size](self.config['model']['N'], self.config['model']['M'])
        self.aux_decoder_f_n = None
        if self.config["use_side_info"] == True:
            self.common_info_enc_n = globals()["Common_info_Encoder_"+ size](self.config['model']['N'], self.config['model']['M_cor'])
            self.common_info_enc_f = globals()["Common_info_Encoder_"+ size](self.config['model']['N'], self.config['model']['M_cor']) 
            if self.config['use_mi_loss'] == True:
                # self.common_info_enc_n = globals()["Common_info_Encoder_"+ size+"_mi_esti"](self.config['model']['N'], self.config['model']['M_cor'])
                # self.common_info_enc_f = globals()["Common_info_Encoder_"+ size+"_mi_esti"](self.config['model']['N'], self.config['model']['M_cor'])
                # self.statistic_n = LocalStatisticsNetwork(img_feature_channels=3+self.config['model']['N']+self.config['model']['M'])
                # self.statistic_f = LocalStatisticsNetwork(img_feature_channels=3+self.config['model']['N']+self.config['model']['M'])
                self.statistic_n = LocalStatisticsNetwork(img_feature_channels=2*self.config['model']['N']+self.config['model']['M'])
                self.statistic_f = LocalStatisticsNetwork(img_feature_channels=2*self.config['model']['N']+self.config['model']['M'])
            else:
                # self.common_info_enc_n = globals()["Common_info_Encoder_"+ size](self.config['model']['N'], self.config['model']['M_cor'])
                # self.common_info_enc_f = globals()["Common_info_Encoder_"+ size](self.config['model']['N'], self.config['model']['M_cor'])    
                self.statistic_n = None
                self.statistic_f = None       
            if self.config['mine_esti'] == True:
                self.mine_net = MINE(input_channels=2*self.config['model']['N']) 
            if self.config['use_cross_attention']:
                self.decoder_n_with_si = globals()["Semantic_Decoder_with_side_info_" + size + "_attention"](self.config['model']['N'], self.config['model']['M'], 
                                                self.config['model']['M_cor'], self.config['model']['embed_dim'], self.config['model']['heads'], 
                                                image_size = (self.config['data'][self.config['data']['dataset']]["H"], 
                                                              (self.config['data'][self.config['data']['dataset']]["W"])), patches = num_patches)
                self.decoder_f_with_si = globals()["Semantic_Decoder_with_side_info_" + size + "_attention"](self.config['model']['N'], self.config['model']['M'], 
                                                self.config['model']['M_cor'], self.config['model']['embed_dim'], self.config['model']['heads'], 
                                                image_size = (self.config['data'][self.config['data']['dataset']]["H"], 
                                                              (self.config['data'][self.config['data']['dataset']]["W"])), patches = num_patches)
            else:
                self.decoder_n_with_si = globals()["Semantic_Decoder_with_side_info_" + size](self.config['model']['N'], self.config['model']['M'], self.config['model']['M_cor'])
                self.decoder_f_with_si = globals()["Semantic_Decoder_with_side_info_" + size](self.config['model']['N'], self.config['model']['M'], self.config['model']['M_cor'])
                
        else:
            self.common_info_enc_n = None
            self.common_info_enc_f = None
            self.decoder_n_with_si = None
            self.decoder_f_with_si = None
            self.statistic_n = None
            self.statistic_f = None
        
        # self.power_factor_net = power_factor_net(self.config['model']['M'])
        self.power_factor_net = None
        if self.config["scheme_tx"] == "fusion_scheme":
            self.fusion_net = fusion_net(self.config['model']['M'], self.h , self.w)
        else:
            self.fusion_net = None
        if self.config['detect_symbol_n']:
            self.symbol_n_esti_n = globals()["symbol_detection_"+ size](self.config['model']['N'], self.config['model']['M'], self.config, forusr= "N", atusr="N") 
        else:
            self.symbol_n_esti_n = None
        # self.symbol_f_esti_n = globals()["symbol_detection_"+ size](self.config['model']['N'], self.config['model']['M'], self.config, forusr="F", atusr = "N") 
        # self.symbol_n_esti_n = None
        self.symbol_f_esti_n = None
        if self.config['detect_symbol_f']:
            self.symbol_f_esti_f = globals()["symbol_detection_"+ size](self.config['model']['N'], self.config['model']['M'], self.config, forusr="F", atusr = "F")
        else:
            self.symbol_f_esti_f = None
        self.tx_scheme = Tx_scheme(self.config, self.power_factor_net, self.fusion_net)
        self.rx_scheme = Rx_scheme(self.config, self.decoder_n, self.decoder_f, 
                                   self.aux_decoder_f_n, 
                                   self.symbol_n_esti_n, self.symbol_f_esti_n, self.symbol_f_esti_f, 
                                   self.common_info_enc_n, self.common_info_enc_f, 
                                   self.decoder_n_with_si, self.decoder_f_with_si,
                                   self.encoder_n)
            
        self.channel = Channel(self.config,mode=self.config['channel']['mode'], power_norm=self.config['channel']['power_norm'])
        self.distortion = Distortion(self.config)
        self.djloss = DJSLoss()
        self.l1_loss = nn.L1Loss()
        if self.config['scheme_rx'] == "two_stage_decoding_with_side_info":
            self.encoder_n_sec = globals()["Semantic_Encoder_"+ size](self.config['model']['N'], self.config['model']['M_cor'])
            self.encoder_f_sec = globals()["Semantic_Encoder_"+ size](self.config['model']['N'], self.config['model']['M_cor'])
            
            self.decoder_n_with_si = globals()["Semantic_Decoder_with_side_info_" + size](self.config['model']['N'], self.config['model']['M_cor'], self.config['model']['M_cor'])
            self.decoder_f_with_si = globals()["Semantic_Decoder_with_side_info_" + size](self.config['model']['N'], self.config['model']['M_cor'], self.config['model']['M_cor'])
        else:
            self.encoder_n_sec = None
            self.encoder_f_sec = None

    def forward(self, input):
        s_n, s_f, csi_n, csi_f = input
        # semantic coding 
        if self.config['use_af_module']:
            x_n, _ = self.encoder_n((s_n, csi_n))
            x_f, _ = self.encoder_f((s_f, csi_f))
        else:
            x_n, _ = self.encoder_n(s_n)
            x_f, _ = self.encoder_f(s_f)
            if self.training:
                csi_n = self.config['data']['train_fix_snr'] * torch.ones_like(csi_n).to(x_n.device)
                csi_f = self.config['data']['train_fix_snr'] * torch.ones_like(csi_f).to(x_f.device)
        assert x_n.shape[2] == self.h
        assert x_n.shape[3] == self.w
        if self.config["scheme_tx"] == "single_user":
            assert self.config["scheme_rx"] == "direct_decoding"
            x_n_norm = power_norm(x_n, type ='real', keep_shape = True)
            x_f_norm = power_norm(x_f, type ='real', keep_shape = True)
            # sc_coding = np.sqrt(1/2) * x_n_norm + np.sqrt(1/2) * x_f_norm
            # sc_coding = x_n + x_f
            # y_n, channel_use_n = self.channel(sc_coding, csi_n, keep_shape = True)
            # y_f, channel_use_f = self.channel(sc_coding, csi_f, keep_shape = True)
            y_n, channel_use_n = self.channel(x_n_norm, csi_n, keep_shape = True)
            y_f, channel_use_f = self.channel(x_f_norm, csi_f, keep_shape = True)
            rho_n = torch.tensor(1.0)
            rho_f = torch.tensor(1.0) 
            if self.config['use_side_info']:
                # _, side_info_list = self.encoder_n((s_f, csi_n))
                s_f_hat = self.decoder_f((y_f, csi_f))
                side_info_list = self.common_info_enc_n((s_f,csi_n))
                s_n_hat = self.decoder_n_with_si((y_n, csi_n), side_info_list)
            else:
                if self.config['use_af_module']:
                    s_n_hat = self.decoder_n((y_n, csi_n))
                    s_f_hat = self.decoder_f((y_f, csi_f))
                else:
                    s_n_hat = self.decoder_n(y_n)
                    s_f_hat = self.decoder_f(y_f)
            aux_s_f_n = None
            s_n_final = None
            x_f_hat_f = None
            x_n_hat_n = None
            mine_loss = torch.tensor(0)
        elif self.config['scheme_rx'] == "mdma":
            assert self.config['only_test'] == True
            s_n_hat, s_f_hat, channel_use_n, channel_use_f = self.MDMA_transmission(x_n, x_f, csi_n, csi_f)
            # s_n_hat, s_f_hat, channel_use_n, channel_use_f = self.MDMA_PEARSON(x_n, x_f, csi_n, csi_f)
            
            rho_n = torch.tensor(1.0)
            rho_f = torch.tensor(1.0)
            aux_s_f_n = None
            s_n_final = None
            x_f_hat_f = None
            x_n_hat_n = None
        elif self.config['scheme_rx'] == "mdma_pearson":
            assert self.config['only_test'] == True
            # s_n_hat, s_f_hat, channel_use_n, channel_use_f = self.MDMA_transmission(x_n, x_f, csi_n, csi_f)
            s_n_hat, s_f_hat, channel_use_n, channel_use_f = self.MDMA_PEARSON(x_n, x_f, csi_n, csi_f)
            
            rho_n = torch.tensor(1.0)
            rho_f = torch.tensor(1.0)
            aux_s_f_n = None
            s_n_final = None
            x_f_hat_f = None
            x_n_hat_n = None
        else:
            sc_coding, rho_n, rho_f = self.tx_scheme(x_n, x_f, csi_n, csi_f)
            # pass through channel
            y_n, channel_use_n = self.channel(sc_coding, csi_n, keep_shape = True)
            y_f, channel_use_f = self.channel(sc_coding, csi_f, keep_shape = True)
            if self.config['scheme_rx'] == "two_stage_decoding_with_side_info":
                s_n_hat, s_f_hat, aux_s_f_n, aux_s_n_f = self.two_stage_decoding_with_side_info(y_n, y_f, csi_n, csi_f)
                x_n_hat_n, x_f_hat_n, x_f_hat_f = None, None, None
            elif self.config['scheme_rx'] == "direct_decoding_with_side_info":
                if self.config['use_mi_loss'] == True:
                    s_n_hat, s_f_hat, aux_s_f_n, aux_s_n_f, symbol_n_esti, symbol_f_esti,\
                    mi_featn_repf, mi_featn_repf_prime, mi_featf_repn, mi_featf_repn_prime, shared_rep_n, shared_rep_f, mine_loss = self.direct_decoding_with_side_info_with_mi_loss(y_n, y_f, csi_n, csi_f)
                else:
                    s_n_hat, s_f_hat, aux_s_f_n, aux_s_n_f, symbol_n_esti, symbol_f_esti, mine_loss = self.direct_decoding_with_side_info(y_n, y_f, csi_n, csi_f, s_n, s_f)
                    mi_featn_repf, mi_featn_repf_prime, mi_featf_repn, mi_featf_repn_prime, shared_rep_n, shared_rep_f = None, None, None, None, None, None
                x_n_hat_n, x_f_hat_n, x_f_hat_f, s_n_final = None, None, None, None
            # elif self.config['use_perfect_side_info'] == True:
            #     s_f_hat = self.decoder_f((y_f, csi_f))
            #     side_info_list = self.common_info_enc_n((s_f,csi_n))
            #     s_n_hat = self.decoder_n_with_si((y_n, csi_n), side_info_list)
            #     aux_s_f_n, x_n_hat_n, x_f_hat_n, x_f_hat_f, s_n_final, mine_loss = None, None, None, None, None,torch.tensor(0)
            else:
                s_n_hat, s_f_hat, aux_s_f_n, x_n_hat_n, x_f_hat_f, x_f_hat_n = self.rx_scheme(y_n, y_f, csi_n, csi_f, rho_n, rho_f)
                s_n_final, mine_loss = None, torch.tensor(0)
        num_pixels_n = s_n.numel() / s_n.shape[0]
        num_pixels_f = s_f.numel() / s_f.shape[0]
        cbr_n = channel_use_n / s_n.shape[0] / num_pixels_n
        cbr_f = channel_use_f / s_f.shape[0] / num_pixels_f
        distortion_n = self.distortion(s_n, s_n_hat)
        distortion_f = self.distortion(s_f, s_f_hat)
        loss = distortion_n + distortion_f
        if self.config['scheme_rx'] == "direct_decoding_with_side_info":
            distortion_aux_f = self.distortion(s_f, aux_s_f_n)
            distortion_aux_n = self.distortion(s_n, aux_s_n_f)
            loss += self.config['training']['hyperpara']['beta'] * distortion_aux_f
            loss += self.config['training']['hyperpara']['beta'] * distortion_aux_n
            
            if self.config['detect_symbol_n'] and self.config['detect_symbol_f']:
                distortion_sym_n = self.distortion(x_n, symbol_n_esti)
                distortion_sym_f = self.distortion(x_f, symbol_f_esti) 
                loss += distortion_sym_n
                loss += distortion_sym_f
            if self.config['use_mi_loss'] == True:
                mutual_loss_n = self.djloss(
                    T = mi_featn_repf,
                    T_prime = mi_featn_repf_prime)
                mutual_loss_f = self.djloss(
                    T = mi_featf_repn,
                    T_prime = mi_featf_repn_prime
                )
                shared_loss = self.l1_loss(shared_rep_n, shared_rep_f)
                total_mi_loss = mutual_loss_n + mutual_loss_f + shared_loss * self.config['training']['hyperpara']['shared_loss_coeff']
                # total_mi_loss = mutual_loss_n + mutual_loss_f + shared_loss 
                loss += self.config['training']['hyperpara']['mi_loss_coeff'] * total_mi_loss
            if self.config['mine_esti'] == True:
                loss += mine_loss
        #loss = distortion_n
        # if s_n_final is not None:
        #     distortion_s_n_final = self.distortion(s_n, s_n_final)
        #     loss += distortion_s_n_final
        #     s_n_hat = s_n_final
        #     distortion_n = distortion_s_n_final
        # if aux_s_f_n is not None:
        #     distortion_aux_f = self.distortion(s_f, aux_s_f_n)
        #     loss += self.config['training']['hyperpara']['beta'] * distortion_aux_f
        if x_f_hat_f is not None:
            distortion_symbol_f_f = self.distortion(x_f, x_f_hat_f)
            loss += self.config['training']['hyperpara']['alpha'] * distortion_symbol_f_f
        # # if x_f_hat_n is not None:
        # #     distortion_symbol_f_n = self.distortion(x_f, x_f_hat_n)
        # #     loss += self.config['training']['hyperpara']['alpha'] * distortion_symbol_f_n
            
        if x_n_hat_n is not None: 
            distortion_symbol_n_n = self.distortion(x_n, x_n_hat_n)
            loss += self.config['training']['hyperpara']['alpha'] * distortion_symbol_n_n
            
        return distortion_n, distortion_f, loss, cbr_n, cbr_f, rho_n, rho_f, s_n_hat, s_f_hat, mine_loss
    def two_stage_decoding_with_side_info(self, y_n, y_f, csi_n, csi_f):
        assert self.config['use_side_info'] == True
        assert self.config['scheme_rx'] == "two_stage_decoding_with_side_info"
        assert self.config['training']['load_pretrained'] == True
        # if self.config['detect_symbol_f']:
        #     x_f_hat_f = self.symbol_f_esti_f((y_f, csi_f))
        #     s_f_hat = self.decoder_f((x_f_hat_f, csi_f))
        # else:
        #     s_f_hat = self.decoder_f((y_f, csi_f))
        #     x_f_hat_f = None
        # for F user
        s_f_hat_first = self.decoder_f((y_f, csi_f))
        s_n_hat_f = self.decoder_n((y_f, csi_f))
        side_info_list_f = self.common_info_enc_f((s_n_hat_f.detach(), csi_f))
        x_f_reencode, _ = self.encoder_f_sec((s_f_hat_first.detach(), csi_f))
        s_f_final = self.decoder_f_with_si((x_f_reencode, csi_f), side_info_list_f)
        
        # for N user
        s_n_hat_first = self.decoder_n((y_n, csi_n))
        s_f_hat_n = self.decoder_f((y_n, csi_n))
        side_info_list_n = self.common_info_enc_n((s_f_hat_n.detach(), csi_n))
        x_n_reencode, _ = self.encoder_n_sec((s_n_hat_first.detach(), csi_n))
        s_n_final = self.decoder_n_with_si((x_n_reencode, csi_n), side_info_list_n)
        
        return s_n_final, s_f_final, s_f_hat_first, s_n_hat_first
        # return s_n_hat_first, s_n_final, s_f_hat_first, s_f_final
    
    def direct_decoding_with_side_info(self, y_n, y_f, csi_n, csi_f, s_n, s_f):
        assert self.config['use_side_info'] == True
        
        s_n_hat_f = self.decoder_n((y_f, csi_f))
        
        s_f_hat_n = self.decoder_f((y_n, csi_n))
        if not self.config['use_perfect_side_info']:
            side_info_n_list = self.common_info_enc_n((s_n_hat_f.detach(), csi_f))
        elif self.config['use_perfect_side_info']:
            side_info_n_list = self.common_info_enc_n((s_n.detach(), csi_f))
        if self.config['detect_symbol_f']:
        
            symbol_f_esti = self.symbol_f_esti_f((y_f, csi_f))
        
            s_f_hat = self.decoder_f_with_si((symbol_f_esti, csi_f), side_info_n_list)
        else:
            s_f_hat = self.decoder_f_with_si((y_f, csi_f), side_info_n_list)
            symbol_f_esti = None
        if not self.config['use_perfect_side_info']:
            side_info_f_list = self.common_info_enc_f((s_f_hat_n.detach(), csi_n))
        elif self.config['use_perfect_side_info']:
            side_info_f_list = self.common_info_enc_f((s_f.detach(), csi_n))            
        
        if self.config['detect_symbol_n']:
            symbol_n_esti = self.symbol_n_esti_n((y_n, csi_n))
        
            s_n_hat = self.decoder_n_with_si((symbol_n_esti, csi_n), side_info_f_list)
        else:
            s_n_hat = self.decoder_n_with_si((y_n, csi_n), side_info_f_list)
            symbol_n_esti = None
        # experiment
        # s_f_hat = self.decoder_f((y_f, csi_f))
        
        # s_n_hat_f = None
        if self.config['mine_esti'] == True:
            feature_n = side_info_n_list[1]
            feature_f = side_info_f_list[1]
            mine_loss = self.mine_net(feature_n.detach(), feature_f.detach())
        else:
            mine_loss = torch.tensor(0)
        return s_n_hat, s_f_hat, s_f_hat_n, s_n_hat_f, symbol_n_esti, symbol_f_esti, mine_loss
    def direct_decoding_with_side_info_with_mi_loss(self, y_n, y_f, csi_n, csi_f):
        assert self.config['use_side_info'] == True
        assert self.config['use_mi_loss'] == True
        
        s_n_hat_f = self.decoder_n((y_f, csi_f))
        
        s_f_hat_n = self.decoder_f((y_n, csi_n))
        
        # side_info_n_list, feature_n = self.common_info_enc_n((s_n_hat_f.detach(), csi_f))
        side_info_n_list = self.common_info_enc_n((s_n_hat_f.detach(), csi_f))
        feature_n = side_info_n_list[1]
        feature_n_prime = torch.cat([feature_n[1:], feature_n[0].unsqueeze(0)], dim=0)
        times1, times2 = feature_n.shape[2] // side_info_n_list[1].shape[2], feature_n.shape[2] // side_info_n_list[2].shape[2]
        shared_rep_n = torch.cat([side_info_n_list[1].repeat(1,1,times1,times1), side_info_n_list[2].repeat(1,1,times2,times2)], dim = 1)
        # times1, times2, times3, times4 = feature_n.shape[2] // side_info_n_list[1].shape[2], feature_n.shape[2] // side_info_n_list[2].shape[2],\
        #                     feature_n.shape[2] // side_info_n_list[3].shape[2], feature_n.shape[2] // side_info_n_list[4].shape[2]
        # shared_rep_n = torch.cat([side_info_n_list[1].repeat(1,1,times1,times1), side_info_n_list[2].repeat(1,1,times2,times2), 
        #                           side_info_n_list[3].repeat(1,1,times3,times3), side_info_n_list[4].repeat(1,1,times4,times4)], dim = 1)
        
        if self.config['detect_symbol_f']:
        
            symbol_f_esti = self.symbol_f_esti_f((y_f, csi_f))
        
            s_f_hat = self.decoder_f_with_si((symbol_f_esti, csi_f), side_info_n_list)
        else:
            s_f_hat = self.decoder_f_with_si((y_f, csi_f), side_info_n_list)
            symbol_f_esti = None
            
        # side_info_f_list, feature_f = self.common_info_enc_f((s_f_hat_n.detach(), csi_n))
        side_info_f_list = self.common_info_enc_f((s_f_hat_n.detach(), csi_n))
        feature_f = side_info_f_list[1]
        feature_f_prime = torch.cat([feature_f[1:], feature_f[0].unsqueeze(0)], dim=0)
        shared_rep_f = torch.cat([side_info_f_list[1].repeat(1,1,times1,times1), side_info_f_list[2].repeat(1,1,times2,times2)], dim = 1)
        # shared_rep_f = torch.cat([side_info_f_list[1].repeat(1,1,times1,times1), side_info_f_list[2].repeat(1,1,times2,times2), 
        #                           side_info_f_list[3].repeat(1,1,times3,times3), side_info_f_list[4].repeat(1,1,times4,times4)], dim = 1)
        if self.config['detect_symbol_n']:
            symbol_n_esti = self.symbol_n_esti_n((y_n, csi_n))
        
            s_n_hat = self.decoder_n_with_si((symbol_n_esti, csi_n), side_info_f_list)
        else:
            s_n_hat = self.decoder_n_with_si((y_n, csi_n), side_info_f_list)
            symbol_n_esti = None
        # experiment
        # s_f_hat = self.decoder_f((y_f, csi_f))
        
        # s_n_hat_f = None
        concat_featn_repf = torch.cat([shared_rep_f, feature_n], dim = 1)
        concat_featn_repf_prime = torch.cat([shared_rep_f, feature_n_prime], dim = 1)
        
        concat_featf_repn = torch.cat([shared_rep_n, feature_f], dim = 1)     
        concat_featf_repn_prime = torch.cat([shared_rep_n, feature_f_prime], dim = 1)
        
        mi_featn_repf = self.statistic_n(concat_featn_repf)
        mi_featn_repf_prime = self.statistic_n(concat_featn_repf_prime)
        mi_featf_repn = self.statistic_f(concat_featf_repn)
        mi_featf_repn_prime = self.statistic_f(concat_featf_repn_prime)
        if self.config['mine_esti'] == True:
            mine_loss = self.mine_net(feature_n.detach(), feature_f.detach())
        else:
            mine_loss = torch.tensor(0)
        return s_n_hat, s_f_hat, s_f_hat_n, s_n_hat_f, symbol_n_esti, symbol_f_esti, \
                mi_featn_repf, mi_featn_repf_prime, mi_featf_repn, mi_featf_repn_prime, shared_rep_n, shared_rep_f, mine_loss
    def MDMA_transmission(self, x_n, x_f, csi_n, csi_f):
        assert self.config['only_test'] == True
        # s_n, s_f, csi_n, csi_f = input
        # x_n, _ = self.encoder_n((s_n, csi_n))
        # x_f, _ = self.encoder_f((s_f, csi_f))
        orgin_shape = x_n.shape
        x_n_norm = power_norm(x_n, type ='real', keep_shape = True)
        x_f_norm = power_norm(x_f, type ='real', keep_shape = True)
        [b_x, c_x, w_x, h_x] = x_n_norm.shape
        [b_y, c_y, w_y, h_y] = x_f_norm.shape
        feature_X = x_n_norm.reshape(b_x, c_x * w_x * h_x)
        feature_Y = x_f_norm.reshape(b_y, c_y * w_y * h_y)
        # 相加
        multi_coding = feature_X + feature_Y
        # 相减
        multi_diff = feature_X - feature_Y
        # 排序获得idx
        sort_diff_v, sort_diff_idx = torch.sort(torch.abs(multi_diff))
        index_S = sort_diff_idx[:, 0:(c_y * w_y * h_y // 2)]  # 一半的比例共性，差值较小
        index_P = sort_diff_idx[:, (c_y * w_y * h_y // 2):(c_y * w_y * h_y)]  # 一半的比例个性
        for i in range(orgin_shape[0]):
            multi_diff[i, index_S[i, :]] = 0  # 将差值小（对应共性）的位置置零
        indices = multi_diff.nonzero()  # 非零值索引
        nonzero_values = multi_diff[indices[:, 0], indices[:, 1]]  # 取出非零值(对应个性)

        nonzero_values1 = torch.zeros((b_x, c_x // 2, w_x, h_x)).to(x_n.device)
        sub_length = c_y // 2 * w_y * h_y
        for i in range(b_x):
            nonzero_values1[i, :, :, :] = nonzero_values[i * sub_length:sub_length * (i + 1)].reshape(1, c_x // 2, w_x,
                                                                                                      h_x)
        multi_coding_1 = multi_coding.reshape(b_x, c_x, w_x, h_x)
        S_feature = torch.cat([nonzero_values1, multi_coding_1], dim=1)
        if S_feature.shape[1] % 2 != 0:
            zero_padding = torch.zeros(S_feature.shape[0], 1, S_feature.shape[2], S_feature.shape[3]).to(S_feature.device)
            channel_input = torch.cat([S_feature, zero_padding], dim = 1)
        else:
            channel_input = S_feature
        D_feature, channel_use = self.channel(channel_input, csi_n, keep_shape = True)
        if S_feature.shape[1] % 2 != 0:
            channel_output = D_feature[:, 0:D_feature.shape[1]-1, :, :]
        else:
            channel_output = D_feature
        CH = channel_output.shape[1]
        D_get1 = channel_output[:, 0:CH // 3, :, :]
        D_get2 = channel_output[:, CH // 3:CH, :, :]
        D_Get1 = D_get1.reshape(b_y * c_y * w_y * h_y // 2)
        D_Get2 = D_get2.reshape(b_y,  c_y * w_y * h_y)

        Y1 = torch.zeros_like(D_Get2)
        Y1[indices[:, 0], indices[:, 1]] = D_Get1
        # for i in range(b_y):
        #     Y1[i, index_P[i, 0:c_y * w_y * h_y // 2]] = D_Get1[i, 0:c_y * w_y * h_y // 2]
        #     Y1[i, index_S[i, 0:c_y * w_y * h_y // 2]] = 0
        Y2 = D_Get2
        user_1_coding_get = (Y2 + Y1) / 2
        user_2_coding_get = (Y2 - Y1) / 2
        # x_n_fla = x_n_norm.reshape((orgin_shape[0], -1))
        # x_f_fla = x_f_norm.reshape((orgin_shape[0], -1))
        # length = x_n_fla.shape[1]
        # length_reuse = int(1/2 * length)
        # # 两用户信号叠加
        # multi_coding = (x_n_fla + x_f_fla)
        # # 两用户信号相减
        # multi_diff = (x_n_fla - x_f_fla)
        # # 相减后取绝对值排序
        # sort_diff_v, sort_diff_idx = torch.sort(torch.abs(multi_diff),dim = 1)


        # channel_input = torch.cat([multi_coding, multi_diff], dim = 1)
        # channel_output_n, channel_use_n = self.channel(channel_input, csi_n, keep_shape = True)
        # channel_output_f, channel_use_f = self.channel(channel_input, csi_f, keep_shape = True)
        
        # multi_coding_output_n = channel_output_n[:, 0:length]
        # multi_diff_output_n = channel_output_n[:, length:]
        # for i in range(orgin_shape[0]):
        #     multi_diff_output_n[i][sort_diff_idx[i, 0: length_reuse]] = 0 
        # user_n_coding_get = (multi_coding_output_n + multi_diff_output_n) / 2
        # indices = multi_diff_output_n.nonzero()
        # multi_coding_output_f = channel_output_f[:, 0:length]
        # multi_diff_output_f = channel_output_f[:, length:]
        # for i in range(orgin_shape[0]):
        #     multi_diff_output_f[i][sort_diff_idx[i, 0: length_reuse]] = 0 
        # user_f_coding_get = (multi_coding_output_f - multi_diff_output_f) / 2
        s_n_hat = self.decoder_n((user_1_coding_get.reshape(orgin_shape), csi_n))
        s_f_hat = self.decoder_f((user_2_coding_get.reshape(orgin_shape), csi_f))
        return s_n_hat, s_f_hat, channel_use, channel_use
    def MDMA_PEARSON(self, x_n, x_f, csi_n, csi_f):
        orgin_shape = x_n.shape
        x_n_norm = power_norm(x_n, type ='real', keep_shape = True)
        x_f_norm = power_norm(x_f, type ='real', keep_shape = True)
        S_feature, Sort_index_S, Sort_index_P = PEARSON(x_n_norm.clone(), x_f_norm.clone())  # 输出语义特征x和y的相关性
        D_Feature, channel_use = self.channel(S_feature, csi_n, keep_shape = True)
        CH = D_Feature.shape[1]
        DataA_P = D_Feature[:, 0:CH // 3 * 1, :, :]  # 挑出用户1个性特征
        DataB_P = D_Feature[:, CH // 3 * 2:CH, :, :]  # 挑出用户2个性特征
        DataAB_S = D_Feature[:, CH // 3 * 1:CH // 3 * 2, :, :]  # 挑出共性特征
        D_feature1 = torch.zeros_like(x_n_norm)  
        D_feature2 = torch.zeros_like(x_f_norm)  
        for i in range(orgin_shape[0]):
            D_feature1[i, Sort_index_P[i, :], :, :] = DataA_P[i]
            D_feature1[i, Sort_index_S[i, :], :, :] = DataAB_S[i]
            D_feature2[i, Sort_index_P[i, :], :, :] = DataB_P[i]
            D_feature2[i, Sort_index_S[i, :], :, :] = DataAB_S[i]
    
        s_n_hat = self.decoder_n((D_feature1.reshape(orgin_shape), csi_n))
        s_f_hat = self.decoder_f((D_feature2.reshape(orgin_shape), csi_f))
        return s_n_hat, s_f_hat, channel_use, channel_use
    def grad_cam(self, input, idx):
        s_n, s_f, csi_n, csi_f = input
        s_n = s_n[0].unsqueeze(0)
        s_f = s_f[0].unsqueeze(0)
        csi_n = csi_n[0].unsqueeze(0)
        csi_f = csi_f[0].unsqueeze(0)
        s_fmap_block = list()
        s_grad_block = list()  
        # 定义获取特征图的函数
        def s_farward_hook(module, input, output):
            s_fmap_block.append(output)    
        def s_backward_hook(module, grad_in, grad_out):
            s_grad_block.append(grad_out[0].detach())
        self.common_info_enc_n.g_a[6].register_forward_hook(s_farward_hook)	# 9
        self.common_info_enc_n.g_a[6].register_backward_hook(s_backward_hook)
        # self.encoder_n.g_a[6].register_forward_hook(s_farward_hook)	# 9
        # self.encoder_n.g_a[6].register_backward_hook(s_backward_hook)
        # forward propagation process
        x_n, _ = self.encoder_n((s_n, csi_n))
        x_f, _ = self.encoder_f((s_f, csi_f))
        x_n_norm = power_norm(x_n, type ='real', keep_shape = True)
        x_f_norm = power_norm(x_f, type ='real', keep_shape = True)
        sc_coding = np.sqrt(1/2) * x_n_norm + np.sqrt(1/2) * x_f_norm
        y_n, channel_use_n = self.channel(sc_coding, csi_n, keep_shape = True)
        y_f, channel_use_f = self.channel(sc_coding, csi_f, keep_shape = True)
        s_n_hat_f = self.decoder_n((y_f, csi_f))
        
        s_f_hat_n = self.decoder_f((y_n, csi_n))
        
        side_info_list_f = self.common_info_enc_f((s_f_hat_n,csi_n))
        side_info_list_n = self.common_info_enc_n((s_n_hat_f,csi_f))
        feature = side_info_list_n[2][0].flatten()
        for n in range(len(feature)):
            self.common_info_enc_n.zero_grad()
            feature[n].backward(retain_graph=True)
        # feature = x_n[0].flatten()
        # for n in range(len(feature)):
        #     self.encoder_n.zero_grad()
        #     feature[n].backward(retain_graph=True)
        grads_val = s_grad_block[0].cpu().data.numpy().squeeze()
        s_fmap = s_fmap_block[0].cpu().data.numpy().squeeze()
        image_n = np.transpose(s_n[0].detach().cpu().numpy(), (1, 2, 0))
        # 计算grad-cam并可视化
        # self.cam_show_img(image_n, s_fmap, s_grad_block, "exclu_x_cam.png") 
        self.cam_show_img(image_n, s_fmap, s_grad_block, "shared_x_cam.png") 
        
        return x_n   
    def tsne(self, input, idx):
        s_n, s_f, csi_n, csi_f = input
        # forward propagation process
        x_n, _ = self.encoder_n((s_n, csi_n))
        x_f, _ = self.encoder_f((s_f, csi_f))
        x_n_norm = power_norm(x_n, type ='real', keep_shape = True)
        x_f_norm = power_norm(x_f, type ='real', keep_shape = True)
        sc_coding = np.sqrt(1/2) * x_n_norm + np.sqrt(1/2) * x_f_norm
        y_n, channel_use_n = self.channel(sc_coding, csi_n, keep_shape = True)
        y_f, channel_use_f = self.channel(sc_coding, csi_f, keep_shape = True)
        s_n_hat_f = self.decoder_n((y_f, csi_f))
        s_f_hat_n = self.decoder_f((y_n, csi_n))
        side_info_list_f = self.common_info_enc_f((s_f_hat_n,csi_n))
        side_info_list_n = self.common_info_enc_n((s_n_hat_f,csi_f))
        return side_info_list_n[2]
    def cam_show_img(self, img, feature_map, s_grad_block, img_name):
        H, W, _ = img.shape
        cam_list = []
        # plt.figure(figsize=(40, 40), dpi = 1000)
        plt.figure(dpi=1000)
        # for i in range(len(s_grad_block)):
        sum_heat_map = 0
        for i in range(64):
            cam = np.zeros(feature_map.shape[1:], dtype=np.float32)		# 4
            grads = s_grad_block[i].cpu().data.numpy().squeeze()
            grads = grads.reshape([grads.shape[0],-1])					# 5
            weights = np.mean(grads, axis=1)							# 6
            for j, w in enumerate(weights):
                cam += w * feature_map[j, :, :]							# 7
            # cam = np.maximum(cam, 0)
            cam = cam / cam.max()
            cam = cv2.resize(cam, (W, H))
            cam_list.append(cam)
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            
            # cam_img = 0.2 * heatmap + 0.8 * np.uint8(img * 255)
            sum_heat_map += cam 
            # cam_img = 1.0 * heatmap
            # cam_img /= 255
            # cam_img = np.uint8(img) * 255
            # plt.subplot(8, 8, i+1)
            # plt.imshow(cam_img)
            # plt.imshow(heatmap)
        # avr_heat_map = np.mean(np.array(cam_list), axis = 0)
        avr_heat_map = sum_heat_map / 64
        heatmap = cv2.applyColorMap(np.uint8(255 * avr_heat_map), cv2.COLORMAP_JET)
        plt.imshow(heatmap)
        plt.savefig("shared_x_grad_cam.svg", format = "svg", dpi = 1000)    

   

        
