import torch
import torch.nn as nn
from utils import *

class Channel(nn.Module):
    def __init__(self, config, mode, power_norm=False) -> None:
        super().__init__()
        assert mode in ["noiseless", "awgn", "mimo"], f"unrecognized channel mode {mode}"
        self.mode = mode
        self.config = config
        # self.register_buffer("snr", torch.tensor(snr))
        self.power_norm = power_norm

    def forward(self, x, snr, keep_shape):
        device = x.device
        self.snr = snr
        input_shape = x.shape
        # 将x reshape为(B,L)
        # x = x.reshape(input_shape[0], -1)
        if self.mode == "mimo":
            # 发射端天线数目
            Nt = self.config['mimo']['Nt']
            Nr = self.config['mimo']['Nr']
            # 生成信道矢量
            H_com = np.sqrt(1/2)*torch.complex(torch.randn((input_shape[0], Nt, Nr)), torch.randn((input_shape[0], Nt, Nr))).to(device)
            
            # reshape 为一维
            z_in = x.reshape(input_shape[0], -1)
            memory_len = z_in.shape[1]
            if z_in.shape[1] % (2 * Nr) != 0:
                add_len = (z_in.shape[1] // 2 // Nr + 1) * (2 * Nr) - z_in.shape[1] 
                z_in = torch.concat([z_in, torch.ones(z_in.shape[0], add_len).to(device)], dim=1)
            # 变为复数信号
            z_complex  = z_in[:, ::2] + z_in[:, 1::2] * 1j
            num_symbol = z_complex.numel()
            # power norm
            power = torch.mean(torch.real(z_complex * torch.conj(z_complex)), dim=-1)
            z_norm = z_complex / torch.sqrt(power.unsqueeze(-1))
            # layer mapping
            z_com = z_norm.reshape(input_shape[0], Nt, z_norm.shape[1]//Nt)
            if self.config['mimo']['scheme'] == 'svd':
                if self.config['channel']['channel_esti']:
                    z_com_pow = torch.mean(torch.real(z_com * torch.conj(z_com)).reshape(input_shape[0], z_norm.shape[1]), dim=-1)
                    noise_stddev = torch.sqrt(10 ** (-self.snr/ 10))[:, 0] * torch.sqrt(z_com_pow * Nt / 2)
                    pilot = torch.eye(Nt).unsqueeze(0).repeat(H_com.shape[0], 1, 10).to(device)+0j
                    noise_stddev_mat = noise_stddev.reshape(-1, 1, 1).repeat(1, Nt, pilot.shape[2])
                    y_pilot = torch.matmul(H_com, pilot) + noise_stddev_mat*torch.complex(torch.randn(pilot.shape), torch.randn(pilot.shape)).to(device)
                    pilot_H = torch.conj(pilot.transpose(1,2))
                    H_esti = torch.matmul(y_pilot, torch.matmul(pilot_H,torch.inverse(torch.matmul(pilot, pilot_H))))
                    [U, S, V] = torch.svd(H_esti)
                else:
                # svd 分解
                    [U, S, V] = torch.svd(H_com)
                # precoding
                z_pre = torch.matmul(V, z_com)
                z_pre_pow = torch.mean(torch.real(z_pre * torch.conj(z_pre)).reshape(input_shape[0], z_norm.shape[1]), dim=-1)
                noise_stddev = torch.sqrt(10 ** (-self.snr/ 10))[:, 0] * torch.sqrt(z_pre_pow * Nt / 2)
                noise_stddev_mat = noise_stddev.reshape(-1, 1, 1).repeat(1, Nt, z_pre.shape[2])
                noise = noise_stddev_mat*torch.complex(torch.randn(z_pre.shape), torch.randn(z_pre.shape)).to(device)
                y = torch.matmul(H_com, z_pre)+noise
                # 后编码
                y_eq = torch.matmul(torch.conj(U.transpose(1,2)), y)
                #y_eq = torch.matmul(torch.inverse(torch.diag_embed(S))+1j*torch.zeros_like(torch.diag_embed(S)).to(device), y_eq)
                # reshape
                y_eq = y_eq.reshape(input_shape[0], z_norm.shape[1])
                channel_rx = torch.zeros_like(z_in)
                channel_rx[:, ::2] = torch.real(y_eq)
                channel_rx[:, 1::2] = torch.imag(y_eq)
                channel_rx = torch.reshape(channel_rx[:, 0:memory_len], input_shape)
            elif self.config['mimo']['scheme'] == "mmse":
                noise_stddev = torch.sqrt(10 ** (-self.snr/ 10))[:, 0] * np.sqrt(1 * Nt / 2)
                noise_stddev_mat = noise_stddev.reshape(-1, 1, 1).repeat(1, Nt, z_com.shape[2])
                noise = noise_stddev_mat*torch.complex(torch.randn(z_com.shape), torch.randn(z_com.shape)).to(device)
                # 经过信道
                y = torch.matmul(H_com, z_com)+noise
                # mmse 均衡
                eye_mat = torch.eye(Nt).unsqueeze(0).repeat(input_shape[0], 1,1).to(device)
                mmse_factor = torch.matmul(torch.inverse(torch.matmul(torch.conj(H_com).transpose(1,2),H_com)+(noise_stddev**2).unsqueeze(-1).unsqueeze(-1)*eye_mat),torch.conj(H_com).transpose(1,2))
                y_mmse = torch.matmul(mmse_factor, y)
                y_mmse = y_mmse.reshape(input_shape[0], z_norm.shape[1])
                channel_rx = torch.zeros_like(z_in)
                channel_rx[:, ::2] = torch.real(y_mmse)
                channel_rx[:, 1::2] = torch.imag(y_mmse)
                channel_rx = torch.reshape(channel_rx[:, 0:memory_len], input_shape)
            return channel_rx, num_symbol
        else:
            if self.power_norm:
                channel_tx, power = power_norm(x, type = 'complex', keep_shape = False, give_pow=True)
            else:
                channel_tx = x
            
            # input_shape = channel_tx.shape
            channel_in = channel_tx
            channel_in = channel_in[:, ::2] + channel_in[:, 1::2] * 1j
            num_symbol = channel_in.numel()

            if self.mode == "noiseless":
                channel_out = channel_in
            else:
                sigma = torch.sqrt(1.0 / (2 * 10 ** (self.snr / 10)))
                # noise_real = torch.normal(mean=0.0, std=sigma, size=channel_in.shape)
                # noise_imag = torch.normal(mean=0.0, std=sigma, size=channel_in.shape)
                noise_real = torch.randn_like(channel_in, device = device) * sigma
                noise_imag = torch.randn_like(channel_in, device = device) * sigma
                noise = noise_real + noise_imag * 1j
                channel_out = channel_in + noise.to(device)

            channel_rx = torch.zeros_like(channel_tx)
            channel_rx[:, ::2] = torch.real(channel_out)
            channel_rx[:, 1::2] = torch.imag(channel_out)
            if self.power_norm:
                channel_rx = channel_rx * torch.sqrt(power * 2)
            else:
                channel_rx = channel_rx
            if keep_shape:
                channel_rx = torch.reshape(channel_rx, input_shape)
            else: 
                channel_rx = channel_rx

            
            return channel_rx, num_symbol
        


