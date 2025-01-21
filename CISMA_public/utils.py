import yaml
import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim, ssim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def save_yaml(filename, text):
    """parse string as yaml then dump as a file"""
    with open(filename, "w") as f:
        yaml.dump(yaml.safe_load(text), f, default_flow_style=False)
def BLN2BCHW(x, H, W):
    B, L, N = x.shape
    return x.reshape(B, H, W, N).permute(0, 3, 1, 2)

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def clear(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

def BCHW2BLN(x):
    return x.flatten(2).permute(0, 2, 1)

        
def CalcuPSNR_int(img1, img2, max_val=255.):
    float_type = 'float64'
    img1 = np.round(torch.clamp(img1, 0, 1).detach().cpu().numpy() * 255)
    img2 = np.round(torch.clamp(img2, 0, 1).detach().cpu().numpy() * 255)

    img1 = img1.astype(float_type)
    img2 = img2.astype(float_type)
    mse = np.mean(np.square(img1 - img2), axis=(1, 2, 3))
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr

class Metrics(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
    def PSNR(self, x, y):
        # x是原图， y是恢复的图像
        # psnr = CalcuPSNR_int(x, y)
        mse = torch.mean(torch.square(x - y), axis=(1, 2, 3))
        psnr = 10 * torch.log10(1.0 ** 2 / mse) 
        return torch.mean(psnr)

    def SSIM(self, x, y):
        if self.config['data']['dataset'] == "mnist":
            window_size = 3
        # msssim = ms_ssim(x, y, data_range=1, size_average=True, win_size=window_size)
        if self.config["metrics"]["similarity"] == "ssim":
            ssim_value = ssim(x, y, data_range=1, size_average=True)
        elif self.config["metrics"]["similarity"] == "ms-ssim":
            ssim_value = ms_ssim(x, y, data_range=1, size_average=True, win_size=3)
        return ssim_value

    def forward(self, output, target):
        psnr = self.PSNR(output, target)
        ssim = self.SSIM(output, target)

        return psnr, ssim
    
def load_weights(net, model_path):
    pretrained = torch.load(model_path) # ['state_dict']
    result_dict = {}
    for key, weight in pretrained.items():
        result_key = key
        if 'attn_mask' not in key and 'rate_adaption.mask' not in key:
            result_dict[result_key] = weight
    print(net.load_state_dict(result_dict, strict=False))
    del result_dict, pretrained
    
# 递归遍历字典并将内容写入文件
def save_test_result(file_dir, my_dict):
    file = open(file_dir, 'w')
    def write_dict_to_file(dictionary, indent=''):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                file.write(f"{indent}{key}:\n")
                write_dict_to_file(value, indent + '  ')
            else:
                file.write(f"{indent}{key}: {value}\n")
    write_dict_to_file(my_dict)
    # 关闭文件
    file.close()    
def save_image_sample(img, cor_img, img_r, cor_img_r):
    plt.figure(figsize=(20,20))
    plt.subplot(221)
    plt.imshow(np.transpose(img[0, :, :, :].detach().cpu().numpy(), (1, 2, 0)))
    plt.subplot(222)
    plt.imshow(np.transpose(img_r[0, :, :, :].detach().cpu().numpy(), (1, 2, 0)))
    plt.subplot(223)
    plt.imshow(np.transpose(cor_img[0, :, :, :].detach().cpu().numpy(), (1, 2, 0)))
    plt.subplot(224)
    plt.imshow(np.transpose(cor_img_r[0, :, :, :].detach().cpu().numpy(), (1, 2, 0)))
    plt.savefig("train_img_show.png")

def power_norm(input,  type, keep_shape, give_pow=False):
    input_shape = input.shape
    input = input.reshape(input_shape[0], -1)
    power = torch.mean(input ** 2, dim = 1).unsqueeze(-1)
    if type == "real":
        input_norm = input / torch.sqrt(power)
    elif type == 'complex':
        input_norm = input / torch.sqrt(2 * power)
    if keep_shape:
        output = input_norm.reshape(input_shape)
    else:
        output = input_norm
    if give_pow:
        return output, power
    else:
        return output