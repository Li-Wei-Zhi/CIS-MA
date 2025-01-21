import os
import sys
path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   #os.path.dirname通俗的讲是返回上一级文件夹绝对路径的意思，多套一层就多返回一层
sys.path.append(path)    #将路径添加到python的搜索路径中
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio
import argparse
import yaml
import time
from omegaconf import OmegaConf
# from loader import get_dataset, get_dataloader
# from models import get_model
from torch.utils import data as Data
import torchvision
import torchvision.transforms as transforms
import time
from utils import *
from network import MDMA_NOMA
from data_modules.colored_mnist_dataloader import Unified_Dataloader, ColoredMNISTDataset
from data_modules.cifar10_datamodule import CIFAR10DataModule
from data_modules.PairKitti import PairKitti
from data_modules.PairCityscape import PairCityscape
from sklearn import svm
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
from utils import Metrics
from tqdm import tqdm
import json
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import torch.distributed as dist
# 可见设置，环境变量使得指定设备对CUDA应用可见
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
def train_val_one_epoch(epoch, net, train_loader, optimizer_G, config, train):
    global global_step
    if train:
        net.train()
    else:
        net.eval()
    running_loss, running_dist_n, running_dist_f, running_cbr_n, running_cbr_f, \
        running_ssim_n, running_ssim_f, running_psnr_n, running_psnr_f, running_mine = [AverageMeter() for _ in range(10)]
    run_metrics = [running_loss, running_dist_n, running_dist_f, running_cbr_n, running_cbr_f, \
        running_ssim_n, running_ssim_f, running_psnr_n, running_psnr_f, running_mine]
    loop = tqdm((train_loader), total = len(train_loader))
    for input in loop:
        device = config['device']
        img, cor_img, csi_n, csi_f = input
        if train:
            optimizer_G.zero_grad()

        start_time = time.time()
        img = img.to(device)
        cor_img = cor_img.to(device)
        csi_n = csi_n.to(device)
        csi_f = csi_f.to(device)
        global_step += 1
        net_input = (img, cor_img, csi_n, csi_f)
        distortion_n, distortion_f, loss, cbr_n, cbr_f, rho_n, rho_f, s_n_hat, s_f_hat, mine_loss = net(net_input)
        with torch.no_grad():
            psnr_n, ssim_n = metrics(img, s_n_hat)
            psnr_f, ssim_f = metrics(cor_img, s_f_hat)      
        # if cfg['training']['pf_regular']:
        #     regular_t = torch.var(rho_n)
        #     loss = distortion_n + distortion_f + cfg['training']['hyperpara']['alpha'] * distortion_symbol_f - cfg['training']['hyperpara']['pf_regu_f'] * regular_t
        # else:
        #     loss = distortion_n + distortion_f + cfg['training']['hyperpara']['alpha'] * distortion_symbol_f
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer_G.step()
        running_loss.update(loss.item())
        running_dist_n.update(distortion_n.item())
        running_dist_f.update(distortion_f.item())
        running_cbr_n.update(cbr_n)
        running_cbr_f.update(cbr_f)
        running_ssim_n.update(ssim_n.item())
        running_ssim_f.update(ssim_f.item())
        running_psnr_n.update(psnr_n.item())
        running_psnr_f.update(psnr_f.item())      
        running_mine.update(-mine_loss.item())    

        if train:
            loop.set_description(f'Training: Epoch [{epoch+1}/{total_epoch}]')
        else:
            loop.set_description(f'Validation: Epoch [{epoch+1}/{total_epoch}]')
        if config['mine_esti']:
            loop.set_postfix(ssim_n = format(running_ssim_n.avg, '.2f'), psnr_n = format(running_psnr_n.avg, '.2f'), ssim_f = format(running_ssim_f.avg, '.2f'), 
                         psnr_f = format(running_psnr_f.avg, '.2f'), mine = format(running_mine.avg, '.2f'))
        else:
            loop.set_postfix(ssim_n = format(running_ssim_n.avg, '.2f'), psnr_n = format(running_psnr_n.avg, '.2f'), ssim_f = format(running_ssim_f.avg, '.2f'), 
                         psnr_f = format(running_psnr_f.avg, '.2f'))
        # if loss.isnan().any() or loss.isinf().any() or loss > 10000:
        #     continue
    save_image_sample(img, cor_img, s_n_hat, s_f_hat)
    avr_loss = format(running_loss.avg, '.3f')
    avr_dist_n = format(running_dist_n.avg, '.3f')
    avr_dist_f = format(running_dist_f.avg, '.3f')
    avr_cbr_n = format(running_cbr_n.avg, '.3f')
    avr_cbr_f = format(running_cbr_f.avg, '.3f')
    avr_ssim_n = format(running_ssim_n.avg, '.3f')
    avr_ssim_f = format(running_ssim_f.avg, '.3f')
    avr_psnr_n = format(running_psnr_n.avg, '.3f')
    avr_psnr_f = format(running_psnr_f.avg, '.3f') 
    avr_mine = format(running_mine.avg, '.3f')
    for i in run_metrics:
        i.clear()
    # if train:
    #     print("Epoch:", epoch + 1,  "/",  "total_epochs:", config['training']['num_epochs'])
    # else: 
    #     print("Epoch:", epoch + 1, "Validation:")
    # print("loss", avr_loss, "dist_n:", avr_dist_n, "dist_f:", avr_dist_f,  "cbr_n:", avr_cbr_n, 
    #       "cbr_f:", avr_cbr_f, "ssim_n:", avr_ssim_n, "ssim_f:", avr_ssim_f, "psnr_n:", avr_psnr_n, "psnr_f:", avr_psnr_f, 'rho_n:', avr_rho_n, 'rho_f:', avr_rho_f)
    return avr_loss, avr_dist_n, avr_dist_f,  avr_cbr_n, avr_cbr_f, avr_ssim_n, avr_ssim_f, avr_psnr_n, avr_psnr_f, avr_mine
 

def test(net, test_loader_list, config):
    with torch.no_grad():
        net.eval()
        test_result = {}
        for loader_idx in range(len(test_loader_list)):
            test_loader = test_loader_list[loader_idx]
            
            running_loss, running_dist_n, running_dist_f, running_cbr_n, running_cbr_f, \
            running_ssim_n, running_ssim_f, running_psnr_n, running_psnr_f, running_mine = [AverageMeter() for _ in range(10)]
            
            run_metrics = [running_loss, running_dist_n, running_dist_f, running_cbr_n, running_cbr_f, \
                running_ssim_n, running_ssim_f, running_psnr_n, running_psnr_f, running_mine]
            test_inner_result = {}
            key = 'loader_' + str(loader_idx)
            loop = tqdm((test_loader), total = len(test_loader))
            for input in loop:
                device = config['device']
                img, cor_img, csi_n, csi_f = input
                
                # test_result.update({'loader_' + str(loader_idx): csi_n[0][0].detach().cpu().numpy()})

                img = img.to(device)
                cor_img = cor_img.to(device)
                csi_n = csi_n.to(device)
                csi_f = csi_f.to(device)
                net_input = (img, cor_img, csi_n, csi_f)
                distortion_n, distortion_f, loss, cbr_n, cbr_f, rho_n, rho_f, s_n_hat, s_f_hat, mine_loss = net(net_input)
                with torch.no_grad():
                    psnr_n, ssim_n = metrics(img, s_n_hat)
                    psnr_f, ssim_f = metrics(cor_img, s_f_hat)      
        
                # loss = distortion_n + distortion_f + cfg['training']['hyperpara']['alpha'] * distortion_symbol_f

                running_loss.update(loss.item())
                running_dist_n.update(distortion_n.item())
                running_dist_f.update(distortion_f.item())
                # running_dist_xf.update(distortion_symbol_f.item())
                running_cbr_n.update(cbr_n)
                running_cbr_f.update(cbr_f)
                running_ssim_n.update(ssim_n.item())
                running_ssim_f.update(ssim_f.item())
                running_psnr_n.update(psnr_n.item())
                running_psnr_f.update(psnr_f.item())
                running_mine.update(-mine_loss.item())    
                loop.set_description(f'Testing: CSI_N [{csi_n[0][0].detach().cpu().numpy()}], CSI_F [{csi_f[0][0].detach().cpu().numpy()}]')
                # loss = distortion_n + distortion_f + cfg['training']['hyperpara']['alpha'] * distortion_symbol_f
                loop.set_postfix(ssim_n = format(running_ssim_n.avg, '.2f'), psnr_n = format(running_psnr_n.avg, '.2f'), ssim_f = format(running_ssim_f.avg, '.2f'), psnr_f = format(running_psnr_f.avg, '.2f'))
            test_inner_result["csi_n"] = csi_n[0][0].detach().cpu().numpy()
            test_inner_result["csi_f"] = csi_f[0][0].detach().cpu().numpy()
            test_inner_result["loss"]  = format(running_loss.avg, '.3f')
            test_inner_result["dist_n"]  = format(running_dist_n.avg, '.3f')
            test_inner_result["dist_f"]  = format(running_dist_f.avg, '.3f')
            # test_inner_result["dist_xf"]  = format(running_dist_xf.avg, '.3f')
            test_inner_result["cbr_n"]  = format(running_cbr_n.avg, '.3f')
            test_inner_result["cbr_f"]  = format(running_cbr_f.avg, '.3f')
            test_inner_result["ssim_n"]  = format(running_ssim_n.avg, '.3f')
            test_inner_result["ssim_f"]  = format(running_ssim_f.avg, '.3f')
            test_inner_result["psnr_n"]  = format(running_psnr_n.avg, '.3f')
            test_inner_result["psnr_f"]  = format(running_psnr_f.avg, '.3f') 
            test_inner_result["mine"]  = format(running_mine.avg, '.3f')
            test_result[key] = test_inner_result
            for i in run_metrics:
                i.clear()
    return test_result
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/config.yaml')
    parser.add_argument("--DDP", default=False, action="store_true", help="if using data distributed parallel")
    # parser.add_argument("--local_rank", default=os.getenv("LOCAL_RANK", -1), type=int, help="used to identify main process and subprocess")
    parser.add_argument("--logdir", default='results/')
    parser.add_argument("--device", default=0) 
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    # if args.DDP:
    #     cfg["device"] = "DDP"
    if args.device == "cpu":
        cfg["device"] = "cpu"
    else:
        cfg["device"] = f"cuda:{args.device}"
    current_time = time.strftime("%Y%m%d-%H%M%S")
    cfg['exp_name'] = cfg['scheme_tx'] + " & " + cfg['scheme_rx']
    image_size = cfg['data'][cfg['data']['dataset']]['H'] * cfg['data'][cfg['data']['dataset']]['W'] * cfg['data'][cfg['data']['dataset']]['C']
    if cfg['data']['dataset'] == "mnist" or cfg['data']['dataset'] == 'cifar':
        downscale_ratio = 16
    else:
        downscale_ratio = 256
    cbr = 1 * cfg['model']['M']/ (downscale_ratio * 3) / 2
    cbr_str = format(cbr, '.3f')
    if cfg['channel']['mode'] == "mimo":
        mimo_str = "& mimo+"+ cfg['mimo']['scheme']+str(cfg['mimo']['Nt'])+"x"+str(cfg['mimo']['Nr'])
        cfg['exp_name'] += mimo_str
    if cfg['use_cross_attention']:
        cfg['exp_name'] += " & ca"
    if cfg['use_mi_loss']:
        cfg['exp_name'] += " & mi" 
    if cfg['mine_esti']:
        cfg['exp_name'] += " & mine" 
    if cfg['orthogonal']:
        cfg['exp_name'] += " & or" 
    if cfg['use_perfect_side_info']:
        cfg['exp_name'] += " & perf" 
    if cfg['detect_symbol_n'] and cfg['detect_symbol_f']:
        cfg['exp_name'] += " & NOMASC" 
    cfg['exp_name'] = cfg['exp_name'] + " & CBR:" + cbr_str
    cfg['logdir'] = os.path.join(args.logdir, cfg['data']['dataset'], cfg['exp_name'], current_time)
    # file = open(cfg['logdir'] + "/test_result.txt", 'w')
    print(OmegaConf.to_yaml(cfg))
    random.seed(cfg['training']["seed"])
    np.random.seed(cfg['training']["seed"])
    torch.manual_seed(cfg['training']["seed"])
    torch.cuda.manual_seed(cfg['training']["seed"])
    torch.cuda.manual_seed_all(cfg['training']["seed"]) 
    metrics = Metrics(cfg)
    net = MDMA_NOMA(cfg).to(cfg['device']) 

    # net = nn.DataParallel(MDMA_NOMA(cfg).cuda())  
    # dataset 
    if cfg['data']['dataset'] == "mnist":
        train_dataset = ColoredMNISTDataset(cfg, rho_req=0.98, train=True, test_rho=True)
        test_dataset = ColoredMNISTDataset(cfg, rho_req=0.98, train=False, test_rho=True)
    elif cfg['data']['dataset'] == 'cifar':
        train_dataset = CIFAR10DataModule(cfg, train=True)
        test_dataset = CIFAR10DataModule(cfg, train=False)
    elif cfg['data']['dataset'] == "kitti":
        train_dataset = PairKitti(path=cfg['data']['kitti']['root'], set_type='train', config=cfg)
        test_dataset = PairKitti(path=cfg['data']['kitti']['root'], set_type='test', config=cfg)
    elif cfg['data']['dataset'] == "cityscape":
        train_dataset = PairCityscape(path=cfg['data']['cityscape']['root'], set_type='train')
        test_dataset = PairCityscape(path=cfg['data']['cityscape']['root'], set_type='test')
    dloader = Unified_Dataloader(config=cfg)
    
    test_loader_list = Unified_Dataloader(config=cfg).test_loader(test_dataset)
    if not cfg["only_test"]:    
        if cfg['training']['load_pretrained'] == True:
            load_weights(net, model_path=cfg['training']['model_dir'] + "/best_loss.model")
        # make save dir
        os.makedirs(cfg['logdir'], exist_ok=True)
        train_log_filename = "train_log.txt"
        train_log_filepath = os.path.join(cfg['logdir'], train_log_filename)
        train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [dist_n] {dist_n_str} [dist_f]{dist_f_str} [val_psnr_n] {val_psnr_n_str} [val_psnr_f] {val_psnr_f_str} [val_ssim_n] {val_ssim_n_str} [val_ssim_f] {val_ssim_f_str}  [train_loss] {train_loss_str} [val_mine] {val_mine_str}\n"
        # copy config file
        copied_yml = os.path.join(cfg['logdir'], os.path.basename(args.config))
        save_yaml(copied_yml, OmegaConf.to_yaml(cfg))
        print(f"config saved as {copied_yml}")

        optim = torch.optim.Adam(net.parameters(), lr=float(cfg['training']['hyperpara']["lr"]))
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg['training']["num_epochs"])
        
        global global_step
        global_step = 0
        total_epoch = cfg['training']['num_epochs']
        best_loss = float("inf")
        for epoch in range(total_epoch):
            # print('======Current epoch %s ======' % epoch)
            if cfg['data']['dataset'] == "mnist" or cfg['data']['dataset'] == 'cifar':
                trainloader, valloader = dloader.train_val_loader(train_dataset, epoch)
            else:
                trainloader, valloader = dloader.train_val_loader_for_large(train_dataset, test_dataset, epoch)
            loss, dist_n, dist_f, cbr_n, cbr_f, ssim_n, ssim_f, psnr_n, psnr_f, mine = train_val_one_epoch(epoch, net, trainloader, optim, cfg, train = True)
            lr_scheduler.step()
            # validate过程
            val_loss, val_dist_n, val_dist_f, val_cbr_n, val_cbr_f, val_ssim_n, val_ssim_f, \
            val_psnr_n, val_psnr_f, val_mine = train_val_one_epoch(epoch, net, valloader, optim, cfg, train = False)
            val_total_dist = float(val_dist_n) + float(val_dist_f)
            to_write = train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"),
                                                epoch = epoch+1,
                                                dist_n_str = " ".join(["{}".format(val_dist_n,'.3f')]),
                                                dist_f_str = " ".join(["{}".format(val_dist_f,'.3f')]),
                                                val_psnr_n_str =" ".join(["{}".format(val_psnr_n,'.3f')]),
                                                val_psnr_f_str =" ".join(["{}".format(val_psnr_f,'.3f')]), 
                                                val_ssim_n_str = " ".join(["{}".format(val_ssim_n,'.3f')]),
                                                val_ssim_f_str = " ".join(["{}".format(val_ssim_f,'.3f')]),
                                                train_loss_str =" ".join(["{}".format(loss,'.3f')]),
                                                val_mine_str = " ".join(["{}".format(val_mine,'.3f')]))
            with open(train_log_filepath, "a") as f:
                f.write(to_write)
            is_best = val_total_dist < best_loss
            best_loss = min(val_total_dist, best_loss)
            if is_best:
                save_model(net, save_path=cfg['logdir'] + '/best_loss.model')
                print("model saved update !")
        # 开始测试
        save_model(net, save_path=cfg['logdir'] + '/last_epoch.model')
        print("training finished, start testing!")
        # load_weights(net, model_path=cfg['logdir'] + '/best_loss.model')
        test_result = test(net, test_loader_list, cfg)
        # print(test_result)
        # 保存为文件
        # write_dict_to_file(cfg['logdir'] + "/test_result.txt", test_result)
        save_test_result(file_dir=cfg['logdir'] + "/test_result.txt", my_dict=test_result)
        # print("test power allocation")
        # test_loader_list_p = dloader.test_loader_for_pow_allo(test_dataset)
        # test_result_p = test(net, test_loader_list_p, cfg)
        # save_test_result(file_dir=cfg['logdir'] + "/test_result_pow_allo.txt", my_dict=test_result_p)    
    else:
        # 开始测试
        load_weights(net, model_path=cfg['testing']['model_dir'] + "/last_epoch.model")
        net.eval()
        tsne = TSNE(n_components=2, perplexity=30)
        # if cfg['data']['dataset'] == "mnist" or cfg['data']['dataset'] == 'cifar':
        #     trainloader, valloader = dloader.train_val_loader(train_dataset)
        # else:
        #     trainloader, valloader = dloader.train_val_loader_for_large(train_dataset, test_dataset)
        # # validate过程
        # if cfg['validate'] == True:
        #     print("start validating!")
        #     global_step = 0
        #     val_loss, val_dist_n, val_dist_f, val_cbr_n, val_cbr_f, val_ssim_n, val_ssim_f, \
        #     val_psnr_n, val_psnr_f, val_rho_n, val_rho_f = train_val_one_epoch(1, net, valloader, None, cfg, train = False)
        
        print("start testing!")
        test_result = test(net, test_loader_list, cfg)
        # print(test_result)
        #c保存为文件
        save_test_result(file_dir=cfg['testing']['model_dir'] + "/test_result_channel_esti.txt", my_dict=test_result)
        # test_corrleation
        device = cfg['device']
        # all_features = []
        # for i_batch, data1 in enumerate(test_loader_list[-1]):
        #     images, cor_img, csi_near, csi_far = data1
            
        #     # distortion_n, distortion_f, loss, cbr_n, cbr_f, rho_n, rho_f, s_n_hat, s_f_hat = net(input)
 
        #     input = images.to(device), cor_img.to(device), csi_near.to(device), csi_far.to(device)
        #     # net.grad_cam(input, i_batch)
        #     net.test_corrleation(input, i_batch)
        #     # feature = net.tsne(input, i_batch)
        #     # all_features.append(feature.detach().cpu())
            
        #     if i_batch == 10:
        #         break
        # digit_labels = test_dataset.data.targets.cpu().numpy()
        # bg_index_list = test_dataset.bg_index_list
        # fg_index_list = test_dataset.fg_index_list
        # labels = bg_index_list
        # # labels = digit_labels
        
        # feats_input = torch.cat([feature for feature in all_features], axis=0)
        # feats_input = feats_input.reshape(feats_input.shape[0], -1).numpy()
        # embeds = tsne.fit_transform(feats_input)
        # plt.figure(dpi=1000)
        # scatter = plt.scatter(embeds[:,0], embeds[:,1], c=labels, alpha=0.7, cmap = plt.cm.Spectral)
        # plt.legend(handles = scatter.legend_elements()[0], labels=[1,2,3,4,5,6,7,8,9,10,11,12])
        # # plt.legend(handles = scatter.legend_elements()[0], labels=[1,2,3,4,5,6,7,8,9,10], loc=1)
        
        # plt.xlabel("dim 1")
        # plt.ylabel("dim 2")
        # plt.grid()
        # plt.colorbar(ticks=range(10))
        # plt.savefig('tsne_original_data_bg_label.svg', format="svg", dpi=1000)
        # plt.savefig('tsne_original_data_digit_label.svg', format="svg", dpi=1000)
        
        # plt.savefig('tsne_common_info_digit_label.svg', format="svg", dpi=1000)
        # plt.savefig('tsne_common_info_bg_label.svg', format="svg", dpi=1000)
        
        
        # svc_model = svm.SVC(gamma="auto", C=10)
        # svc_model.fit(feats_input[0:6000], labels[0:6000])
        # score = svc_model.score(feats_input[6000:len(feats_input)], labels[6000:len(labels)])
        # print("svm:", score)
        
        # estimator = KNeighborsClassifier()
        # param_dict = {"n_neighbors": [1,2,3,4,5,6,7,8,9,10]}
        # estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
        # estimator.fit(feats_input[0:6000], labels[0:6000])
        # score = estimator.score(feats_input[6000:len(feats_input)], labels[6000:len(labels)])
        # print("KNN: ", score)
        # print(score)
        


        

        
        
