import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import SimpleITK as sitk
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nmse
from skimage.metrics import structural_similarity as ssim
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    cnt, cnt3d = 0, 0
    IPETimg = np.zeros([128, 128, 128])
    SPETimg = np.zeros([128, 128, 128])
    RPETimg = np.zeros([128, 128, 128])
    RSPETimg = np.zeros([128, 128, 128])
    EPETimg = np.zeros([128, 128, 128])
    SRimg = np.zeros([128, 128, 128])
    total_psnr, total_ssim, total_nmse = [], [], []
    total_psnr_ip, total_ssim_ip, total_nmse_ip = [], [], []
    total_psnr_sr, total_ssim_sr, total_nmse_sr = [], [], []
    time_start = time.time()

    for _, val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)
        diffusion.test(continous=False)
        visuals = diffusion.get_current_visuals(need_LR=False)
        image_s = np.squeeze(visuals['HR'].cpu().detach().numpy())
        IP = np.squeeze(visuals['IP'].cpu().detach().numpy())# 获取SR图像
        # SR = np.squeeze(visuals['SR'].cpu().detach().numpy())# 获取SR图像
        # RS = np.squeeze(visuals['RS'].cpu().detach().numpy())
        IPETimg[cnt, :, :] = IP
        SPETimg[cnt, :, :] = image_s
        # EPETimg[cnt, :, :] = SR
        # RSPETimg[cnt, :, :] = RS
        cnt += 1


        if cnt == 128:
            time_end = time.time()
            print('time cost', time_end - time_start, 's')
            time_start = time.time()
            cnt = 0
            cnt3d += 1
            chann, weight, height = IPETimg.shape

            for c in range(chann):  # 遍历高
                for w in range(weight):  # 遍历宽
                    for h in range(height):
                        if IPETimg[c][w][h] <= 0.0:
                            IPETimg[c][w][h] = 0
                        if EPETimg[c][w][h] <= 0.0:
                            EPETimg[c][w][h] = 0
                        if RSPETimg[c][w][h] <= 0.0:
                            RSPETimg[c][w][h] = 0

            y = np.nonzero(SPETimg)  # 取非黑色部分
            SPETimg_1 = SPETimg[y]
            IPETimg_1 = IPETimg[y]
            EPETimg_1 = EPETimg[y]

            # IPET图像指标计算
            cur_psnr_ip = psnr(IPETimg_1, SPETimg_1, data_range=1)
            cur_ssim_ip = ssim(IPETimg, SPETimg, data_range=1)
            cur_nmse_ip = nmse(IPETimg, SPETimg) ** 2
            total_psnr_ip.append(cur_psnr_ip)
            total_ssim_ip.append(cur_ssim_ip)
            total_nmse_ip.append(cur_nmse_ip)


            # cur_psnr_sr = psnr(EPETimg_1, SPETimg_1, data_range=1)
            # cur_ssim_sr = ssim(EPETimg, SPETimg, data_range=1)
            # cur_nmse_sr = nmse(EPETimg, SPETimg) ** 2
            # total_psnr_sr.append(cur_psnr_sr)
            # total_ssim_sr.append(cur_ssim_sr)
            # total_nmse_sr.append(cur_nmse_sr)

            print('IP PSNR: {:6f} SSIM: {:6f} NMSE: {:6f}'.format(cur_psnr_ip, cur_ssim_ip, cur_nmse_ip))
            # print('SR PSNR: {:6f} SSIM: {:6f} NMSE: {:6f}'.format(cur_psnr_sr, cur_ssim_sr, cur_nmse_sr))

            Metrics.save_img(IPETimg, '{}/{}_{}_IP.img'.format(result_path, current_step, cnt3d))
            # Metrics.save_img(EPETimg, '{}/{}_{}_SR.img'.format(result_path, current_step, cnt3d))
            # Metrics.save_img(RSPETimg, '{}/{}_{}_RS.img'.format(result_path, current_step, cnt3d))

            Metrics.save_img(SPETimg, '{}/{}_{}_HR.img'.format(result_path, current_step, cnt3d))

    avg_psnr_ip = np.mean(total_psnr_ip)
    avg_ssim_ip = np.mean(total_ssim_ip)
    avg_nmse_ip = np.mean(total_nmse_ip)

    avg_psnr_sr = np.mean(total_psnr_sr)
    avg_ssim_sr = np.mean(total_ssim_sr)
    avg_nmse_sr = np.mean(total_nmse_sr)
    print('Avg. IP PSNR: {:6f} SSIM: {:6f} NMSE: {:6f}'.format(avg_psnr_ip, avg_ssim_ip, avg_nmse_ip))
    print('Avg. SR PSNR: {:6f} SSIM: {:6f} NMSE: {:6f}'.format(avg_psnr_sr, avg_ssim_sr, avg_nmse_sr))

    if wandb_logger and opt['log_infer']:
        wandb_logger.log_eval_table(commit=True)
