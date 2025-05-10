import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.autograd as autograd
import os
from functools import partial
from timm.models.layers import DropPath
import model.networks as networks
from torchvision import models
from .base_model import BaseModel
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
import math
import torch.nn.functional as F
from einops import rearrange, repeat
logger = logging.getLogger('base')
from torch.nn.modules.loss import _Loss
from model.sr3_modules.guide import ResidualGuidedMambaBlock
from typing import Callable  # 用于类型提示的 Callable

from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, input, target):
        return F.l1_loss(input, target)

# SSIM 损失函数
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1  # 假设输入是单通道图像

    def gaussian_window(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        # 使用 img1.shape[1] 动态获取输入的通道数
        channel = img1.shape[1]  # 获取输入图像的实际通道数
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        window = self.create_window(self.window_size, channel).to(img1.device)
        return 1 - self.ssim(img1, img2, window, self.window_size, channel, self.size_average)

# 对比边缘损失（Contrastive Edge Loss, CELoss）
class ContrastiveEdgeLoss(nn.Module):
    def __init__(self, beta=0.1):
        super(ContrastiveEdgeLoss, self).__init__()
        self.beta = beta
        # 定义用于边缘检测的卷积核
        self.kernels = nn.ParameterList([
            nn.Parameter(torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0), requires_grad=False),
            nn.Parameter(torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0), requires_grad=False),
            nn.Parameter(torch.tensor([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0), requires_grad=False)
        ])

    def forward(self, sr, hr):
        loss = 0
        for kernel in self.kernels:
            sr_edge = F.conv2d(sr, kernel.to(sr.device), padding=1)
            hr_edge = F.conv2d(hr, kernel.to(hr.device), padding=1)
            loss += F.mse_loss(sr_edge, hr_edge)
        return self.beta * loss

class CombinedLoss(nn.Module):
    def __init__(self, l1_weight=1.0, ssim_weight=0.1, celoss_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss()
        self.celoss = ContrastiveEdgeLoss(beta=celoss_weight)
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight

    def forward(self, sr, hr):
        l1 = self.l1_loss(sr, hr)
        ssim = self.ssim_loss(sr, hr)
        edge = self.celoss(sr, hr)
        total_loss = self.l1_weight * l1 + self.ssim_weight * ssim + edge
        #print(f"L1 Loss: {l1.item():.4f}, SSIM Loss: {ssim.item():.4f}, Edge Loss (CELoss): {edge.item():.4f}")


        return total_loss
class SobelConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, requires_grad=True):
        assert kernel_size % 2 == 1, 'SobelConv2d\'s kernel_size must be odd.'
        assert out_channels % 4 == 0, 'SobelConv2d\'s out_channels must be a multiple of 4.'
        assert out_channels % groups == 0, 'SobelConv2d\'s out_channels must be a multiple of groups.'

        super(SobelConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # In non-trainable case, it turns into normal Sobel operator with fixed weight and no bias.
        self.bias = bias if requires_grad else False

        if self.bias:
            self.bias = nn.Parameter(torch.zeros(size=(out_channels,), dtype=torch.float32), requires_grad=True)
        else:
            self.bias = None

        self.sobel_weight = nn.Parameter(torch.zeros(
            size=(out_channels, int(in_channels / groups), kernel_size, kernel_size)), requires_grad=False)

        # Initialize the Sobel kernal
        kernel_mid = kernel_size // 2
        for idx in range(out_channels):
            if idx % 4 == 0:
                self.sobel_weight[idx, :, 0, :] = -1
                self.sobel_weight[idx, :, 0, kernel_mid] = -2
                self.sobel_weight[idx, :, -1, :] = 1
                self.sobel_weight[idx, :, -1, kernel_mid] = 2
            elif idx % 4 == 1:
                self.sobel_weight[idx, :, :, 0] = -1
                self.sobel_weight[idx, :, kernel_mid, 0] = -2
                self.sobel_weight[idx, :, :, -1] = 1
                self.sobel_weight[idx, :, kernel_mid, -1] = 2
            elif idx % 4 == 2:
                self.sobel_weight[idx, :, 0, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid - i, i] = -1
                    self.sobel_weight[idx, :, kernel_size - 1 - i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, -1, -1] = 2
            else:
                self.sobel_weight[idx, :, -1, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid + i, i] = -1
                    self.sobel_weight[idx, :, i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, 0, -1] = 2

        # Define the trainable sobel factor
        if requires_grad:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=True)
        else:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=False)

    def forward(self, x):
        if torch.cuda.is_available():
            self.sobel_factor = self.sobel_factor.cuda()
            if isinstance(self.bias, nn.Parameter):
                self.bias = self.bias.cuda()

        sobel_weight = self.sobel_weight * self.sobel_factor

        if torch.cuda.is_available():
            sobel_weight = sobel_weight.cuda()

        out = F.conv2d(x, sobel_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class AdaptiveEdgeConnector(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(AdaptiveEdgeConnector, self).__init__()

        self.edge_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.edge_conv2 = nn.Conv2d(in_channels, kernel_size=3, out_channels=out_channels, dilation=2, padding=2)  # 等效 5x5
        self.edge_conv3 = nn.Conv2d(in_channels, kernel_size=3, out_channels=out_channels, dilation=3, padding=3)  # 等效 7x7


        self.gating_conv = nn.Conv2d(out_channels * 3, 3, kernel_size=1) 


        self.cbam = CBAM(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, edges):

        edge_x = self.relu(self.edge_conv1(edges)) 
        edge_y = self.relu(self.edge_conv2(edges)) 
        edge_z = self.relu(self.edge_conv3(edges))

        combined_edges = torch.cat((edge_x, edge_y, edge_z), dim=1)  # (B, 3C, H, W)


        gates = self.gating_conv(combined_edges)  # (B, 3, H, W)
        gates = torch.softmax(gates, dim=1)  

        gates_x = gates[:, 0:1, :, :] 
        gates_y = gates[:, 1:2, :, :]  
        gates_z = gates[:, 2:3, :, :]  

        enhanced_edges = gates_x * edge_x + gates_y * edge_y + gates_z * edge_z


        output_edges = self.cbam(enhanced_edges)

        return output_edges


# define ED-Mamba net
class ED_Mamba(nn.Module):
    def __init__(self, in_ch=1, out_ch=32, sobel_ch=32, guide_dim=64, norm_groups=32):
        super(ED_Mamba, self).__init__()

        self.conv_sobel = SobelConv2d(in_ch, sobel_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.edge_connector = AdaptiveEdgeConnector(sobel_ch, sobel_ch)
        self.guide_block1 = ResidualGuidedMambaBlock(2*out_ch, 4*out_ch, guide_dim=guide_dim, dropout=0.1,drop=0.1)
        self.guide_block2 = ResidualGuidedMambaBlock(8*out_ch, 8*out_ch, guide_dim=guide_dim, dropout=0.2,drop=0.2)
        self.guide_block3 = ResidualGuidedMambaBlock(16*out_ch, 16*out_ch, guide_dim=guide_dim, dropout=0.2,drop=0.4)

        self.conv_p1 = nn.Conv2d( sobel_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f1 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p2 = nn.Conv2d(sobel_ch+ out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p3 = nn.Conv2d( sobel_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p4 = nn.Conv2d(sobel_ch+ out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f4 = nn.Conv2d(out_ch+1, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_f44 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)


        self.downsample1 = nn.Conv2d(out_ch,2*out_ch, kernel_size=3, stride=2, padding=1)

        self.downsample2 = nn.Conv2d(4*out_ch,8*out_ch, kernel_size=3, stride=2, padding=1)

        self.downsample3 = nn.Conv2d(8*out_ch,16*out_ch, kernel_size=3, stride=2, padding=1)



        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_1 = nn.Conv2d(16*out_ch, 8*out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_p15 = nn.Conv2d(16*out_ch, 8*out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f15 = nn.Conv2d(8*out_ch, 8*out_ch, kernel_size=3, stride=1, padding=1)




        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_2 = nn.Conv2d(8*out_ch, 4*out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_p16 = nn.Conv2d(8*out_ch, 4*out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f16 = nn.Conv2d(4*out_ch, 4*out_ch, kernel_size=3, stride=1, padding=1)

   
        self.upsample3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_3 = nn.Conv2d(4*out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_p5 = nn.Conv2d(2*out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_o5 = nn.Conv2d(out_ch, in_ch, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU()

    def forward(self, x, ax_feature=None, fr_feature=None):

        out_0 = self.conv_sobel(x)
        out_1 = torch.cat((x, out_0), dim=1)  
        out_1 = self.relu(self.conv_f4(out_1))


        edge_features = self.edge_connector(out_1)
        out_0 = self.relu(out_1+ edge_features)


        out_1 = self.relu(self.conv_p1(out_0))
        out_1 = self.relu(self.conv_f1(out_1))
        out_1 = torch.cat((out_0, out_1), dim=1)  

        out_2 = self.relu(self.conv_p2(out_1))
        out_2 = self.relu(self.conv_f2(out_2))
        out_2 = torch.cat((out_0, out_2), dim=1)  

        out_3 = self.relu(self.conv_p3(out_2))
        out_3 = self.relu(self.conv_f3(out_3))
        out_3 = torch.cat((out_0, out_3), dim=1) 

        out_3 = self.relu(self.conv_p3(out_3))
        out_3 = self.relu(self.conv_f3(out_3))
        out_3 = torch.cat((out_0, out_3), dim=1)  

        out1 = self.relu(self.conv_p4(out_3))
        out1 = self.relu(self.conv_f44(out1))




        out_1_down1 = self.downsample1(out1)
        out_06 = self.guide_block1(out_1_down1,None, ax_feature[0], fr_feature[0])



        out_2_down1 = self.downsample2(out_06)
        out_2_down = self.guide_block2(out_2_down1,None,ax_feature[1], fr_feature[1])



        out_3_down1 = self.downsample3(out_2_down)
        out_04 = self.guide_block3(out_3_down1, None, ax_feature[2], fr_feature[2])



        out_1_up = self.upsample1(out_04)
        out_15 = self.relu(self.conv_1(out_1_up))
        out_15 = torch.cat((out_15, out_2_down), dim=1)

        out_15 = self.relu(self.conv_p15(out_15))
        out_15 = self.relu(self.conv_f15(out_15))

        out_2_up = self.upsample2(out_15)
        out_16 = self.relu(self.conv_2(out_2_up))
        out_16 = torch.cat((out_16, out_06), dim=1)

        out_16 = self.relu(self.conv_p16(out_16))
        out_16 = self.relu(self.conv_f16(out_16))


        out_3_up = self.upsample3(out_16)

        out_5 = self.relu(self.conv_3(out_3_up))
        out_5 = torch.cat((out1, out_5), dim=1)

        out_5 = self.relu(self.conv_p5(out_5))
        out_5= self.relu(self.conv_o5(out_5))
        out = self.relu(x + out_5)
        return out


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)

        self.netP = self.set_device(ED_Mamba(in_ch=1, out_ch=32, sobel_ch=32,guide_dim=64))
        self.netguide_3D_1 = self.set_device(networks.define_guide(opt, 5, 1))
        self.netguide_3D_2 = self.set_device(networks.define_guide(opt, 5, 2))
        self.netguide_3D_3 = self.set_device(networks.define_guide(opt, 5, 3))
        self.netguide_spectrum_1 = self.set_device(networks.define_guide(opt, 1, 1))
        self.netguide_spectrum_2 = self.set_device(networks.define_guide(opt, 1, 2))
        self.netguide_spectrum_3 = self.set_device(networks.define_guide(opt, 1, 3))
        self.schedule_phase = None
        self.lr = opt['train']['optimizer']['lr']

        self.loss_func = nn.L1Loss(reduction='sum').to(self.device)
        self.compound_loss = CombinedLoss(l1_weight=1, ssim_weight=0.5, celoss_weight=0.5).to(self.device)

        optim_params_P = list(self.netP.parameters())
        optim_params_guide_3D_1 = list(self.netguide_3D_1.parameters())
        optim_params_guide_3D_2 = list(self.netguide_3D_2.parameters())
        optim_params_guide_3D_3 = list(self.netguide_3D_3.parameters())
        optim_params_guide_spectrum_1 = list(self.netguide_spectrum_1.parameters())
        optim_params_guide_spectrum_2 = list(self.netguide_spectrum_2.parameters())
        optim_params_guide_spectrum_3 = list(self.netguide_spectrum_3.parameters())

        self.optP = torch.optim.AdamW(optim_params_P, lr=opt['train']['optimizer']['lr']*2,
                                               weight_decay=0.0001)
        self.optguide_3D_1 = torch.optim.AdamW(optim_params_guide_3D_1, lr=opt['train']['optimizer']['lr'],
                                               weight_decay=0.0001)
        self.optguide_3D_2 = torch.optim.AdamW(optim_params_guide_3D_2, lr=opt['train']['optimizer']['lr'],
                                               weight_decay=0.0001)
        self.optguide_3D_3 = torch.optim.AdamW(optim_params_guide_3D_3, lr=opt['train']['optimizer']['lr'],
                                              weight_decay=0.0001)
        self.optguide_spectrum_1 = torch.optim.AdamW(optim_params_guide_spectrum_1, lr=opt['train']['optimizer']['lr'],
                                                    weight_decay=0.0001)
        self.optguide_spectrum_2 = torch.optim.AdamW(optim_params_guide_spectrum_2, lr=opt['train']['optimizer']['lr'],
                                                     weight_decay=0.0001)
        self.optguide_spectrum_3 = torch.optim.AdamW(optim_params_guide_spectrum_3, lr=opt['train']['optimizer']['lr'],
                                                 weight_decay=0.0001)
        self.lr_scheduler_P = CosineAnnealingLR(self.optP, T_max=50000, eta_min=1e-6)

        self.d_steps = 0
        self.log_dict = OrderedDict()
        self.load_network()

    def feed_data(self, data):
        self.data = self.set_device(data)
    def print_gradients(self, model):
        for name, param in model.named_parameters():
            if param.grad is not None:
                logger.info(f'{name} gradient: {param.grad.norm().item()}')
            else:
                logger.info(f'{name} has no gradient')

    def print_weights(self, model):
        for name, param in model.named_parameters():
            logger.info(f'{name} weight: {param.data.norm().item()}')
    def guide_predict(self):
        # 调用第一个3D网络并计算损失
        _,loss1 = self.netguide_3D_1(self.data['L3D'],self.data['H3D'],t=None)
        _,loss2 = self.netguide_3D_2(self.data['L3D'],self.data['H3D'],t=None)
        _,loss3 = self.netguide_3D_3(self.data['L3D'],self.data['H3D'],t=None)
        # 调用第一个频谱网络并计算损失
        _,loss9 = self.netguide_spectrum_1(self.data['LP'],self.data['HP'],t=None)
        _,loss10 = self.netguide_spectrum_2(self.data['LP'],self.data['HP'],t=None)
        _,loss11 = self.netguide_spectrum_3(self.data['LP'],self.data['HP'],t=None)

        # 从三个3D网络中获取特征，并将它们存储在ax_feature列表中
        ax_feature=[self.netguide_3D_1.get_feature(),self.netguide_3D_2.get_feature(),self.netguide_3D_3.get_feature()]
        # 从三个频谱网络中获取特征，并将它们存储在fr_feature列表中
        fr_feature=[self.netguide_spectrum_1.get_feature(),self.netguide_spectrum_2.get_feature(),self.netguide_spectrum_3.get_feature()]
        loss = loss1+loss2+loss3+loss9+loss10+loss11
        return ax_feature,fr_feature,loss

    def optimize_parameters(self):

        self.optguide_3D_1.zero_grad()
        self.optguide_3D_2.zero_grad()
        self.optguide_3D_3.zero_grad()

        self.optguide_spectrum_1.zero_grad()
        self.optguide_spectrum_2.zero_grad()
        self.optguide_spectrum_3.zero_grad()
        ax_feature, fr_feature, loss_guide = self.guide_predict()
        self.initial_predict(ax_feature, fr_feature)
        self.optP.zero_grad()

        # 加入其他损失，例如感知损失或复合损失
        compound_loss = self.compound_loss(self.IP, self.data['HR'])

        # 计算总损失
        total_loss = compound_loss + loss_guide

        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            logger.warning("NaN or Inf found in total_loss before backward")
            return

        total_loss.backward()
        self.optP.step()
        self.optguide_3D_1.step()
        self.optguide_3D_2.step()
        self.optguide_3D_3.step()
        self.optguide_spectrum_1.step()
        self.optguide_spectrum_2.step()
        self.optguide_spectrum_3.step()

        self.lr_scheduler_P.step()
        current_lr_P = self.lr_scheduler_P.get_last_lr()[0]
        self.log_dict['l_total'] = total_loss.item()
        self.log_dict['loss_guide'] = loss_guide.item()

        self.log_dict['compound_loss'] = compound_loss.item()

        self.log_dict['iter'] = self.log_dict.get('iter', 0) + 1

        logger.info(
           f"Learning rates: PreNet={current_lr_P:.6f}, "
            f'<iter: {self.log_dict["iter"]}> l_total: {self.log_dict["l_total"]:.4f} loss_guide: {self.log_dict["loss_guide"]:.4f} '
            f'compound_loss: {self.log_dict["compound_loss"]:.4f} ')

    def initial_predict(self,ax_feature,fr_feature):
        self.IP = self.netP(self.data['SR'],ax_feature=ax_feature, fr_feature=fr_feature)
    def finale_predict(self,x):
        self.RS = self.netG(x)
    def test(self, continous=False):
        self.netP.eval()
        self.netguide_3D_1.eval()
        self.netguide_3D_2.eval()
        self.netguide_3D_3.eval()
        self.netguide_spectrum_1.eval()
        self.netguide_spectrum_2.eval()
        self.netguide_spectrum_3.eval()


        ax_feature, fr_feature, _ = self.guide_predict()
        with torch.no_grad():
            # 初步预测
            self.IP = self.netP(self.data['SR'],ax_feature=ax_feature, fr_feature=fr_feature)
        self.netP.train()
        self.netguide_3D_1.train()
        self.netguide_3D_2.train()
        self.netguide_3D_3.train()
        self.netguide_spectrum_1.train()
        self.netguide_spectrum_2.train()
        self.netguide_spectrum_3.train()



    def sample(self, batch_size=1, continous=False):
        self.netP.eval()
        with torch.no_grad():
            self.IP = self.netP.sample(batch_size, continous)
        self.netP.train()

    def set_loss(self):
        if isinstance(self.netP, nn.DataParallel):
            self.netP.module.set_loss(self.device)
        else:
            self.netP.set_loss(self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        out_dict['IP'] = self.IP.detach().float().cpu()
        out_dict['HR'] = self.data['HR'].detach().float().cpu()
        out_dict['SR'] = self.data['SR'].float().cpu()

        if need_LR and 'LR' in self.data:
            out_dict['LR'] = self.data['LR'].detach().float().cpu()
        else:
            out_dict['LR'] = out_dict['IP']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netP)
        if isinstance(self.netP, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netP.__class__.__name__,
                                             self.netP.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netP.__class__.__name__)
        logger.info(
            'Network P structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_state_dict(self, net, iter_step, epoch, name):
        if isinstance(net, nn.DataParallel):
            net = net.module
        state_dict = net.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()

        gen_path = os.path.join(self.opt['path']['checkpoint'], f'I{iter_step}_E{epoch}_{name}_gen.pth')
        torch.save(state_dict, gen_path)
        return gen_path

    def save_optimizer_state(self, opt_net, iter_step, epoch, name):
        opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None, 'optimizer': opt_net.state_dict()}
        opt_path = os.path.join(self.opt['path']['checkpoint'], f'I{iter_step}_E{epoch}_{name}_opt.pth')
        torch.save(opt_state, opt_path)

    def save_network(self, epoch, iter_step):
        networks = [
            (self.netP, self.optP, "PreNet"),
            (self.netguide_3D_1, self.optguide_3D_1, "Guide3D_1"),
            (self.netguide_3D_2, self.optguide_3D_2, "Guide3D_2"),
            (self.netguide_3D_3, self.optguide_3D_3, "Guide3D_3"),
            (self.netguide_spectrum_1, self.optguide_spectrum_1, "GuideSpectrum_1"),
            (self.netguide_spectrum_2, self.optguide_spectrum_2, "GuideSpectrum_2"),
            (self.netguide_spectrum_3, self.optguide_spectrum_3, "GuideSpectrum_3")
        ]

        for net, opt_net, name in networks:
            gen_path = self.save_state_dict(net, iter_step, epoch, name)
            self.save_optimizer_state(opt_net, iter_step, epoch, name)

        logger.info(f'Saved model in [{gen_path}] ...')
    def load_network_state(self, network, load_path, model_name):
        gen_path = f'{load_path}_{model_name}_gen.pth'
        logger.info(f'Loading pretrained model for {model_name} [{gen_path}] ...')
        if isinstance(network, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(gen_path), strict=(not self.opt['model']['finetune_norm']))
        return network

    def load_optimizer_state(self, opt_net, load_path, model_name):
        opt_path = f'{load_path}_{model_name}_opt.pth'
        opt = torch.load(opt_path)
        opt_net.load_state_dict(opt['optimizer'])

        self.begin_step = opt['iter']
        self.begin_epoch = opt['epoch']

    def load_network(self):
        if self.opt['path']['resume_state'] is not None:
            load_path = self.opt['path']['resume_state']
            networks = [
                (self.netP, self.optP, "PreNet"),
                (self.netguide_3D_1, self.optguide_3D_1, "Guide3D_1"),
                (self.netguide_3D_2, self.optguide_3D_2, "Guide3D_2"),
                (self.netguide_3D_3, self.optguide_3D_3, "Guide3D_3"),
                (self.netguide_spectrum_1, self.optguide_spectrum_1, "GuideSpectrum_1"),
                (self.netguide_spectrum_2, self.optguide_spectrum_2, "GuideSpectrum_2"),
                (self.netguide_spectrum_3, self.optguide_spectrum_3, "GuideSpectrum_3"),
            ]

            for net, opt_net, name in networks:
                net = self.load_network_state(net, load_path, name)
                if self.opt['phase'] == 'train':
                    self.load_optimizer_state(opt_net, load_path, name)





