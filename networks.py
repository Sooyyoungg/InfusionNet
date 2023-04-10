import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad as torch_grad

from blocks import *
import os
import math
from collections import OrderedDict
import numpy as np
import math

from vgg19 import calc_mean_std

#####################################################################################
## Define Networks : Content & Style Encoder, Decoder(=Generator)
#####################################################################################
def define_network(net_type, alpha_in=0.5, alpha_out=0.5, init_type='normal', gpu_ids=[], config = None):
    net = None

    if net_type == 'Encoder':
        net = Encoder(in_dim=config.input_nc, nf=config.nf, alpha_in=alpha_in, alpha_out=alpha_out)
    elif net_type == 'StyleEncoder':
        net = StyleEncoder(nf=config.nf, style_channel=[512, 512], style_kernel=[3, 3, 3], alpha_in=alpha_in, alpha_out=alpha_out)
    elif net_type == 'Generator':
        net = Decoder(config, nf=config.nf, out_dim=config.output_nc, style_channel=[512, 512], style_kernel=[3, 3, 3], alpha_in=alpha_in, alpha_out=alpha_out)

    return net


class Encoder(nn.Module):    
    def __init__(self, in_dim, nf=64, alpha_in=0.5, alpha_out=0.5, norm='in', pad_type='zeros'):    
        super(Encoder, self).__init__()
        
        # before AOIR: 3 x 256 x 256 -> 64 x 256 x 256
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=nf, kernel_size=7, stride=1, padding=3)        
        
        # 1st AOIR: 128 x 128 x 128    
        self.OctConv1_1 = OctConv(in_channels=nf, out_channels=nf, kernel_size=3, stride=2, padding=1, groups=64, alpha_in=alpha_in, alpha_out=alpha_out, type="first")       
        self.OctConv1_2 = OctConv(in_channels=nf, out_channels=2*nf, kernel_size=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        
        # 2nd AOIR: 256 x 64 x 64     
        self.OctConv2_1 = OctConv(in_channels=2*nf, out_channels=2*nf, kernel_size=3, stride=2, padding=1, groups=128, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv2_2 = OctConv(in_channels=2*nf, out_channels=4*nf, kernel_size=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        
        self.relu = Oct_conv_lreLU()

    def forward(self, x):   
        enc_feat = []
        out = self.conv(x)   
        
        out = self.OctConv1_1(out)
        out = self.relu(out)
        out = self.OctConv1_2(out)
        out = self.relu(out) 
        enc_feat.append(out)
        
        out = self.OctConv2_1(out)   
        out = self.relu(out)
        out = self.OctConv2_2(out)
        out = self.relu(out)
        enc_feat.append(out)

        return out, enc_feat


class StyleEncoder(nn.Module):
    def __init__(self, nf=64, style_channel=[512, 512], style_kernel=[3, 3, 3], alpha_in=0.5, alpha_out=0.5, norm='in', pad_type='reflect'):
        super(StyleEncoder, self).__init__()

        self.style_channel_h = style_channel[0]
        self.style_channel_l = style_channel[1]
        
        self.style_kernel_h = style_kernel[0]
        self.style_kernel_l = style_kernel[1]
        
        self.OctConv1_1 = OctConv(in_channels=4*nf, out_channels=4*nf, kernel_size=3, stride=2, padding=1, groups=4*nf, pad_type=pad_type, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv1_2 = OctConv(in_channels=4*nf, out_channels=8*nf, kernel_size=1, stride=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        
        self.OctConv2_1 = OctConv(in_channels=8*nf, out_channels=8*nf, kernel_size=3, stride=2, padding=1, groups=8*nf, pad_type=pad_type, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv2_2 = OctConv(in_channels=8*nf, out_channels=8*nf, kernel_size=1, stride=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        
        in_hf_channel = int(8*nf * (1 - alpha_in))
        in_lf_channel = 8*nf - in_hf_channel  

        self.OctConv3_1_high = nn.Conv2d(in_channels=in_hf_channel, out_channels=in_hf_channel, kernel_size=3, stride=2, padding=1, groups=in_hf_channel, padding_mode=pad_type)
        self.OctConv3_2_high = nn.Conv2d(in_channels=in_hf_channel, out_channels=self.style_channel_h, kernel_size=1)
        
        self.OctConv3_1_low = nn.Conv2d(in_channels=in_lf_channel, out_channels=in_lf_channel, kernel_size=3, stride=1, padding=1, groups=in_lf_channel, padding_mode=pad_type)
        self.OctConv3_2_low = nn.Conv2d(in_channels=in_lf_channel, out_channels=self.style_channel_l, kernel_size=1)
        
        
        self.pool_h = nn.AdaptiveAvgPool2d((self.style_kernel_h, self.style_kernel_h))
        self.pool_l = nn.AdaptiveAvgPool2d((self.style_kernel_l, self.style_kernel_l))

        self.relu_h = nn.LeakyReLU()
        self.relu = Oct_conv_lreLU()

    def forward(self, x):
        sty_feat = []
        sty = self.OctConv1_1(x)
        sty = self.relu(sty)
        sty = self.OctConv1_2(sty)
        sty = self.relu(sty)
        sty_feat.append(sty)
        
        sty = self.OctConv2_1(sty)
        sty = self.relu(sty)        
        sty = self.OctConv2_2(sty)
        sty = self.relu(sty)
        sty_feat.append(sty)
        
        sty_h, sty_l = sty
        sty_h = self.OctConv3_1_high(sty_h)
        sty_h = self.relu_h(sty_h)
        sty_h = self.OctConv3_2_high(sty_h)
        sty_h = self.relu_h(sty_h)
        
        sty_l = self.OctConv3_1_low(sty_l)
        sty_l = self.relu_h(sty_l)
        sty_l = self.OctConv3_2_low(sty_l)
        sty_l = self.relu_h(sty_l)
        sty_feat.append([sty_h, sty_l])
        
        sty_h = self.pool_h(sty_h)
        sty_l = self.pool_l(sty_l)
        sty_feat.append([sty_h, sty_l])
        
        # 512 x 3 x 3
        sty = sty_h, sty_l
        return sty, sty_feat
    

class Decoder(nn.Module):
    def __init__(self, config, nf=64, out_dim=3, style_channel=[512, 512], style_kernel=[3, 3, 3], alpha_in=0.5, alpha_out=0.5, pad_type='reflect'):
        super(Decoder, self).__init__()

        group_div = [1, 2, 4, 8]
        norm='in'
        self.up_oct = Oct_conv_up(scale_factor=2)

        # 1st AdaOctConv layer
        self.AdaOctConv1_1 = AdaOctConv(in_channels=4*nf, out_channels=4*nf, group_div=group_div[0], style_channels=style_channel, kernel_size=style_kernel, stride=1, padding=1, oct_groups=4*nf, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv1_2 = OctConv(in_channels=4*nf, out_channels=2*nf, kernel_size=1, stride=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.oct_conv_aftup_1 = Oct_Conv_aftup(in_channels=2*nf, out_channels=2*nf, kernel_size=3, stride=1, padding=1, pad_type=pad_type, alpha_in=alpha_in, alpha_out=alpha_out)

        # 2nd AdaOctConv layer
        self.AdaOctConv2_1 = AdaOctConv(in_channels=2*nf, out_channels=2*nf, group_div=group_div[1], style_channels=style_channel, kernel_size=style_kernel, stride=1, padding=1, oct_groups=2*nf, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv2_2 = OctConv(in_channels=2*nf, out_channels=nf, kernel_size=1, stride=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.oct_conv_aftup_2 = Oct_Conv_aftup(64, 64, 3, 1, 1, pad_type, alpha_in, alpha_out)

        # 3rd AdaOctConv layer
        self.AdaOctConv3_1 = AdaOctConv(in_channels=nf, out_channels=nf, group_div=group_div[2], style_channels=style_channel, kernel_size=style_kernel, stride=1, padding=1, oct_groups=nf, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv3_2 = OctConv(in_channels=nf, out_channels=nf//2, kernel_size=1, stride=1, alpha_in=alpha_in, alpha_out=alpha_out, type="last", freq_ratio=config.freq_ratio)
       
        # Conv layer
        self.conv4 = nn.Conv2d(in_channels=nf//2, out_channels=out_dim, kernel_size=1)

    def forward(self, content, style):        
        # 1st AdaOctConv layer
        out, kernel_pred1 = self.AdaOctConv1_1(content, style)
        out = self.OctConv1_2(out)
        out = self.up_oct(out)
        out = self.oct_conv_aftup_1(out)

        # 2nd AdaOctConv layer
        out, kernel_pred2 = self.AdaOctConv2_1(out, style)
        out = self.OctConv2_2(out)
        out = self.up_oct(out)
        out = self.oct_conv_aftup_2(out)

        # 3rd AdaOctConv layer        
        out, kernel_pred3 = self.AdaOctConv3_1(out, style)
        out = self.OctConv3_2(out)
        out, out_high, out_low = out

        # Conv layer
        out = self.conv4(out)
        out_high = self.conv4(out_high)
        out_low = self.conv4(out_low)

        kernel_preds = [kernel_pred1, kernel_pred2, kernel_pred3]
        return out, out_high, out_low, kernel_preds

    
#####################################################################################
## Define Loss function
#####################################################################################
def calc_content_loss(input, target):
    assert (input.size() == target.size())
    mse_loss = nn.MSELoss()
    return mse_loss(input, target)

def calc_style_loss(input, target):
    assert (input.size() == target.size())
    mse_loss = nn.MSELoss()
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)

    loss = mse_loss(input_mean, target_mean) + \
            mse_loss(input_std, target_std)
    return loss


class EFDM_loss(nn.Module):
    def __init__(self):
        super(EFDM_loss, self).__init__()
        self.mse_loss = nn.MSELoss()
    
    def efdm_single(self, style, trans):
        B, C, W, H = style.size(0), style.size(1), style.size(2), style.size(3)
        
        value_style, index_style = torch.sort(style.view(B, C, -1))
        value_trans, index_trans = torch.sort(trans.view(B, C, -1))
        inverse_index = index_trans.argsort(-1)
        
        return self.mse_loss(trans.view(B, C,-1), value_style.gather(-1, inverse_index))

    def forward(self, style_E, style_S, translate_E, translate_S, neg_idx):
        loss = 0.
        batch = style_E[0][0].shape[0]
        for b in range(batch):
            poss_loss = 0.
            neg_loss = 0.
        
            # Positive loss
            for i in range(len(style_E)):
                poss_loss += self.efdm_single(style_E[i][0][b].unsqueeze(0), translate_E[i][0][b].unsqueeze(0)) + \
                            self.efdm_single(style_E[i][1][b].unsqueeze(0), translate_E[i][1][b].unsqueeze(0))
            for i in range(len(style_S)):
                poss_loss += self.efdm_single(style_S[i][0][b].unsqueeze(0), translate_S[i][0][b].unsqueeze(0)) + \
                            self.efdm_single(style_S[i][1][b].unsqueeze(0), translate_S[i][1][b].unsqueeze(0))
                
            # Negative loss
            nb = neg_idx[b]
            for i in range(len(style_E)):
                neg_loss += self.efdm_single(style_E[i][0][nb].unsqueeze(0), translate_E[i][0][b].unsqueeze(0)) + \
                        self.efdm_single(style_E[i][1][nb].unsqueeze(0), translate_E[i][1][b].unsqueeze(0))
            for i in range(len(style_S)):
                neg_loss += self.efdm_single(style_S[i][0][nb].unsqueeze(0), translate_S[i][0][b].unsqueeze(0)) + \
                        self.efdm_single(style_S[i][1][nb].unsqueeze(0), translate_S[i][1][b].unsqueeze(0))
            
            loss += poss_loss / neg_loss
        
        return loss
