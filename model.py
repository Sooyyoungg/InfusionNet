import torch
from torch import nn
from sklearn.metrics import mean_squared_error
import os
import sys
import networks
import gc

from vgg19 import vgg, VGG_loss
from networks import EFDM_loss

class InfusionNet(nn.Module):
    def __init__(self, config):
        super(InfusionNet, self).__init__()

        self.config = config
        self.device = torch.device('cuda:{}'.format(config.gpu) if torch.cuda.is_available() else 'cpu')

        self.lr = config.lr
        self.lambda_percept = config.lambda_percept
        self.lambda_const_style = config.lambda_const_style
        
        self.alpha_in = config.alpha_in
        self.alpha_out = config.alpha_out
        
        torch.cuda.empty_cache()

        # Encoder & Generator
        self.netE = networks.define_network(net_type='Encoder', alpha_in=self.alpha_in, alpha_out=self.alpha_out, 
                                            init_type=self.config.init_type, gpu_ids=self.config.gpu, config = config)
        self.netS = networks.define_network(net_type='StyleEncoder', alpha_in=self.alpha_in, alpha_out=self.alpha_out, 
                                            init_type=self.config.init_type, gpu_ids=self.config.gpu, config = config)
        self.netG = networks.define_network(net_type='Generator', alpha_in=self.alpha_in, alpha_out=self.alpha_out, 
                                            init_type=self.config.init_type, gpu_ids=self.config.gpu, config = config)

        # Loss
        self.vgg_loss = VGG_loss(config, vgg)
        self.efdm_loss = EFDM_loss()

        # Optimizer
        self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.99))
        self.optimizer_S = torch.optim.Adam(self.netS.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.99))
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.99))

        # Scheduler
        self.E_scheduler = networks.get_scheduler(self.optimizer_E, config)
        self.S_scheduler = networks.get_scheduler(self.optimizer_S, config)
        self.G_scheduler = networks.get_scheduler(self.optimizer_G, config)

        
    def forward(self, data):
        self.real_A = data['real_A'].to(self.device)
        self.real_B = data['style_img'].to(self.device)
       
        # 1. Encode content & style information from images
        self.content_A, _ = self.netE(self.real_A)
        self.style_A, _ = self.netS(self.content_A)
        self.content_B, self.cont_feat_B = self.netE(self.real_B)
        self.style_B, self.sty_feat_B = self.netS(self.content_B)
        
        # 2. Generate infusioned images
        self.trs_AtoB, self.trs_AtoB_high, self.trs_AtoB_low, self.sty_kernel_preds = self.netG(self.content_A, self.style_B)

        # 3. Style description of infusioned images
        self.content_trs_AtoB, self.cont_feat_trs_AtoB = self.netE(self.trs_AtoB)
        self.style_trs_AtoB, self.sty_feat_trs_AtoB = self.netS(self.content_trs_AtoB)
        
        ## Predict kernel of infusioned image
        _, _, _, self.trs_kernel_preds = self.netG(self.content_A, self.style_trs_AtoB)

        
    """ Generator loss """
    def calc_G_loss(self):
        ## 1. Perceptual Loss using VGG
        self.G_percept, self.neg_idx = self.vgg_loss.perceptual_loss(self.real_A, self.real_B, self.trs_AtoB)
        self.G_percept *= self.lambda_percept
        
        ## 2. Style Contrastive Loss
        self.G_const_style = self.efdm_loss(self.cont_feat_B, self.sty_feat_B, self.cont_feat_trs_AtoB, self.sty_feat_trs_AtoB, self.neg_idx) * self.lambda_const_style

        self.G_loss = self.G_percept + self.G_const_style

        
    def train_step(self, data):
        self.set_requires_grad([self.netE, self.netS, self.netG], True)

        self.forward(data)
        self.calc_G_loss()

        self.optimizer_E.zero_grad()
        self.optimizer_S.zero_grad()
        self.optimizer_G.zero_grad()
        self.G_loss.backward()
        self.optimizer_E.step()
        self.optimizer_S.step()
        self.optimizer_G.step()

        train_dict = {}
        train_dict['G_loss'] = self.G_loss
        train_dict['G_Percept'] = self.G_percept
        train_dict['G_Const_Style'] = self.G_const_style
        
        train_dict['style_img'] = self.real_B
        train_dict['fake_AtoB'] = self.trs_AtoB
        train_dict['fake_AtoB_high'] = self.trs_AtoB_high
        train_dict['fake_AtoB_low'] = self.trs_AtoB_low
        
        return train_dict

    def val(self, data):
        with torch.no_grad():
            self.forward(data)
        
            val_dict = {}
            val_dict['real_A'] = self.real_A
            val_dict['fake_AtoB'] = self.trs_AtoB
            val_dict['real_B'] = self.real_B

        return val_dict

    def test(self, data):
        with torch.no_grad():
            real_A = data['real_A'].to(self.device)
            real_B = data['style_img'].to(self.device)

            # 1. Encode content & style information from images
            content_A, _ = self.netE(real_A)
            content_B, _ = self.netE(real_B)
            style_B, _ = self.netS(content_B)

            # 2. Generate infusioned images
            trs_AtoB, trs_AtoB_high, trs_AtoB_low, _ = self.netG(content_A, style_B)
            
            test_dict = {}
            test_dict['style_img'] = real_B
            test_dict['fake_AtoB'] = trs_AtoB
            test_dict['fake_AtoB_high'] = trs_AtoB_high
            test_dict['fake_AtoB_low'] = trs_AtoB_low

        return test_dict
    
    def style_mixing(self, data):
        with torch.no_grad():
            # B1: high style image / B2: low style image
            real_A = data['real_A'].to(self.device)
            real_B1_h = data['style_high'].to(self.device)
            real_B2_l = data['style_low'].to(self.device)

            # 1. Encode content & style information from images
            content_A = self.netE(real_A)
            content_B1_h = self.netE(real_B1_h)
            content_B2_l = self.netE(real_B2_l)
            style_B1_h, _ = self.netS(content_B1_h)[0]
            _, style_B2_l = self.netS(content_B2_l)[0]
            style_B = style_B1_h, style_B2_l

            # 2. Generate infusioned images
            trs_AtoB, trs_AtoB_high, trs_AtoB_low, _ = self.netG(content_A, style_B)
            
            mix_dict = {}
            mix_dict['style_high_img'] = real_B1_h
            mix_dict['style_low_img'] = real_B2_l
            mix_dict['fake_AtoB'] = trs_AtoB
            mix_dict['fake_AtoB_high'] = trs_AtoB_high
            mix_dict['fake_AtoB_low'] = trs_AtoB_low

        return mix_dict
    
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

