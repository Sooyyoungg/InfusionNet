import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import numpy as np

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class VGG_loss(nn.Module):
    def __init__(self, config, vgg):
        super(VGG_loss, self).__init__()
       
        self.config = config

        vgg_pretrained = config.vgg_model
        vgg.load_state_dict(torch.load(vgg_pretrained))
        vgg = nn.Sequential(*list(vgg.children())[:31])
        vgg_enc_layers = list(vgg.children())
        
        self.vgg_enc_1 = nn.Sequential(*vgg_enc_layers[:3])  # input -> conv1_1
        self.vgg_enc_2 = nn.Sequential(*vgg_enc_layers[3:10])  # conv1_1 -> conv2_1
        self.vgg_enc_3 = nn.Sequential(*vgg_enc_layers[10:17])  # conv2_1 -> conv3_1
        self.vgg_enc_4 = nn.Sequential(*vgg_enc_layers[17:30])  # conv3_1 -> conv4_1

        self.mse_loss = nn.MSELoss()

        for name in ['vgg_enc_1', 'vgg_enc_2', 'vgg_enc_3', 'vgg_enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_vgg_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'vgg_enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
   
    # extract relu3_1 from input image
    def encode_vgg_content(self, input):
        for i in range(3):
            input = getattr(self, 'vgg_enc_{:d}'.format(i + 1))(input)
        return input
        
    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        
        loss = self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

        return loss

    def style_loss(self, style, res_img):
        style_feats_vgg = self.encode_with_vgg_intermediate(style)
        res_feats_vgg = self.encode_with_vgg_intermediate(res_img)

        loss_s = self.calc_style_loss(res_feats_vgg[0], style_feats_vgg[0])
        for i in range(1, 5):
            loss_s = loss_s + self.calc_style_loss(res_feats_vgg[i], style_feats_vgg[i])

        return loss_s
    
    # EFDM loss (style contrastive loss)
    def efdm_single(self, style, trans):
        B, C, W, H = style.size(0), style.size(1), style.size(2), style.size(3)
        
        value_style, index_style = torch.sort(style.view(B, C, -1))
        value_trans, index_trans = torch.sort(trans.view(B, C, -1))
        inverse_index = index_trans.argsort(-1)
        
        return self.mse_loss(trans.view(B, C,-1), value_style.gather(-1, inverse_index))

    # Perceptual loss
    def perceptual_loss(self, content, style, trs_img):
        # normalization for ReLU
        content = content.permute(0, 2, 3, 1)
        style = style.permute(0, 2, 3, 1)
        trs_img = trs_img.permute(0, 2, 3, 1)
        
        content = content * torch.from_numpy(np.array((0.229, 0.224, 0.225))).to(content.device) + torch.from_numpy(np.array((0.485, 0.456, 0.406))).to(content.device)
        style = style * torch.from_numpy(np.array((0.229, 0.224, 0.225))).to(style.device) + torch.from_numpy(np.array((0.485, 0.456, 0.406))).to(style.device)
        trs_img = trs_img * torch.from_numpy(np.array((0.229, 0.224, 0.225))).to(trs_img.device) + torch.from_numpy(np.array((0.485, 0.456, 0.406))).to(trs_img.device)
        
        content = content.permute(0, 3, 1, 2).float()
        style = style.permute(0, 3, 1, 2).float()
        trs_img = trs_img.permute(0, 3, 1, 2).float()
        
        # calculate perceptual loss
        content_feats_vgg = self.encode_vgg_content(content)
        style_feats_vgg = self.encode_with_vgg_intermediate(style)
        trs_feats_vgg = self.encode_with_vgg_intermediate(trs_img)

        loss_c = self.calc_content_loss(trs_feats_vgg[-2], content_feats_vgg)
        loss_s = self.efdm_single(trs_feats_vgg[0], style_feats_vgg[0])
        for i in range(1, 4):
            loss_s = loss_s + self.efdm_single(trs_feats_vgg[i], style_feats_vgg[i])
        
        # calculate which pair in batch has the lowest style contrastive loss
        neg_idx = []
        batch = content.shape[0]
        for a in range(batch):
            neg_s = 100000
            for b in range(batch):
                if a != b:
                    loss_s_single = 0
                    for i in range(0, 4):
                        loss_s_single += self.efdm_single(trs_feats_vgg[i][a].unsqueeze(0), style_feats_vgg[i][b].unsqueeze(0))
                    if loss_s_single < neg_s:
                        neg_s = loss_s_single
                        neg_i = b
            neg_idx.append(b)
        
        loss = loss_c * self.config.lambda_perc_cont + loss_s * self.config.lambda_perc_style
        
        return loss, neg_idx
