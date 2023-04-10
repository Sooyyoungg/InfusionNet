from path import Path
import glob
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop, Normalize, RandomCrop
import random

Image.MAX_IMAGE_PIXELS = 1000000000

class DataSplit(nn.Module):
    def __init__(self, config, phase='train', do_transform=True):
        super(DataSplit, self).__init__()

        self.do_transform = do_transform

        transform_list = []
        if phase == 'test':
            osize = config.test_load_size
            crop = CenterCrop(size=(config.crop_size, config.crop_size))
        else:
            osize = config.load_size
            crop = RandomCrop(size=(config.crop_size, config.crop_size))
        
        self.transform = Compose([Resize(size=[osize, osize]),  # Resize to keep aspect ratio
                                crop,  # Center crop to square
                                ToTensor(),
                                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self.config = config
        self.phase = phase
        data_num = config.data_num
        img_dir = Path(self.config.data_dir)
        file_type = '*.jpg'

        ################ Content images ################
        if phase == 'train':
            img_dir = Path(self.config.data_dir+'/train')
        elif phase == 'valid':
            img_dir = Path(self.config.data_dir+'/val')
        elif phase == 'test':
            data_num = config.data_num_test
            img_dir = Path(self.config.data_dir+'/test')
        elif phase == 'style_mixing':
            img_dir = Path(self.config.data_dir)
        print(img_dir)
        self.images = sorted(img_dir.glob(file_type))
        
        if data_num < len(self.images):
            if phase == 'test':
                self.images = self.images[:data_num]
            else:
                self.images = random.sample(self.images, data_num)

        ################ Style images ################
        if phase == 'train':
            sty_dir = Path(self.config.style_dir+'/train')
        elif phase == 'valid':
            sty_dir = Path(self.config.style_dir+'/val')
        elif phase == 'test':
            sty_dir = Path(self.config.style_dir+'/500')
        elif phase == 'style_mixing':
            sty_dir = Path(self.config.style_dir)
        self.style_images = sorted(sty_dir.glob(file_type))

        if len(self.images) < len(self.style_images):
            if phase == 'test':
                self.style_images = self.style_images[:len(self.images)]
            else:
                self.style_images = random.sample(self.style_images, len(self.images))
        elif len(self.images) > len(self.style_images):
            ratio = len(self.images) // len(self.style_images)
            bias = len(self.images) - ratio * len(self.style_images)
            self.style_images = self.style_images * ratio
            self.style_images += random.sample(self.style_images, bias)
        assert len(self.images) == len(self.style_images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        img = Image.open(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        sty_img = self.style_images[index]
        sty_img = Image.open(sty_img).convert('RGB')

        if self.phase == 'style_mixing':
            n = random.randint(0, len(self.images))
            while n == index:
                n = random.randint(0, len(self.images))
            sty_img2 = self.style_images[index + n]
            sty_img2 = Image.open(sty_img2).convert('RGB')

            if self.transform is not None:
                style_img = self.transform(sty_img)
                style_img2 = self.transform(sty_img2)

            return {'real_A': img, 'style_high': style_img, 'style_low': style_img2}
        
        else:
            if self.transform is not None:
                style_img = self.transform(sty_img)

            return {'real_A': img, 'style_img': style_img}
        
        

