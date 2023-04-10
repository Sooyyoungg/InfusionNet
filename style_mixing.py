import random
import torch
import pandas as pd
import tensorboardX
import cv2
import imageio
import numpy as np

from sklearn.metrics import mean_squared_error
import torchsummary
import time
from datetime import datetime
from matplotlib import pyplot as plt
from pytz import timezone
from PIL import Image
from torchvision import transforms

from Config import Config
from DataSplit import DataSplit
from model import InfusionNet
from blocks import model_save, model_load
import networks
from metrics import ssim

#DDP
import os
import builtins
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

def _get_sync_file():
        """Logic for naming sync file using slurm env variables"""
        if 'SCRATCH' in os.environ:
            sync_file_dir = '%s/pytorch-sync-files' % os.environ['SCRATCH'] # Perlmutter
        else:
            raise Exception('there is no env variable SCRATCH. Please check sync_file dir')
        os.makedirs(sync_file_dir, exist_ok=True)

        #temporally add two lines below for torchrun
        if ('SLURM_JOB_ID' in os.environ) and ('SLURM_STEP_ID' in os.environ) :
            sync_file = 'file://%s/pytorch_sync.%s.%s' % (                                                          
                sync_file_dir, os.environ['SLURM_JOB_ID'], os.environ['SLURM_STEP_ID'])
        else:
            if 'SYNC_CODE' in os.environ:
                sync_file = 'file://%s/pytorch_sync.%s.%s' % (
                sync_file_dir, os.environ['SYNC_CODE'], os.environ['SYNC_CODE'])
            else:
                sync_file = 'file://%s/pytorch_sync.%s.%s' % (
                sync_file_dir, 12345, 12345) 
        return sync_file
def init_net(net, init_type='normal', gpu_ids=[], config=None):
    if config.distributed:
        if config.gpu is not None:
                config.device = torch.device('cuda:{}'.format(config.gpu))
                torch.cuda.set_device(config.gpu)
                net.cuda(config.gpu)
                
                net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[config.gpu], broadcast_buffers=False, find_unused_parameters=True) 
                net_without_ddp = net.module
        else:
            config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net.cuda()

            net = torch.nn.parallel.DistributedDataParallel(net) 
            net_without_ddp = net.module
    else:
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = torch.nn.DataParallel(net).to(config.device)  
        
    return net

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach().numpy()
    image = image.transpose(0, 2, 3, 1)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image

def transform_single(load_size, crop_size):
    transform_list = []
    osize = [load_size, load_size]
    method = transforms.InterpolationMode.BICUBIC
    
    transform_list.append(transforms.Resize(osize, method))
    transform_list.append(transforms.RandomCrop(crop_size))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    transform = transforms.Compose(transform_list)
    return transform
            
def main():
    config = Config()
    if not os.path.exists(config.img_dir+'/Mixing'):
        os.makedirs(config.img_dir+'/Mixing')
    
    ## DDP
    # sbatch script에서 WORLD_SIZE를 지정해준 경우 (노드 당 gpu * 노드의 수)
    if "WORLD_SIZE" in os.environ:  # for torchrun
        config.world_size = int(os.environ["WORLD_SIZE"])
    # 혹은 슬럼에서 자동으로 ntasks per node * nodes 로 구해줌
    elif 'SLURM_NTASKS' in os.environ:
        config.world_size = int(os.environ['SLURM_NTASKS'])
    else:
        pass # torch.distributed.launch

    config.distributed = config.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if config.distributed:
        if config.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'RANK' in os.environ: # for torchrun
            config.rank = int(os.environ['RANK'])
            config.gpu = int(os.environ["LOCAL_RANK"])
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            config.rank = int(os.environ['SLURM_PROCID'])
            config.gpu = config.rank % torch.cuda.device_count()
        print('distributed gpus:', config.gpu)
        sync_file = _get_sync_file()
        dist.init_process_group(backend=config.dist_backend, init_method=sync_file,
                            world_size=config.world_size, rank=config.rank)
    else:
        config.rank = 0
        config.gpu = 0

    # suppress printing if not on master gpu
    if config.rank!=0:
        def print_pass(*config):
            pass
        builtins.print = print_pass

    print('cuda:', config.gpu)
    print('Version:', config.file_n)

    
    ## Model load
    model = InfusionNet(config)
    model = init_net(model, 'normal', config.gpu, config)
    
    # load saved model
    model, model.module.optimizer_E, model.module.optimizer_S, model.module.optimizer_G, epoch_start, tot_itr = model_load(save_type='normal', checkpoint=None, ckpt_dir=config.ckpt_dir, model=model, 
                           optim_E=model.module.optimizer_E,
                           optim_S=model.module.optimizer_S, 
                           optim_G=model.module.optimizer_G)
    print(epoch_start, "th epoch ", tot_itr, "th iteration model load")
    
    
    ## Start Mixing
    start_time = datetime.now(timezone('Asia/Seoul'))
    print("===============================================================================================================")
    
    if config.phase == 'style_mixing_single':
        ssim_sc = 0
        with torch.no_grad():
            real_A = Image.open(config.content_img).convert('RGB')
            style_high = Image.open(config.style_high_img).convert('RGB')
            style_low = Image.open(config.style_low_img).convert('RGB')

            transform = transform_single(config.load_size, config.crop_size)
            real_A = transform(real_A).unsqueeze(0)
            style_high = transform(style_high).unsqueeze(0)
            style_low = transform(style_low).unsqueeze(0)
            
            data = {'real_A': real_A, 'style_high': style_high, 'style_low': style_low}
            mix_dict = model.module.style_mixing(data)
            
            real_B_h = mix_dict['style_high_img']
            real_B_l = mix_dict['style_low_img']
            trs_AtoB = mix_dict['fake_AtoB']
            trs_AtoB_high = mix_dict['fake_AtoB_high']
            trs_AtoB_low = mix_dict['fake_AtoB_low']
            
            # Image Save
            real_A = im_convert(real_A)
            real_B_h = im_convert(real_B_h)
            real_B_l = im_convert(real_B_l)
            trs_AtoB = im_convert(trs_AtoB)
            trs_AtoB_high = im_convert(trs_AtoB_high)
            trs_AtoB_low = im_convert(trs_AtoB_low)
            
            A_image = Image.fromarray((real_A[0] * 255.0).astype(np.uint8))
            B_h_image = Image.fromarray((real_B_h[0] * 255.0).astype(np.uint8))
            B_l_image = Image.fromarray((real_B_l[0] * 255.0).astype(np.uint8))
            trs_image = Image.fromarray((trs_AtoB[0] * 255.0).astype(np.uint8))
            trsl_image = Image.fromarray((trs_AtoB_high[0] * 255.0).astype(np.uint8))
            trsh_image = Image.fromarray((trs_AtoB_low[0] * 255.0).astype(np.uint8))

            A_image.save('{}/Mixing/Single_A.png'.format(config.img_dir))
            B_h_image.save('{}/Mixing/Single_B_high.png'.format(config.img_dir))
            B_l_image.save('{}/Mixing/Single_B_low.png'.format(config.img_dir))
            trs_image.save('{}/Mixing/Single_trs.png'.format(config.img_dir))
            trsl_image.save('{}/Mixing/Single_trs_low.png'.format(config.img_dir))
            trsh_image.save('{}/Mixing/Single_trs_high.png'.format(config.img_dir))

            
    elif config.phase == 'style_mixing_multi':
        ## Data Loader
        test_data = DataSplit(config=config, phase='style_mixing_multi')
        data_loader_test = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=False)
        print("Mixing data: ", test_data.__len__(), "images: ", len(data_loader_test), "x", config.batch_size,"(batch size) =", test_data.__len__())

        mssim = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader_test):
                mix_dict = model.module.style_mixing(data)

                real_A = data['real_A']
                real_B_h = mix_dict['style_high_img']
                real_B_l = mix_dict['style_low_img']
                trs_AtoB = mix_dict['fake_AtoB']
                trs_AtoB_low = mix_dict['fake_AtoB_low']
                trs_AtoB_high = mix_dict['fake_AtoB_high']

                real_A = im_convert(real_A)
                real_B_h = im_convert(real_B_h)
                real_B_l = im_convert(real_B_l)
                trs_AtoB = im_convert(trs_AtoB)
                trs_AtoB_high = im_convert(trs_AtoB_high)
                trs_AtoB_low = im_convert(trs_AtoB_low)

                ## Generated Image Save
                for j in range(real_A.shape[0]):
                    print(i, '-', j, "th image save")
                    A_image = Image.fromarray((real_A[j] * 255.0).astype(np.uint8))
                    B_h_image = Image.fromarray((real_B_h[j] * 255.0).astype(np.uint8))
                    B_l_image = Image.fromarray((real_B_l[j] * 255.0).astype(np.uint8))
                    trs_image = Image.fromarray((trs_AtoB[j] * 255.0).astype(np.uint8))
                    trsl_image = Image.fromarray((trs_AtoB_high[j] * 255.0).astype(np.uint8))
                    trsh_image = Image.fromarray((trs_AtoB_low[j] * 255.0).astype(np.uint8))

                    # save
                    A_image.save('{}/Mixing/{}_{}_A.png'.format(config.img_dir, i+1, j+1))
                    B_h_image.save('{}/Mixing/{}_{}_B_high.png'.format(config.img_dir, i+1, j+1))
                    B_l_image.save('{}/Mixing/{}_{}_B_low.png'.format(config.img_dir, i+1, j+1))
                    trs_image.save('{}/Mixing/{}_{}_trs.png'.format(config.img_dir, i+1, j+1))
                    trsl_image.save('{}/Mixing/{}_{}_trs_low.png'.format(config.img_dir, i+1, j+1))
                    trsh_image.save('{}/Mixing/{}_{}_trs_high.png'.format(config.img_dir, i+1, j+1))
                    

        mssim = float(mssim / test_data.__len__())
        print("Total images:", test_data.__len__(), "Total avg SSIM:", mssim)

    end_time = datetime.now(timezone('Asia/Seoul'))
    print ('Test start time: %s month, %s day, %s hours, %s minutes and %s seconds.' % (start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second))
    print ('Test finish time: %s month, %s day, %s hours, %s minutes and %s seconds.' % (end_time.month, end_time.day, end_time.hour, end_time.minute, end_time.second))

if __name__ == '__main__':
    main()
