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

from Config import Config
from DataSplit import DataSplit
from model import InfusionNet
from blocks import model_save, model_load
import networks
from vgg19_test import vgg, VGG_loss

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

def main():
    config = Config()
    
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

    ## Data Loader
    test_data = DataSplit(config=config, phase='test')
    data_loader_test = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=16, pin_memory=False)
    print("Test: ", test_data.__len__(), "images: ", len(data_loader_test), "x", 1,"(batch size) =", test_data.__len__())

    ## Model load
    model = InfusionNet(config)
    model = init_net(model, 'normal', config.gpu, config)

    # load saved model
    ckpt = 'model_iter_160000_epoch_22.pth'
    model, model.module.optimizer_E, model.module.optimizer_S, model.module.optimizer_G, epoch_start, tot_itr = model_load(save_type='normal', checkpoint=ckpt, ckpt_dir=config.ckpt_dir, model=model,
                           optim_E=model.module.optimizer_E,
                           optim_S=model.module.optimizer_S,
                           optim_G=model.module.optimizer_G)
    
    print(epoch_start, "th epoch ", tot_itr, "th iteration model load")
        
    if not os.path.exists(config.img_dir+'/Test'):
        os.makedirs(config.img_dir+'/Test')

    vgg_loss = VGG_loss(config, vgg)

    ## Start Testing
    print("Start Testing!!")
    start_time = datetime.now(timezone('Asia/Seoul'))
    print ('Test start time: %s month, %s day, %s h, %s m and %s s.' % (start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second))
    
    t_during = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader_test):
            start = time.time()
            test_dict = model.module.test(data)

            real_A_r = data['real_A']
            real_B_r = test_dict['style_img']
            trs_AtoB_r = test_dict['fake_AtoB']
            trs_AtoB_low_r = test_dict['fake_AtoB_low']
            trs_AtoB_high_r = test_dict['fake_AtoB_high']
          
            real_A = im_convert(real_A_r)
            real_B = im_convert(real_B_r)
            trs_AtoB = im_convert(trs_AtoB_r)
            trs_AtoB_low = im_convert(trs_AtoB_low_r)
            trs_AtoB_high = im_convert(trs_AtoB_high_r)

            ## Generated Image Save
            for j in range(real_A.shape[0]):
                A_image = Image.fromarray((real_A[j] * 255.0).astype(np.uint8))
                B_image = Image.fromarray((real_B[j] * 255.0).astype(np.uint8))
                trs_image = Image.fromarray((trs_AtoB[j] * 255.0).astype(np.uint8))    
                trsl_image = Image.fromarray((trs_AtoB_low[j] * 255.0).astype(np.uint8))    
                trsh_image = Image.fromarray((trs_AtoB_high[j] * 255.0).astype(np.uint8))    
                
                # save
                A_image.save('{}/Test/{}_{}_A.png'.format(config.img_dir, i+1, j+1))
                B_image.save('{}/Test/{}_{}_B.png'.format(config.img_dir, i+1, j+1))
                trs_image.save('{}/Test/{}_{}_trs.png'.format(config.img_dir, i+1, j+1))
                trsl_image.save('{}/Test/{}_{}_trs_low.png'.format(config.img_dir, i+1, j+1))
                trsh_image.save('{}/Test/{}_{}_trs_high.png'.format(config.img_dir, i+1, j+1))

            end = time.time()
            t_during += end - start

    t_during = float(t_during / test_data.__len__())
    print("Total images:", test_data.__len__(), "Avg Testing time:", t_during)
            
    end_time = datetime.now(timezone('Asia/Seoul'))
    print ('Test start time: %s month, %s day, %s hours, %s minutes and %s seconds.' % (start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second))
    print ('Test finish time: %s month, %s day, %s hours, %s minutes and %s seconds.' % (end_time.month, end_time.day, end_time.hour, end_time.minute, end_time.second))

if __name__ == '__main__':
    main()
