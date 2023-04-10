import numpy as np
import torch
import pandas as pd
import tensorboardX
import torchsummary
import time
from datetime import datetime
from pytz import timezone
from torchvision.utils import save_image

from Config import Config
from DataSplit import DataSplit
from model import InfusionNet
from blocks import model_save, model_load
import networks
from skimage.metrics import structural_similarity as ssim

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

def mkoutput_dir(config):
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.ckpt_dir):
        os.makedirs(config.ckpt_dir)
    if not os.path.exists(config.img_dir):
        os.makedirs(config.img_dir)
    if not os.path.exists(config.img_dir+'/Train'):
        os.makedirs(config.img_dir+'/Train')
        os.makedirs(config.img_dir+'/Validation')
        os.makedirs(config.img_dir+'/Test')

def get_n_params(model):
        total_params=0
        net_params = {'netE':0, 'netS':0, 'netG':0, 'vgg_loss':0}
        
        for name, param in model.named_parameters():
            net = name.split('.')[0]
            nn=1
            for s in list(param.size()):
                nn = nn*s
            net_params[net] += nn
            total_params += nn
        return total_params, net_params

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
    if "WORLD_SIZE" in os.environ:
        config.world_size = int(os.environ["WORLD_SIZE"])
    elif 'SLURM_NTASKS' in os.environ:
        config.world_size = int(os.environ['SLURM_NTASKS'])
    else:
        config.world_size = torch.cuda.device_count()
        pass
    
    config.distributed = config.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if config.distributed:
        if config.local_rank != -1:
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'RANK' in os.environ:
            config.rank = int(os.environ['RANK'])
            config.gpu = int(os.environ["LOCAL_RANK"])
        elif 'SLURM_PROCID' in os.environ:
            config.rank = int(os.environ['SLURM_PROCID'])
            config.gpu = config.rank % torch.cuda.device_count()
        else:
            config.gpu = [0, 1]
        print('distributed gpus:', config.gpu, " / rank:", config.rank)
        sync_file = _get_sync_file()
        dist.init_process_group(backend=config.dist_backend, init_method=sync_file,
                            world_size=config.world_size, rank=config.rank)
        #dist.init_process_group(backend=config.dist_backend, world_size=config.world_size)
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
    

    
    ########## Data Loader ##########
    train_data = DataSplit(config=config, phase='train')
    valid_data = DataSplit(config=config, phase='valid')

    if config.distributed:
        train_sampler = DistributedSampler(train_data , shuffle=True)
    else:
        train_sampler = RandomSampler(train_data)
       
    data_loader_train = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size,  num_workers=config.num_workers, pin_memory=False, sampler=train_sampler)
    data_loader_valid = torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=False, sampler=None)
    
    print("Train: ", train_data.__len__(), "images: ", len(data_loader_train), "x", config.batch_size,"(batch size) =", train_data.__len__())
    print("Valid: ", valid_data.__len__(), "images: ", len(data_loader_valid), "x", config.batch_size,"(batch size) =", valid_data.__len__())
    
    n_gpus = (train_data.__len__() // len(data_loader_train)) // config.batch_size
    print("# of GPUs: ", n_gpus)

    
    # make log, ckpt, Generated_images output directory
    if (not config.distributed) or config.rank == 0 :
        mkoutput_dir(config)

        
        
    ########## Model load ##########
    print("====================================================================================")
    model = InfusionNet(config)
    
    # # of parameter
    param_num, net_params = get_n_params(model)
    print("# of parameter:", param_num)
    print("parameters of networks:", net_params)
    print("====================================================================================")

    model = init_net(model, 'normal', config.gpu, config)
    print("model init!")
    
    # load saved model
    if config.train_continue == 'on':
        model, model.module.optimizer_E, model.module.optimizer_S, model.module.optimizer_G, epoch_start, tot_itr = model_load(save_type='normal', checkpoint=None, ckpt_dir=config.ckpt_dir, model=model, 
                           optim_E=model.module.optimizer_E,
                           optim_S=model.module.optimizer_S, 
                           optim_G=model.module.optimizer_G)
        print(epoch_start, "th epoch ", tot_itr, "th iteration model load")
    else:
        epoch_start = 0
        tot_itr = 0

    train_writer = tensorboardX.SummaryWriter(config.log_dir)
    
    
    
    ########## Start Training ##########
    print("Start Training!!")
    start_time = datetime.now(timezone('Asia/Seoul'))
    print ('Train start time: %s month, %s day, %s h, %s m and %s s.' % (start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second))
   
    min_mssim = 100000000
    epoch = epoch_start - 1
    tot_itr = tot_itr - 1
    while tot_itr < config.n_iter:
        epoch += 1
        epoch_start = datetime.now(timezone('Asia/Seoul'))
        
        if config.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        for i, data in enumerate(data_loader_train):
            tot_itr += n_gpus
            train_dict = model.module.train_step(data)

            real_A = data['real_A']
            real_B = train_dict['style_img']
            fake_B = train_dict['fake_AtoB']
            trs_high = train_dict['fake_AtoB_high']
            trs_low = train_dict['fake_AtoB_low']

            if (not config.distributed) or config.rank == 0 :
                real_A = im_convert(real_A)
                real_B = im_convert(real_B)
                fake_B = im_convert(fake_B)
                trs_low = im_convert(trs_low)
                trs_high = im_convert(trs_high)
                
                if i % 20 == 0:
                    save_image(torch.Tensor(real_A.transpose(0, 3, 1, 2))[0], '{}/Train/{}_{}_A.png'.format(config.img_dir, epoch+1, i+1))
                    save_image(torch.Tensor(real_B.transpose(0, 3, 1, 2))[0], '{}/Train/{}_{}_B.png'.format(config.img_dir, epoch+1, i+1))
                    save_image(torch.Tensor(fake_B.transpose(0, 3, 1, 2))[0], '{}/Train/{}_{}_fake.png'.format(config.img_dir, epoch+1, i+1))
    
                ## Tensorboard ##
                # tensorboard - loss
                train_writer.add_scalar('Loss_G', train_dict['G_loss'], tot_itr)
                train_writer.add_scalar('Loss_G_Percept', train_dict['G_Percept'], tot_itr)
                train_writer.add_scalar('Loss_G_Const_Style', train_dict['G_Const_Style'], tot_itr)

                # tensorboard - images
                train_writer.add_image('Content_Image_A', real_A, tot_itr, dataformats='NHWC')
                train_writer.add_image('Style_Image_B', real_B, tot_itr, dataformats='NHWC')
                train_writer.add_image('Generated_Image_AtoB', fake_B, tot_itr, dataformats='NHWC')
                train_writer.add_image('Translation_AtoB_high', trs_high, tot_itr, dataformats='NHWC')
                train_writer.add_image('Translation_AtoB_low', trs_low, tot_itr, dataformats='NHWC')
            
                print("Tot_itrs: %d/%d | Epoch: %d | itr: %d/%d | Loss_G: %.5f"%(tot_itr+1, config.n_iter, epoch+1, (i+1)*n_gpus, len(data_loader_train)*n_gpus, train_dict['G_loss']))
                
                if (tot_itr + 1) % 10000 == 0:
                    model_save(save_type='normal', ckpt_dir=config.ckpt_dir, model=model, optim_E=model.module.optimizer_E, optim_S=model.module.optimizer_S, optim_G=model.module.optimizer_G, epoch=epoch, itr=tot_itr, ssim=None)
                    print(tot_itr+1, "th iteration model save")

        networks.update_learning_rate(model.module.E_scheduler, model.module.optimizer_E)
        networks.update_learning_rate(model.module.S_scheduler, model.module.optimizer_S)
        networks.update_learning_rate(model.module.G_scheduler, model.module.optimizer_G)

        
        if (not config.distributed) or config.rank == 0 :
            ## Model Save
            model_save(save_type='normal', ckpt_dir=config.ckpt_dir, model=model, optim_E=model.module.optimizer_E, optim_S=model.module.optimizer_S, optim_G=model.module.optimizer_G, epoch=epoch, itr=tot_itr, ssim=None)
            print(epoch+1, "th model save")
        
            # Validation
            with torch.no_grad():
                
                t_ssim = 0
                for v, v_data in enumerate(data_loader_valid):
                    val_dict = model.module.val(v_data)
                    
                    v_real_A_r = v_data['real_A']
                    v_real_B_r = v_data['style_img']
                    v_fake_B_r = val_dict['fake_AtoB']

                    # save image
                    # save_image(v_real_A[0], '{}/Validation/{}_{}_A.png'.format(config.img_dir, epoch+1, v+1))
                    # save_image(v_real_B[0], '{}/Validation/{}_{}_B.png'.format(config.img_dir, epoch+1, v+1))
                    # save_image(v_fake_B[0], '{}/Validation/{}_{}_fake.png'.format(config.img_dir, epoch+1, v+1))
                    
                    # post-processing
                    v_real_A = im_convert(v_real_A_r)
                    v_real_B = im_convert(v_real_B_r)
                    v_fake_B = im_convert(v_fake_B_r)

                    # SSIM
                    t_ssim += ssim(torch.Tensor(v_real_A.transpose(0, 3, 1, 2))[0], torch.Tensor(v_fake_B.transpose(0, 3, 1, 2))[0])

                mssim = float(t_ssim / (v+1))
                print("===> Validation <=== Epoch: %d | MSSIM: %.5f"%(epoch+1, mssim))

            # save best performance model
            if mssim < min_mssim:
                min_mssim = mssim
                model_save(save_type='best', ckpt_dir=config.ckpt_dir, model=model, optim_E=model.module.optimizer_E, optim_S=model.module.optimizer_S, optim_G=model.module.optimizer_G, epoch=epoch, itr=tot_itr, ssim=mssim)
                print("best model save")

        end_time = datetime.now(timezone('Asia/Seoul'))
        print ('Epoch start time: %s month, %s day, %s h, %s m and %s s.' % (epoch_start.month, epoch_start.day, epoch_start.hour, epoch_start.minute, epoch_start.second))
        print ('Epoch finish time: %s month, %s day, %s h, %s m and %s s.' % (end_time.month, end_time.day, end_time.hour, end_time.minute, end_time.second))

        
if __name__ == '__main__':
    main()
