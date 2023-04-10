class Config:
    phase = 'train'
    train_continue = 'on'  # on / off
    
    if phase == 'style_mixing_single':
        content_img = '/pscratch/sd/s/sooyoung/COCO/test/000000332649.jpg'
        style_high_img = '/pscratch/sd/s/sooyoung/WikiArt/test/37659.jpg'
        style_low_img = '/pscratch/sd/s/sooyoung/WikiArt/test/76732.jpg'
        
    data_num = 60000       # Maximum # of training data
    data_num_test = 500
    data_num_mixing = 200

    data_dir = '/pscratch/sd/s/sooyoung/COCO'
    style_dir = '/pscratch/sd/s/sooyoung/WikiArt'
        
    # output directory
    file_n = 'coco_wiki_EFDM_vggNstyle_Norm_sty15:15'
    
    log_dir = '/pscratch/sd/s/sooyoung/Results_0203/log/' + file_n
    ckpt_dir = '/pscratch/sd/s/sooyoung/Results_0203/ckpt/' + file_n
    img_dir = '/pscratch/sd/s/sooyoung/Results_0203/Generated_images/' + file_n
    
    
    # VGG pre-trained model
    vgg_model = './vgg_normalised.pth'

    ## basic parameters
    n_iter = 160000
    batch_size = 8
    lr = 0.0001
    lr_policy = 'step'
    lr_decay_iters = 50
    beta1 = 0.0

    # preprocess parameters
    test_load_size = 256
    load_size = 512
    crop_size = 256

    # model parameters
    input_nc = 3         # of input image channel
    nf = 64              # of feature map channel after Encoder first layer
    output_nc = 3        # of output image channel
    
    # Octave Convolution parameters
    alpha_in = 0.5
    alpha_out = 0.5
    freq_ratio = [1,1]  # [low, high] ratio

    # Loss ratio
    lambda_percept = 1.0
    lambda_perc_cont = 1.0
    lambda_perc_style = 15.0
    lambda_const_style = 15.0

    # Else
    norm = 'instance'     # [instance | batch | none]
    init_type = 'normal'  # [normal | xavier | kaiming | orthogonal]
    init_gain = 0.02      # scaling factor for normal, xavier and orthogonal
    no_dropout = 'store_true'   # no dropout for generator

    # DDP configs:
    num_workers = 4
    world_size = -1
    rank = -1
    dist_backend = 'nccl'
    local_rank =-1
