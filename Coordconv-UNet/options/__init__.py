import os
import argparse
import torch

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_option(print_option=True):    
    p = argparse.ArgumentParser(description='')

    # Data Directory
    p.add_argument('--data_root', default= "/home/vfuser/sungjoo/data/new_dataset/mask/*", type=str, help='root directory of dataset files.')
    
    # Data augmentation
    p.add_argument('--rot_factor', default=30, type=float)
    p.add_argument('--scale_factor', default=0.15, type=float)
    p.add_argument('--flip', default='True', type=str2bool)
    p.add_argument('--trans_factor', default=0.1, type=float)

    # Input image
    p.add_argument('--crop_size', default=200, type=float, help='Center crop width')
    p.add_argument('--input_size', default=224, type=int, help='input resolution using resize process')
    p.add_argument('--w_min', default=-100., type=float, help='Min value of HU Windowing')
    p.add_argument('--w_max', default=300., type=float, help='Max value of HU Windowing')

    # Network
    p.add_argument('--base_n_filter', default=32, type=int)

    # Optimizer
    p.add_argument('--optim', default='Adam', type=str, help='RMSprop | SGD | Adam')
    p.add_argument('--lr', default=2e-5, type=float)
    p.add_argument('--lr_decay_epoch', default='150', type=str, help="decay epochs with comma (ex - '20,40,60')")
    p.add_argument('--lr_warmup_epoch', default=0, type=int)
    p.add_argument('--momentum', default=0.99, type=float, help='momentum')
    p.add_argument('--wd', default=1e-4, type=float, help='weight decay')
    p.add_argument('--no_bias_decay', default='True', type=str2bool, help='weight decay for bias')

    # Hyper-parameter
    p.add_argument('--batch_size', default=1, type=int, help='use 1 batch size in 3D training.')
    p.add_argument('--start_epoch', default=0, type=int)
    p.add_argument('--max_epoch', default=100, type=int)
    p.add_argument('--threshold', default=0.9, type=float)

    # Loss function
    p.add_argument('--loss', default='dice', type=str)
    p.add_argument('--iou_smooth', default=1e-6, type=float, help='avoid 0/0')

    # Resume trained network
    p.add_argument('--resume', default='coordconv', type=str, help="pth file path to resume")

    # Resource option
    p.add_argument('--workers', default=10, type=int, help='#data-loading worker-processes')
    p.add_argument('--use_gpu', default="True", type=str2bool, help='use gpu or not (cpu only)')
    p.add_argument('--gpu_id', default="0,1,2,3", type=str)

    # Output directory
    p.add_argument('--exp', default='/home/vfuser/sungjoo/Resize_model/exp/coordconv_final_final', type=str, help='checkpoint dir.')
    p.add_argument('--save_dir', default='plots_prob', type=str, help='evaluation plot directory')
    
    p.add_argument('--patch', default='True', type=str2bool, help='patch analysis?')


    opt = p.parse_args()
    
    # Make output directory
    if not os.path.exists(opt.exp):
        os.makedirs(opt.exp)

    # GPU Setting
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu_id

    if opt.use_gpu:
        opt.ngpu = len(opt.gpu_id.split(","))
    else:
        opt.gpu_id = 'cpu'
        opt.ngpu = 'cpu'

    # lr decay setting
    if ',' in opt.lr_decay_epoch:
        opt.lr_decay_epoch = opt.lr_decay_epoch.split(',')
        opt.lr_decay_epoch = [int(epoch) for epoch in opt.lr_decay_epoch]

    if print_option:
        print("\n==================================== Options ====================================\n")
    
        print('   Data root : %s' % (opt.data_root))
        print()
        print('   Data Crop size : Crop to (%d,%d)' % (opt.crop_size,opt.crop_size))
        print('   Data input size : Resized to (%d,%d)' % (opt.input_size,opt.input_size))
        print()
        print('   Base #Filters of Network : %d' % (opt.base_n_filter))
        print()
        print('   Optimizer : %s (weight decay %f)' % (opt.optim, opt.wd))
        print('   Loss function : %s' % opt.loss)
        print('   Batch size : %d' % opt.batch_size)
        print('   Max epoch : %d' % opt.max_epoch)
        print('   Learning rate : %s (linear warm-up until %s / decay at %s)' % (opt.lr, opt.lr_warmup_epoch, opt.lr_decay_epoch))
        print()
        print('   Resume pre-trained weights path : %s' % opt.resume)
        print('   Output dir : %s' % opt.exp)
        print()
        print('   GPU ID : %s' % opt.gpu_id)
        print('   #Workers : %s' % opt.workers)
        print('   pytorch version: %s (CUDA : %s)' % (torch.__version__, torch.cuda.is_available()))
        print("\n=================================================================================\n")

    return opt
