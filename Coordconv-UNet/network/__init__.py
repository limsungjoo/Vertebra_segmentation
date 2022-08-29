import os
import torch
from network.modified_unet import Modified2DUNet
from network.net import DSS



def create_model(opt):
    # Load network
    # print(Modified2DUNet(1, 1, opt.base_n_filter).ds3_1x1_conv3d)
    net = Modified2DUNet(1, 1, opt.base_n_filter)

    
    
    # net = DSS()

    # GPU settings
    if opt.use_gpu:
        net.cuda()
        if opt.ngpu > 1:
            net = torch.nn.DataParallel(net)
    
    if opt.resume:
        if os.path.isfile('/home/vfuser/sungjoo/coordconv/exp/coordconv_final_final/epoch_0076_iou_0.8482_loss_0.08735962.pth'):
            pretrained_dict = torch.load('/home/vfuser/sungjoo/coordconv/exp/coordconv_final_final/epoch_0076_iou_0.8482_loss_0.08735962.pth', map_location=torch.device('cuda'))
            # model_dict = net.state_dict()

            # match_cnt = 0
            # mismatch_cnt = 0
            # pretrained_dict_matched = dict()
            # for k, v in pretrained_dict.items():
            #     if k in model_dict and v.size() == model_dict[k].size():
            #         pretrained_dict_matched[k] = v
            #         match_cnt += 1
            #     else:
            #         mismatch_cnt += 1
                    
            # model_dict.update(pretrained_dict_matched) 
            
            # Single GPU
            if opt.ngpu == 1:
                net.load_state_dict(pretrained_dict)
            # Multi GPU
            else:
                net.module.load_state_dict(pretrained_dict)

            print("=> Successfully loaded weights from %s " % (opt.resume))

        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    return net
