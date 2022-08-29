import os 
import sys
import time
import numpy as np

import torch
import torch.optim as optim

from utils.losses import DiceLoss, IoULoss

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
      self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


class Stopwatch:
    def __init__(self, title, silance=True):
        self.title = title
        self.silance = silance

    def __enter__(self):
        self.t0 = time.time()
        #logging.debug('{} begin'.format(self.title))

    def __exit__(self, type, value, traceback):
        current_time = time.time()
        if not self.silance:
            print('{} : {}ms'.format(self.title, int((current_time - self.t0) * 1000)))
        self.latency = current_time - self.t0


def get_current_lr(optimizer):
  return optimizer.state_dict()['param_groups'][0]['lr']


def lr_update(epoch, opt, optimizer):
  prev_lr = get_current_lr(optimizer)
  if 0 <= epoch < opt.lr_warmup_epoch:
    mul_rate = 10 ** (1/opt.lr_warmup_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] *= mul_rate
    
    current_lr = get_current_lr(optimizer)
    print("LR warm-up : %.7f to %.7f" % (prev_lr, current_lr))
  
  else:
    if isinstance(opt.lr_decay_epoch, list):
      if (epoch+1) in opt.lr_decay_epoch:
        for param_group in optimizer.param_groups:
          param_group['lr'] = (prev_lr * 0.1)
          print("LR Decay : %.7f to %.7f" % (prev_lr, prev_lr * 0.1))
      
      
def get_optimizer(net, opt):

  if isinstance(net, list):
    optims = []
    for network in net:
      optims.append(get_optimizer(network, opt))
    return optims

  else:
    params_to_update = []
    for name,param in net.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
    # if opt.no_bias_decay:
    #   weight_params = []
    #   bias_params = []
    #   for n, p in net.named_parameters():
    #       if 'bias' in n:
    #           bias_params.append(p)
    #       else:
    #           weight_params.append(p)
    #   parameters = [{'params' : bias_params, 'weight_decay' : 0},
    #                 {'params' : weight_params}]
    # else:
    #   parameters = net.parameters()

    if opt.optim.lower() == 'rmsprop':
      optimizer = optim.RMSprop(params_to_update, lr=opt.lr, momentum=opt.momentum, weight_decay=opt.wd)
    elif opt.optim.lower() == 'sgd':
      optimizer = optim.SGD(params_to_update, lr=opt.lr, momentum=opt.momentum, weight_decay=opt.wd)
    elif opt.optim.lower() == 'adam':
      optimizer = optim.Adam(params_to_update, lr=opt.lr)
  
    return optimizer

def get_loss_function(opt):
    if opt.loss == 'dice':
        loss = DiceLoss(sigmoid_normalization=True)
    elif opt.loss.lower() == 'IoU':
        loss = IoULoss()
    else:
        raise ValueError("Only 'dice' loss is supported now.")
    
    return loss