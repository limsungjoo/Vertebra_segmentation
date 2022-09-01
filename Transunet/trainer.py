import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from datasets.dataset import VertebralDataset
def trainer_synapse(args, model, snapshot_path):
    
    # logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
    #                     format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    trn_dataset = VertebralDataset(is_Train=True, augmentation=False)
    trainloader = DataLoader(trn_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  pin_memory=True,
                                  worker_init_fn=worker_init_fn)
  
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    index = 1
    for epoch_num in iterator:
        for i_batch, (img,mask) in enumerate(trainloader):
            img, mask = torch.Tensor(img).float(), torch.Tensor(mask).float()
            img, mask = img.cuda(), mask.cuda()
            image_batch, label_batch = img,mask
            
            outputs = model(image_batch)
            print(outputs.shape)
            loss_ce = ce_loss(outputs, label_batch)
            loss_dice = dice_loss(outputs, label_batch)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
            import cv2
            if iter_num % 20 == 0:
                outputs = outputs[:,0,:,:]
                label_batch = label_batch[:,0,:,:]
                y = outputs.sigmoid()
                zeros = torch.zeros(y.size())
                ones = torch.ones(y.size())
                y = y.cpu()

                # make mask
                y = torch.where(y > 0.9, ones, zeros)
                
                for iw in range(len(img.cpu().numpy())):
                    origin = image_batch.cpu().numpy()[iw,:,:] 
                    pred = y.cpu().numpy()[iw,:,:]
                    true = label_batch.cpu().numpy()[iw,:,:]	
                    pred=pred*255
                    true = true*255
                    origin=origin *255

                    origin = origin[0][:,:,np.newaxis]
                    pred = pred[:,:,np.newaxis]
                    true = true[:,:,np.newaxis]
                    cv2.imwrite('./train/Image/'+str(index)+'.jpg',origin)
                    cv2.imwrite('./train/Prediction/'+str(index)+'.png',pred)
                    cv2.imwrite('./train/GroundTruth/'+str(index)+'.png',true)
                    index+=1
                # image = image_batch
                # image = (image - image.min()) / (image.max() - image.min())
                # writer.add_image('train/Image', image, iter_num)
                # outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                # writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                # labs = label_batch[1, ...].unsqueeze(0) * 50
                # writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join('./train/pth/1024/', 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join('./train/pth/1024/', 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"