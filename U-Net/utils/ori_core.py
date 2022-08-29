import os
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
import time
import datetime
from visdom import Visdom
import torch
from torch.autograd import Variable
import sys
from utils import AverageMeter
from utils.metrics import DiceCoef
from utils.losses import iou_modified, avg_precision
from utils.psave import *
import cv2
from matplotlib import pyplot as plt 

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

class Logger():
    def __init__(self, n_epochs, batches_epoch):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}

    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].data
            else:
                self.losses[loss_name] += losses[loss_name].data

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title':image_name})
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=torch.cuda.FloatTensor([self.epoch]), Y=torch.cuda.FloatTensor([loss/self.batch]), 
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=torch.cuda.FloatTensor([self.epoch]), Y=torch.cuda.FloatTensor([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1


def train(net, dataset_trn, optimizer, criterion, epoch, opt,train_writer):
    print("Start Training...")
    net.train()

    losses, total_dices, total_iou = AverageMeter(), AverageMeter(), AverageMeter()
    logger = Logger(epoch, len(dataset_trn))
    for it, (img, mask,iindex) in enumerate(dataset_trn):
        
        # Optimizer
        optimizer.zero_grad()

        # Load Data
        img, mask = torch.Tensor(img).float(), torch.Tensor(mask).float()
        if opt.use_gpu:
            img, mask = img.cuda(non_blocking=True), mask.cuda(non_blocking=True)

        # Predict
        pred = net(img)

        # Loss Calculation
        loss = criterion(pred, mask)

        pred = pred.sigmoid()
        # Backward and step
        loss.backward()
        optimizer.step()

        logger.log({'loss:':loss},images = {'img:':img,'pred:':pred})
        # Calculation Dice Coef Score
        dice = DiceCoef(return_score_per_channel=False)(pred, mask)
        total_dices.update(dice.item(), img.size(0))
        
        # Convert to Binary
        zeros = torch.zeros(pred.size())
        ones = torch.ones(pred.size())
        pred = pred.cpu()

        pred = torch.where(pred > 0.5, ones, zeros).cuda() # threshold 0.99

        # Calculation IoU Score
        iou_score = iou_modified(pred, mask,opt)

        total_iou.update(iou_score.mean().item(), img.size(0))

        # Stack Results
        losses.update(loss.item(), img.size(0))

        if (it==0) or (it+1) % 10 == 0:
            print('Epoch[%3d/%3d] | Iter[%3d/%3d] | Loss %.4f | Dice %.4f | Iou %.4f'
                % (epoch+1, opt.max_epoch, it+1, len(dataset_trn), losses.avg, total_dices.avg, total_iou.avg))

    print(">>> Epoch[%3d/%3d] | Training Loss : %.4f | Dice %.4f | Iou %.4f\n "
        % (epoch+1, opt.max_epoch, losses.avg, total_dices.avg, total_iou.avg))

    train_writer.add_scalar("train/loss", losses.avg, epoch+1)
    train_writer.add_scalar("train/dice", total_dices.avg, epoch+1)
    train_writer.add_scalar("train/IoU", total_iou.avg, epoch+1)


def validate(dataset_val, net, criterion, epoch, opt, best_iou, best_epoch,train_writer):
    print("Start Evaluation...")
    net.eval()

    # Result containers
    losses, total_dices, total_iou = AverageMeter(), AverageMeter(), AverageMeter()

    for it, (img, mask, iindex) in enumerate(dataset_val):
        # Load Data
        img, mask = torch.Tensor(img).float(), torch.Tensor(mask).float()
        if opt.use_gpu:
            img, mask = img.cuda(non_blocking=True), mask.cuda(non_blocking=True)

        # Predict
        pred = net(img)

        # Loss Calculation
        loss = criterion(pred, mask)

        pred = pred.sigmoid()

        # Calculation Dice Coef Score
        dice = DiceCoef(return_score_per_channel=False)(pred, mask)
        total_dices.update(dice.item(), img.size(0))
        
        # Convert to Binary
        zeros = torch.zeros(pred.size())
        ones = torch.ones(pred.size())
        pred = pred.cpu()

        pred = torch.where(pred > 0.5, ones, zeros).cuda()
        
        # Calculation IoU Score
        iou_score = iou_modified(pred, mask,opt)

        total_iou.update(iou_score.mean().item(), img.size(0))

        # Stack Results
        losses.update(loss.item(), img.size(0))

        # if (it==0) or (it+1) % 10 == 0:
        #     print('Epoch[%3d/%3d] | Iter[%3d/%3d] | Loss %.4f | Dice %.4f | Iou %.4f'
        #         % (epoch+1, opt.max_epoch, it+1, len(dataset_trn), losses.avg, total_dices.avg, total_iou.avg))

    print(">>> Epoch[%3d/%3d] | Test Loss : %.4f | Dice %.4f | Iou %.4f"
        % (epoch+1, opt.max_epoch, losses.avg, total_dices.avg, total_iou.avg))

    train_writer.add_scalar("valid/loss", losses.avg, epoch+1)
    train_writer.add_scalar("valid/dice", total_dices.avg, epoch+1)
    train_writer.add_scalar("valid/IoU", total_iou.avg, epoch+1)

    # Update Result
    if total_iou.avg > best_iou:
        print('Best Score Updated...')
        best_iou = total_iou.avg
        best_epoch = epoch

        # # Remove previous weights pth files
        # for path in glob('%s/*.pth' % opt.exp):
        #     os.remove(path)

        model_filename = '%s/epoch_%04d_iou_%.4f_loss_%.8f.pth' % (opt.exp, epoch+1, best_iou, losses.avg)

        # Single GPU
        if opt.ngpu == 1:
            torch.save(net.state_dict(), model_filename)
        # Multi GPU
        else:
            torch.save(net.module.state_dict(), model_filename)

    print('>>> Current best: IoU: %.8f in %3d epoch\n' % (best_iou, best_epoch+1))
    
    return best_iou, best_epoch


def evaluate(dataset_val, net, opt):
    print("Start Evaluation...")
    net.eval()
    img_dict={}
    iou_scores = []
    for cnt,mskID in tqdm(enumerate(sorted(glob(opt.data_root)))):
        msk = cv2.imread(mskID,0)         
        msk = np.zeros(msk.shape)
        h = msk.shape[0]
        w = msk.shape[1]
        for idx, (img, mask,iindex) in enumerate(dataset_val):
            # Load Data
            img = torch.Tensor(img).float()
            if opt.use_gpu:
                img = img.cuda(non_blocking=True)
            print(iindex)
            # Predict
            with torch.no_grad():
                pred = net(img)
                
                y = pred.sigmoid()
                dice = DiceCoef(return_score_per_channel=False)(y, mask.cuda())
                pred = y.cpu().numpy()[0,0,:,:]
                # pred = pred *255

                img_dict[iindex]=pred
            # # Convert to Binary
            # zeros = torch.zeros(y.size())
            # ones = torch.ones(y.size())
            # y = y.cpu()

            # make mask
            # y = torch.where(y > 0.5, ones, zeros) # threshold 0.99

            # y = Variable(y).cuda()

    #         # iou_score = iou_modified(y, mask.cuda(),opt)

        k = list(img_dict.keys())[0][0].split('_')[0]
   

        for i,height in enumerate(range(0,h,200)):
            for j,width in enumerate(range(0,w,200)):  
                try:
                # new_patch = np.moveaxis(mask_list[ind], 0, -1)
                    print(img_dict[(str(k)+'_'+str(i)+'_'+str(j),)])
                    new_patch = cv2.resize(img_dict[(str(k)+'_'+str(i)+'_'+str(j),)], (200,200))
                    
                    if height + 200 >= h:
                        if width + 200 >= w:
                            msk[h-200:h,w-200:w] = new_patch
                        else:
                            msk[h-200:h,width:width+200] = new_patch
                    else:
                        if width +200 >= w:
                            msk[height:height+200,w-200:w] = new_patch                     
                        else:
                            msk[height:height+200,width:width+200] = new_patch
                    

                except:
                        print('error')
        msk = ((msk - msk.min()) / (msk.max() -msk.min()) * 255).astype(np.uint8)
        cv2.imwrite((opt.exp + "/" + opt.save_dir+'/original_label_pred_image_'+str(cnt)+'.png'),msk)
                # cv2.imwrite(opt.exp + "/" + opt.save_dir + "/original_label_pred_image_file_{}_{}_dice_{:.4f}.png".format(idx,iw, dice.item()),pred)
                # fig = plt.figure()

                # ax1 = fig.add_subplot(1,3,1)
                # ax1.axis("off")
                # ax1.imshow(origin, cmap = "gray")

                # ax2= fig.add_subplot(1,3,2)
                # ax2.axis("off")
                # ax2.imshow(origin,cmap = "gray")
                # ax2.contour(true, cmap='Greens', linewidths=0.3)

                # ax3 = fig.add_subplot(1,3,3)
                # ax3.axis("off")
                # ax3.imshow(origin,cmap = "gray")
                # ax3.contour(pred, cmap='plasma')

                # plt.axis('off')
                # plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)

                # plt.savefig(opt.exp + "/" + opt.save_dir + "/original_label_pred_image_file_{}_{}_dice_{:.4f}.png".format(idx,iw, dice.item()),bbox_inces='tight', dpi=300)
                # plt.cla()
                # plt.close(fig)
                # plt.gray()

                # break
            ###############################

    

def evaluate_prob(dataset_val, net, opt):
    print("Start Evaluation...")
    net.eval()

    iou_scores = []
    img_dict = {}
    msk_dict ={}
    for idx, (img, mask,iindex) in enumerate(dataset_val):
        # Load Data
        img = torch.Tensor(img).float()
        if opt.use_gpu:
            img = img.cuda(non_blocking=True)
        # print(iindex)
        # Predict
        with torch.no_grad():
            pred = net(img)
            pred /= 50
            y = pred.sigmoid()
            dice = DiceCoef(return_score_per_channel=False)(y, mask.cuda())
            print(len(iindex))
            print(len(img))
            # print(y)
            # Convert to Binary
            zeros = torch.zeros(y.size())
            ones = torch.ones(y.size())
            y = y.cpu()

            # make mask
            # y = torch.where(y > 0.5, ones, zeros) # threshold 0.99

            y = Variable(y).cuda()

            # iou_score = iou_modified(y, mask.cuda(),opt)
            # print(dice.item())
            

            # if iou_score < 0.75:
            ###### Plot & Save Figure #########
            for iw in range(len(img.cpu().numpy())):
                origin = img.cpu().numpy()[iw,0,:,:] 
                pred = y.cpu().numpy()[iw,0,:,:]
                true = mask.cpu().numpy()[iw,0,:,:] 
                iid = iindex[iw]
                img_dict[iw] = origin
                msk_dict[iw] = pred

            print('img_dict',len(img_dict[0]))

                # h = origin.shape[0]
                # w = origin.shape[1]
    #             # fig = plt.figure()

    #             # ax1 = fig.add_subplot(1,3,1)
    #             # ax1.axis("off")
    #             # ax1.imshow(origin, cmap = "gray")

    #             # ax2= fig.add_subplot(1,3,2)
    #             # ax2.axis("off")
    #             # ax2.imshow(origin,cmap = "gray")
    #             # ax2.contour(true, cmap='Greens', linewidths=0.3)

    #             # ax3 = fig.add_subplot(1,3,3)
    #             # ax3.axis("off")
    #             # ax3.imshow(origin,cmap = "gray")
    #             # ax3.imshow(pred, cmap='plasma')

    #             # plt.axis('off')
    #             # plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)

    #             # plt.savefig(opt.exp + "/" + opt.save_dir + "/original_label_pred_image_file_{}_{}_dice_{:.4f}.png".format(idx,iw, dice.item()),bbox_inces='tight', dpi=300)
    #             # plt.cla()
    #             # plt.close(fig)
    #             # plt.gray()

    #             # break
    #         ###############################
    # save_pickle("exp/patch_predict.pkl",img_dict)
    # prec_thresh1, prec_thresh2, iou_mean = avg_precision(iou_scores)

    # print("Presion with threshold 0.5: {}, 0.75: {}, Average: {}".format(prec_thresh1, prec_thresh2, iou_mean))




