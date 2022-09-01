import torch
from torch.utils.data import Dataset

import cv2
import os
import numpy as np
from glob import glob
import SimpleITK as sitk

from utils.transforms import image_windowing, image_minmax, mask_binarization, augment_imgs_and_masks, center_crop
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.modules.conv as conv

def image_p(img):   
    h,w = img.shape
        
    bg_img = np.zeros((512,512))
#     bg_msk = np.zeros((1024,1024))


    if w>h:
        x=512
        y=int(h/w *x)
    else:
        y=512
        x=int(w/h *y)

        if x >512:
            x =512
            y= int(h/w *x)

    img_resize = cv2.resize(img, (x,y))
#     msk_resize = cv2.resize(mask, (x,y))
    xs = int((512 - x)/2)
    ys = int((512-y)/2)
    bg_img[ys:ys+y,xs:xs+x]=img_resize
#     bg_msk[ys:ys+y,xs:xs+x]=msk_resize
    img = bg_img
#     mask = bg_msk
    return img
class VertebralDataset(Dataset):
    def __init__(self,  is_Train=True, augmentation=True):
        super(VertebralDataset, self).__init__()

        self.augmentation = augmentation
        self.is_Train = is_Train
        
        if is_Train:
            self.mask_list = sorted(glob('/data/workspace/vfuser/sungjoo/TransUnet/data/mask/*'))
            # print(self.mask_list)
            self.mask_list = self.mask_list[:int(len(self.mask_list)*0.8)]
            # print(self.mask_list)
            print(len(self.mask_list),"dataset: Training")
        
        else:
            self.mask_list = sorted(glob('/data/workspace/vfuser/sungjoo/data/dongsam/mask/*'))
            # self.mask_list = self.mask_list[int(len(self.mask_list)*0.6):int(len(self.mask_list)*0.8)]
            self.mask_list = self.mask_list[:]
            print(len(self.mask_list),"dataset: Vailidation")
                
        
        self.len = len(self.mask_list)

        # self.augmentation = augmentation
        # self.opt = opt

        # self.is_Train = is_Train


    def __getitem__(self, index):
        # Load Image and Mask
        mask_path = self.mask_list[index]
        print(mask_path)
        xray_path = mask_path.replace('/mask/', '/image/').replace('.png', '.jpg')
        
        img = cv2.imread(xray_path, 0)
        mask = cv2.imread(mask_path, 0)
        
        # HU Windowing
        # img = image_windowing(img, self.opt.w_min, self.opt.w_max)

        # Center Crop and MINMAX to [0, 255] and Resize
        # img = center_crop(img, self.opt.crop_size)
        # mask = center_crop(mask, self.opt.crop_size)
        
        img = image_minmax(img)
        
        # img = cv2.resize(img, (self.opt.input_size, self.opt.input_size))
        # mask = cv2.resize(mask, (self.opt.input_size, self.opt.input_size))
        img = image_p(img)
        mask =image_p(mask)
        img = cv2.resize(img, (512,512))
        mask = cv2.resize(mask, (512,512))

        # MINMAX to [0, 1]
        img = img / 255.

        # Mask Binarization (0 or 1)
        mask = mask_binarization(mask)

        # Add channel axis
        # img = img.astype(np.float32)
        # mask = mask.astype(np.float32)
        img = img[None, ...].astype(np.float32)
        mask = mask[None, ...].astype(np.float32)
        # print(img.shape)
        # img = img[None, ...].astype(np.float32)
        # mask = mask[None, ...].astype(np.float32)
        # print(img.shape)
        # img = torch.from_numpy(img)
        # mask = torch.from_numpy(mask)
        # print(img.shape)
        # Add Coordconv
        # img = CoordConv2d(img)
        # mask =  CoordConv2d(mask)
        
        # img = np.reshape(1,224,224)
        # mask = np.reshape(1,224,224)

        # Augmentation
        

        return img, mask
        
    def __len__(self):
        return self.len


