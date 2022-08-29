import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
import numpy as np
from glob import glob
import SimpleITK as sitk

from utils.preprocessing import centercropping
from utils.transforms import image_windowing, image_minmax, mask_binarization, augment_imgs_and_masks, center_crop
import matplotlib.pyplot as plt

class VertebralDataset(Dataset):
    def __init__(self, opt, is_Train=True, augmentation=True):
        super(VertebralDataset, self).__init__()

        self.augmentation = augmentation
        self.opt = opt
        self.is_Train = is_Train
        
        if is_Train:
            self.mask_list = sorted(glob(opt.data_root))

            # print(self.mask_list)
            self.mask_list = self.mask_list[:int(len(self.mask_list)*0.80)]
            # print(self.mask_list)
            print(len(self.mask_list),"dataset: Training")
        
        else:
            self.mask_list = sorted(glob(opt.data_root))
            self.mask_list = self.mask_list[:]
            # self.mask_list = self.mask_list[int(len(self.mask_list)*0.9):]
            print(len(self.mask_list),"dataset: Vailidation")
                
        
        self.len = len(self.mask_list)

        # self.augmentation = augmentation
        # self.opt = opt

        # self.is_Train = is_Train


    def __getitem__(self, index):
        # Load Image and Mask
        mask_path = self.mask_list[index]
        
        
        # xray_path = mask_path.replace('/Label/', '/Dataset/').replace('_label.png', '.png')
        xray_path = mask_path.replace('/mask/', '/spine/').replace('.png', '.jpg')
        img = cv2.imread(xray_path, 0)
        # print('img:',xray_path)
        # print('msk:',mask_path)
        mask = cv2.imread(mask_path, 0)
        img = cv2.equalizeHist(img)
        # img,mask = centercropping(img,mask)
        

        # HU Windowing
        # img = image_windowing(img, self.opt.w_min, self.opt.w_max)

        # Center Crop and MINMAX to [0, 255] and Resize
        # img = center_crop(img, self.opt.crop_size)
        # mask = center_crop(mask, self.opt.crop_size)
        
        img = image_minmax(img)
        
        ori_size = img.shape

        h,w = img.shape
        bg_img = np.zeros((1024,512))
        bg_msk = np.zeros((1024,512))

        if w>h:
            x=512
            y=int(h/w *x)
        else:
            y=1024
            x=int(w/h *y)

            if x >512:
                x =512
                y= int(h/w *x)
        
        img_resize = cv2.resize(img, (x,y))
        msk_resize = cv2.resize(mask, (x,y))

        xs = int((512 - x)/2)
        ys = int((1024-y)/2)
        bg_img[ys:ys+y,xs:xs+x]=img_resize
        bg_msk[ys:ys+y,xs:xs+x]=msk_resize

        img = bg_img
        mask = bg_msk

        cv2.imwrite('/home/vfuser/sungjoo/Resize_model/exp/image_check/img/'+str(index)+'.jpg',img)
        cv2.imwrite('/home/vfuser/sungjoo/Resize_model/exp/image_check/msk/'+str(index)+'.png',mask)
        # MINMAX to [0, 1]
        img = img / 255.

        # Mask Binarization (0 or 1)
        mask = mask_binarization(mask)

        # cv2.imwrite('/home/vfuser/sungjoo/Resize_model/exp/image_check/img/'+str(index)+'.jpg',img)
        # cv2.imwrite('/home/vfuser/sungjoo/Resize_model/exp/image_check/msk/'+str(index)+'.png',mask)
        # Add channel axis
        img = img[None, ...].astype(np.float32)
        mask = mask[None, ...].astype(np.float32)
                
        # Augmentation
        if self.augmentation:
            img, mask = augment_imgs_and_masks(img, mask, self.opt.rot_factor, self.opt.scale_factor, self.opt.trans_factor, self.opt.flip)

        return img, mask
        
    def __len__(self):
        return self.len
class Vertebral_patchbasedDataset(Dataset):
    def __init__(self, opt, is_Train=True, augmentation=True):
        super(Vertebral_patchbasedDataset, self).__init__()

        self.augmentation = augmentation
        self.opt = opt
        self.is_Train = is_Train
        
        if is_Train:
            self.mask_list = sorted(glob(opt.data_root))
            self.mask_list = self.mask_list[:int(len(self.mask_list)*0.95)]
            # self.mask_list = self.mask_list[:10]
            
            self.elementslist = self.convert2patches()

            print(len(self.mask_list),"Train dataset creates {} patches".format(len(self.elementslist)))
        
        else:
            self.mask_list = sorted(glob(opt.data_root))
            self.mask_list = self.mask_list[int(len(self.mask_list)*0.95):]
            # self.mask_list = self.mask_list[10:12]
            
            self.elementslist = self.convert2patches()

            print(len(self.mask_list)," Vailidation dataset creates {} patches".format(len(self.elementslist)))
                
        
        self.len = len(self.elementslist)


    def __getitem__(self, index):
        # Load Image and Mask
        img,mask,iindex = self.elementslist[index]
        
        img = image_minmax(img)
        
        img = cv2.resize(img, (self.opt.input_size, self.opt.input_size))
        mask = cv2.resize(mask, (self.opt.input_size, self.opt.input_size))

        # MINMAX to [0, 1]
        img = img / 255.

        # Mask Binarization (0 or 1)
        mask = mask_binarization(mask)

        # Add channel axis
        img = img[None, ...].astype(np.float32)
        mask = mask[None, ...].astype(np.float32)
                
        # Augmentation
        if self.augmentation:
            img, mask = augment_imgs_and_masks(img, mask, self.opt.rot_factor, self.opt.scale_factor, self.opt.trans_factor, self.opt.flip)

        return img,mask,iindex
        
    def __len__(self):
        return self.len

    def convert2patches(self):
        res_list = []
        index_list = []

        stride = 200
        imgsize = 200

        for i,mask_path in enumerate(self.mask_list):
            xray_path = mask_path.replace('/Label/', '/Dataset/').replace('_label.png', '.png')
            mask =cv2.imread(mask_path ,0)
            img = cv2.imread(xray_path, 0)


            h,w = img.shape

            for i_r,height in enumerate(range(0,h-2*stride,stride)):
                for i_c,width in enumerate(range(0,w-2*stride,stride)):
                    img_patch = img[height:height+imgsize,width:width+imgsize]
                    mask_patch = mask[height:height+imgsize,width:width+imgsize]

                    if mask_patch.sum() > 0:
                        res_list.append((img_patch,mask_patch,str(i)+"_"+str(i_r)+"_"+str(i_c)))

                img_patch = img[height:height+imgsize,width-imgsize:width]
                mask_patch = mask[height:height+imgsize,width-imgsize:width]
                
                if img_patch.shape == (200,200) and mask_patch.sum() > 0:
                    res_list.append((img_patch,mask_patch,str(i)+"_"+str(i_r)+"_l"))

            # last row
            for i_c,width in enumerate(range(0,w-2*stride,stride)):
                img_patch = img[height-imgsize:height,width:width+imgsize]
                mask_patch = mask[height-imgsize:height,width:width+imgsize]
                if img_patch.shape == (200,200) and mask_patch.sum() > 0:
                    res_list.append((img_patch,mask_patch,str(i)+"_"+"l_"+str(i_c)))

            img_patch = img[height-imgsize:height,width-imgsize:width]
            mask_patch = mask[height-imgsize:height,width-imgsize:width]
            # print(img_patch.shape,mask_patch.shape)
            if img_patch.shape == (200,200) and mask_patch.sum() > 0:
                res_list.append((img_patch,mask_patch,str(i)+"_"+"l_l"))

        return res_list

# class Vertebral_patchbasedDataset(Dataset):
#     def __init__(self, opt, is_Train=True, augmentation=True):
#         super(Vertebral_patchbasedDataset, self).__init__()

#         self.augmentation = augmentation
#         self.opt = opt
#         self.is_Train = is_Train
        
#         if is_Train:
#             self.mask_list = sorted(glob(opt.data_root))
#             self.mask_list = self.mask_list[:int(len(self.mask_list)*0.6)]
#             # self.mask_list = self.mask_list[:10]
            
#             self.elementslist = self.convert2patches()

#             print(len(self.mask_list),"Train dataset creates {} patches".format(len(self.elementslist)))
        
#         else:
#             self.mask_list = sorted(glob(opt.data_root))
#             self.mask_list = self.mask_list[:]
#             # self.mask_list = self.mask_list[10:12]
            
#             self.elementslist = self.convert2patches()

#             print(len(self.mask_list)," Vailidation dataset creates {} patches".format(len(self.elementslist)))
                
        
#         self.len = len(self.elementslist)


#     def __getitem__(self, index):
#         # Load Image and Mask
#         img_list,basename = self.elementslist[index]
        
#         return_list = []

#         for img,mask,label in img_list:

#             img = image_minmax(img)

#             img = cv2.resize(img, (self.opt.input_size, self.opt.input_size))
#             mask = cv2.resize(mask, (self.opt.input_size, self.opt.input_size))

#             # Mask Binarization (0 or 1)
#             mask = mask_binarization(mask)

#             # Add channel axis
#             img = img[None, ...].astype(np.float32)
#             mask = mask[None, ...].astype(np.float32)

#             return_list.append((img,mask,label))
        

#         return return_list,basename
        
#     def __len__(self):
#         return self.len

#     def convert2patches(self):
#         index_list = []
#         img_list = []

#         stride = 100
#         imgsize = 200
#         cnt =0
#         for i,mask_path in enumerate(self.mask_list):
#             res_list = []

#             xray_path = mask_path.replace('/Label/', '/Dataset/').replace('_label.png', '.png')
#             mask =cv2.imread(mask_path ,0)
#             img = cv2.imread(xray_path, 0)
#             mask = cv2.resize(mask, (800,1200))
#             img = cv2.resize(img, (800,1200))
#             cv2.imwrite('/home/sungjoo/VF/Coord_seg/exp/overall/gt'+'/original_label_'+str(cnt)+'.png',mask)
#             cnt+=1

#             h,w = img.shape

#             for i_r,height in enumerate(range(0,h-2*stride,stride)):
#                 for i_c,width in enumerate(range(0,w-2*stride,stride)):
#                     img_patch = img[height:height+imgsize,width:width+imgsize]
#                     mask_patch = mask[height:height+imgsize,width:width+imgsize]

#                     if mask_patch.sum() >= 0:
#                         res_list.append((img_patch,mask_patch,str(i)+"_"+str(i_r)+"_"+str(i_c)))

#                 img_patch = img[height:height+imgsize,width-imgsize:width]
#                 mask_patch = mask[height:height+imgsize,width-imgsize:width]
                
#                 if img_patch.shape == (200,200) and mask_patch.sum() >= 0:
#                     res_list.append((img_patch,mask_patch,str(i)+"_"+str(i_r)+"_l"))

#             # last row
#             for i_c,width in enumerate(range(0,w-2*stride,stride)):
#                 img_patch = img[height-imgsize:height,width:width+imgsize]
#                 mask_patch = mask[height-imgsize:height,width:width+imgsize]
#                 if img_patch.shape == (200,200) and mask_patch.sum() >= 0:
#                     res_list.append((img_patch,mask_patch,str(i)+"_"+"l_"+str(i_c)))

#             img_patch = img[height-imgsize:height,width-imgsize:width]
#             mask_patch = mask[height-imgsize:height,width-imgsize:width]
#             # print(img_patch.shape,mask_patch.shape)
#             if img_patch.shape == (200,200) and mask_patch.sum() >= 0:
#                 res_list.append((img_patch,mask_patch,str(i)+"_"+"l_l"))

#             img_list.append((res_list,mask_path))

#         return img_list


class Vertebral_patchbasedDataset_train(Dataset):
    def __init__(self, opt, is_Train=True, augmentation=True):
        super(Vertebral_patchbasedDataset, self).__init__()

        self.augmentation = augmentation
        self.opt = opt
        self.is_Train = is_Train
        
        if is_Train:
            self.mask_list = sorted(glob(opt.data_root))
            self.mask_list = self.mask_list[:int(len(self.mask_list)*0.6)]
            # self.mask_list = self.mask_list[:10]
            
            self.elementslist = self.convert2patches()

            print(len(self.mask_list),"Train dataset creates {} patches".format(len(self.elementslist)))
        
        else:
            self.mask_list = sorted(glob(opt.data_root))
            self.mask_list = self.mask_list[int(len(self.mask_list)*0.6):int(len(self.mask_list)*0.8)]
            # self.mask_list = self.mask_list[10:12]
            
            self.elementslist = self.convert2patches()

            print(len(self.mask_list)," Vailidation dataset creates {} patches".format(len(self.elementslist)))
                
        
        self.len = len(self.elementslist)


    def __getitem__(self, index):
        # Load Image and Mask
        img,mask,iindex = self.elementslist[index]
        
        img = image_minmax(img)
        
        img = cv2.resize(img, (self.opt.input_size, self.opt.input_size))
        mask = cv2.resize(mask, (self.opt.input_size, self.opt.input_size))

        # MINMAX to [0, 1]
        img = img / 255.

        # Mask Binarization (0 or 1)
        mask = mask_binarization(mask)

        # Add channel axis
        img = img[None, ...].astype(np.float32)
        mask = mask[None, ...].astype(np.float32)
                
        # Augmentation
        if self.augmentation:
            img, mask = augment_imgs_and_masks(img, mask, self.opt.rot_factor, self.opt.scale_factor, self.opt.trans_factor, self.opt.flip)

        return img,mask,iindex
        
    def __len__(self):
        return self.len

    def convert2patches(self):
        res_list = []
        index_list = []

        stride = 50
        imgsize = 200

        for i,mask_path in enumerate(self.mask_list):
            xray_path = mask_path.replace('/mask/', '/spine/').replace('.png', '.jpg')
            mask =cv2.imread(mask_path ,0)
            img = cv2.imread(xray_path, 0)


            h,w = img.shape

            for i_r,height in enumerate(range(0,h-2*stride,stride)):
                for i_c,width in enumerate(range(0,w-2*stride,stride)):
                    img_patch = img[height:height+imgsize,width:width+imgsize]
                    mask_patch = mask[height:height+imgsize,width:width+imgsize]

                    if mask_patch.sum() > 0:
                        res_list.append((img_patch,mask_patch,str(i)+"_"+str(i_r)+"_"+str(i_c)))

                img_patch = img[height:height+imgsize,width-imgsize:width]
                mask_patch = mask[height:height+imgsize,width-imgsize:width]
                
                if img_patch.shape == (200,200) and mask_patch.sum() > 0:
                    res_list.append((img_patch,mask_patch,str(i)+"_"+str(i_r)+"_l"))

            # last row
            for i_c,width in enumerate(range(0,w-2*stride,stride)):
                img_patch = img[height-imgsize:height,width:width+imgsize]
                mask_patch = mask[height-imgsize:height,width:width+imgsize]
                if img_patch.shape == (200,200) and mask_patch.sum() > 0:
                    res_list.append((img_patch,mask_patch,str(i)+"_"+"l_"+str(i_c)))

            img_patch = img[height-imgsize:height,width-imgsize:width]
            mask_patch = mask[height-imgsize:height,width-imgsize:width]
            # print(img_patch.shape,mask_patch.shape)
            if img_patch.shape == (200,200) and mask_patch.sum() > 0:
                res_list.append((img_patch,mask_patch,str(i)+"_"+"l_l"))

        return res_list

# class Points_patchbasedDataset(Dataset):
#     def __init__(self, opt, is_Train=True, augmentation=True):
#         super(Points_patchbasedDataset, self).__init__()

#         self.augmentation = augmentation
#         self.opt = opt
#         self.is_Train = is_Train
        
#         if is_Train:
#             self.mask_list = sorted(glob(opt.data_root))
#             self.mask_list = self.mask_list[:int(len(self.mask_list)*0.8)]
            
#             self.elementslist = self.convert2patches(self.mask_list)

#             print(len(self.mask_list),"Train dataset creates {} patches".format(len(self.elementslist)))
        
#         else:
#             self.mask_list = sorted(glob(opt.data_root))
#             self.mask_list = self.mask_list[int(len(self.mask_list)*0.8):]
            
#             self.elementslist = self.convert2patches(self.mask_list)

#             print(len(self.mask_list)," Vailidation dataset creates {} patches".format(len(self.elementslist)))
                
        
#         self.len = len(self.elementslist)


#     def __getitem__(self, index):
#         # Load Image and Mask
#         img,mask,iindex = self.elementslist[index]
        
#         try:
#             img = image_minmax(img)
#         except:
#             print(img)
#             print(img.shape,mask.shape)

        
#         img = cv2.resize(img, (self.opt.input_size, self.opt.input_size))
#         mask = cv2.resize(mask, (self.opt.input_size, self.opt.input_size))

#         # MINMAX to [0, 1]
#         img = img / 255.

#         # Mask Binarization (0 or 1)
#         mask = mask_binarization(mask)

#         # Add channel axis
#         img = img[None, ...].astype(np.float32)
#         mask = mask[None, ...].astype(np.float32)
                
#         # Augmentation
#         if self.augmentation:
#             img, mask = augment_imgs_and_masks(img, mask, self.opt.rot_factor, self.opt.scale_factor, self.opt.trans_factor, self.opt.flip)

#         return img, mask, iindex
        
#     def __len__(self):
#         return self.len

#     def convert2patches(self,mask_list):
#         res_list = []
#         index_list = []

#         stride = 200
#         imgsize = 600
#         for i in range(len(mask_list)):

#             # Load Image and Mask
#             mask_path = self.mask_list[i]
#             xray_path = mask_path.replace('/point_mask/', '/img_dcm/').replace('.jpg', '.dcm')

#             if self.is_Train == False:
#                 img = sitk.ReadImage(xray_path)
#                 img = sitk.GetArrayFromImage(img)[0]
#                 plt.imshow(img)
#                 plt.show()

#             img = sitk.ReadImage(xray_path)
#             img = sitk.GetArrayFromImage(img)[0]
#             mask = cv2.imread(mask_path, 0)
#             # print(img.shape)
#             h,w = img.shape

#             for i_r,height in enumerate(range(0,h-2*stride,stride)):
#                 for i_c,width in enumerate(range(0,w-2*stride,stride)):
#                     img_patch = img[height:height+imgsize,width:width+imgsize]
#                     mask_patch = mask[height:height+imgsize,width:width+imgsize]

#                     res_list.append((img_patch,mask_patch,str(i)+"_"+str(i_r)+"_"+str(i_c)))

#                 img_patch = img[height:height+imgsize,width-imgsize:width]
#                 mask_patch = mask[height:height+imgsize,width-imgsize:width]
                
#                 if img_patch.shape == (600,600):
#                     res_list.append((img_patch,mask_patch,str(i)+"_"+str(i_r)+"_l"))

#             # last row
#             for i_c,width in enumerate(range(0,w-2*stride,stride)):
#                 img_patch = img[height-imgsize:height,width:width+imgsize]
#                 mask_patch = mask[height-imgsize:height,width:width+imgsize]
#                 if img_patch.shape == (600,600):
#                     res_list.append((img_patch,mask_patch,str(i)+"_"+"l_"+str(i_c)))

#             img_patch = img[height-imgsize:height,width-imgsize:width]
#             mask_patch = mask[height-imgsize:height,width-imgsize:width]
#             # print(img_patch.shape,mask_patch.shape)
#             if img_patch.shape == (600,600):
#                 res_list.append((img_patch,mask_patch,str(i)+"_"+"l_l"))

#         return res_list




