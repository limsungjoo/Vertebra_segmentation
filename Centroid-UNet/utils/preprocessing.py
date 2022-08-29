import os
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
from glob import glob

def centercropping(image,mask):
    black = np.zeros((mask.shape[0],mask.shape[1]))

    kernel = np.ones((17, 17), np.uint8)
    dilate = cv2.dilate(mask,kernel,iterations = 2)
    contours, _= cv2.findContours(dilate.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area_list =[]
    if len(contours)<=1:
        black[:,:] = dilate
        
    elif len(contours)>1:
        for i,cnt in enumerate(contours):
            area=cv2.contourArea(cnt)
            area_list.append((area,i))
        area_t =[]
        for j in range(len(area_list)):


            area_t.append(area_list[j][0])
        
        area_t.sort(reverse=True)

        for j in range(len(area_list)):
            if area_list[j][0]==area_t[0]:
                print(area_list[j][0])
                index = area_list[j][1]
                black = cv2.drawContours(black, [contours[index]],0,(255,255,255),-1)

    
    rect,contours = findcontour(black)

    crop_imgs,crop_msks=crop_spine(rect, mask,image)
    return crop_imgs,crop_msks

def findcontour(threshold_img):
    contours, _ = cv2.findContours(threshold_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = [cv2.boundingRect(cnt) for cnt in contours]
    return rect,contours

def crop_spine(rect, threshold_img, org_img):
#     sub_crop_img = ''
#     sub_crop_msk = ''
    crop_imgs = []
    crop_msks = []
    i=0
#     for i in range(len(rect)):
    for i,(x, y, w, h) in enumerate(rect):
        left, top, right, bottom = x, y, x+w, y+h
        sub_crop_msk = crop(threshold_img, give_border((left, top, right, bottom),20,threshold_img.shape, True))
        sub_crop_img = crop(org_img, give_border((left, top, right, bottom),20,threshold_img.shape, True))
 
    return sub_crop_img,sub_crop_msk

def crop(image, coords, border=0, logging=False):
    # coords should be a tuple with (x1, y1, x2, y2)
    # if logging:
    #     print("Shape of image: " + str(shape(image)))
    left, top, right, bottom = coords

    if isinstance(image, np.ndarray):
        return image[top: bottom, left: right]
    else:
        return image.crop((left, top, right, bottom))

def give_border(coords, border, shp, logging=False):
    flag = False
    coords = coords[0]-border, coords[1]-border, coords[2]+border, coords[3]+border

    if logging:
        print("Give border: " + str(coords))

    if coords[0] < 0:
        print("Left small: " + str(coords[0]))
        coords = (0, coords[1], coords[2], coords[3])
        flag = True
    if coords[1] < 0:
        print("Top small: " + str(coords[1]))
        coords = (coords[0], 0, coords[2], coords[3])
        flag = True

    if coords[2] > shp[1]:
        print("Right big: " + str(coords[2]))
        coords = (coords[0], coords[1], shp[1], coords[3])
        flag = True

    if coords[3] > shp[0]:
        print("Bottom Big: " + str(coords[3]))
        coords = (coords[0], coords[1], coords[2], shp[0])
        flag = True

    # if flag == False:
    #     print("Nothing has changed in give border")
    return coords