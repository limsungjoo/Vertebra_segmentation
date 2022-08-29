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
import os
import torch
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
from torch.autograd import Variable

from utils import AverageMeter
from utils.metrics import DiceCoef
from utils.losses import iou_modified, avg_precision
from utils.psave import *

from matplotlib import pyplot as plt

import numpy
from scipy.ndimage.morphology import generate_binary_structure,binary_erosion
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.measurements import label, find_objects

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
    for it, (img, mask) in enumerate(dataset_trn):
        
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

        # Progress report (http://localhost:8097) (python -m visdom.server)
        logger.log({'loss':loss}, 
                   images={'image': img, 'pred': pred, 'GT':mask})
        # Calculation Dice Coef Score
        dice = DiceCoef(return_score_per_channel=False)(pred, mask)
        total_dices.update(dice.item(), img.size(0))
        
        # Convert to Binary
        zeros = torch.zeros(pred.size())
        ones = torch.ones(pred.size())
        pred = pred.cpu()

        pred = torch.where(pred > 0.9, ones, zeros).cuda() # threshold 0.99

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

    for it, (img, mask) in enumerate(dataset_val):
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

        pred = torch.where(pred > 0.9, ones, zeros).cuda()
        
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
            print(net.state_dict(), model_filename)
            torch.save(net.state_dict(), model_filename)
        # Multi GPU
        else:
            torch.save(net.module.state_dict(), model_filename)

    print('>>> Current best: IoU: %.8f in %3d epoch\n' % (best_iou, best_epoch+1))
    
    return best_iou, best_epoch


# def evaluate(dataset_val, net, opt):
#     print("Start Evaluation...")
#     net.eval()
    
#     iou_scores = []
#     cnt =0
#     for idx, (img_list,mskID) in enumerate(dataset_val):
#         img_dict={}
#         print(len(img_list))
#         for img, mask,iindex in (img_list):
#             # Load Data
#             img = torch.Tensor(img).float()
#             if opt.use_gpu:
#                 img = img.cuda(non_blocking=True)
#             # print(iindex)
#             # Predict
#             with torch.no_grad():
#                 pred = net(img)
                
#                 y = pred.sigmoid()
#                 zeros = torch.zeros(pred.size())
#                 ones = torch.ones(pred.size())
#                 pred = pred.cpu()

#                 pred = torch.where(pred > 0.9, ones, zeros).cuda()
#                 dice = DiceCoef(return_score_per_channel=False)(y, mask.cuda())
#                 pred = y.cpu().numpy()[0,0,:,:]
                 
#                 img_dict[iindex]=pred
            
#         max_row_col=[0,0]
        
#         for k in img_dict.keys():
            
#             if k[0].split('_')[1]!='l' and (int(max_row_col[0])<int(k[0].split('_')[1])):
                
#                 max_row_col[0] = k[0].split('_')[1]
#             if k[0].split('_')[2]!='l' and (int(max_row_col[1])<int(k[0].split('_')[2])):
#                 max_row_col[1] = k[0].split('_')[2]
#         msk = cv2.imread(mskID[0],0) 
#         msk = cv2.resize(msk, (800,1200))
#         probimg = np.zeros(msk.shape)
#         # probimg = np.zeros((200+(1*int(max_row_col[0])),200+(1*int(max_row_col[1]))))
#         probimg_real = probimg.copy()

#         i = list(img_dict.keys())[0][0].split('_')[0]
#         print(max_row_col[0])
#         print(max_row_col[1])
#         for h in range(0,int(max_row_col[0])+1):
#             for w in range(0,int(max_row_col[1])+1):  
#                     img = cv2.resize(img_dict[(str(i)+'_'+str(h)+'_'+str(w),)], (200,200))
                    
#                     if probimg[h*100:h*100+200,w*100:w*100+200].sum() ==0:
#                         probimg[h*100:h*100+200,w*100:w*100+200] +=img
#                     else:
#                         probimg[h*100:h*100+200,w*100:w*100+200] = (probimg[h*100:h*100+200,w*100:w*100+200])
#         probimg = probimg/2
#         probimg[probimg<=0.49]=0 #0.49
#         probimg = ((probimg - probimg.min()) / (probimg.max() -probimg.min()) * 255).astype(np.uint8)
#         cv2.imwrite((opt.exp + "/" + opt.save_dir+'/original_label_'+str(cnt)+'.png'),probimg)
#         cnt+=1
#         print('post_shape:',probimg.shape)
def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = numpy.asarray(voxelspacing, dtype=numpy.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
            
    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)
    
    # test for emptiness
    if 0 == numpy.count_nonzero(result): 
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == numpy.count_nonzero(reference): 
        raise RuntimeError('The second supplied array does not contain any binary object.')    
            
    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    
    # compute average surface distance        
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    
    return sds

def hd95(result, reference, voxelspacing=None, connectivity=1):
    import numpy
    """
    95th percentile of the Hausdorff Distance.

    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of 
        elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`hd`

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = numpy.percentile(numpy.hstack((hd1, hd2)), 95)
    return hd95
def averaged_hausdorff_distance(set1, set2, max_ahd=np.inf):
    from sklearn.metrics import pairwise_distances
    """
    Compute the Averaged Hausdorff Distance function
    between two unordered sets of points (the function is symmetric).
    Batches are not supported, so squeeze your inputs first!
    :param set1: Array/list where each row/element is an N-dimensional point.
    :param set2: Array/list where each row/element is an N-dimensional point.
    :param max_ahd: Maximum AHD possible to return if any set is empty. Default: inf.
    :return: The Averaged Hausdorff Distance between set1 and set2.
    """

    if len(set1) == 0 or len(set2) == 0:
        return max_ahd

    set1 = np.array(set1)
    set2 = np.array(set2)

    assert set1.ndim == 2, 'got %s' % set1.ndim
    assert set2.ndim == 2, 'got %s' % set2.ndim

    assert set1.shape[1] == set2.shape[1], \
        'The points in both sets must have the same number of dimensions, got %s and %s.'\
        % (set2.shape[1], set2.shape[1])

    d2_matrix = pairwise_distances(set1, set2, metric='euclidean')

    res = np.average(np.min(d2_matrix, axis=0)) + \
        np.average(np.min(d2_matrix, axis=1))

    return res   
def __distinct_binary_object_correspondences(reference, result, connectivity=1):
    """
    Determines all distinct (where connectivity is defined by the connectivity parameter
    passed to scipy's `generate_binary_structure`) binary objects in both of the input
    parameters and returns a 1to1 mapping from the labelled objects in reference to the
    corresponding (whereas a one-voxel overlap suffices for correspondence) objects in
    result.
    
    All stems from the problem, that the relationship is non-surjective many-to-many.
    
    @return (labelmap1, labelmap2, n_lables1, n_labels2, labelmapping2to1)
    """
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))
    
    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)
    
    # label distinct binary objects
    labelmap1, n_obj_result = label(result, footprint)
    labelmap2, n_obj_reference = label(reference, footprint)
    
    # find all overlaps from labelmap2 to labelmap1; collect one-to-one relationships and store all one-two-many for later processing
    slicers = find_objects(labelmap2) # get windows of labelled objects
    mapping = dict() # mappings from labels in labelmap2 to corresponding object labels in labelmap1
    used_labels = set() # set to collect all already used labels from labelmap2
    one_to_many = list() # list to collect all one-to-many mappings
    for l1id, slicer in enumerate(slicers): # iterate over object in labelmap2 and their windows
        l1id += 1 # labelled objects have ids sarting from 1
        bobj = (l1id) == labelmap2[slicer] # find binary object corresponding to the label1 id in the segmentation
        l2ids = numpy.unique(labelmap1[slicer][bobj]) # extract all unique object identifiers at the corresponding positions in the reference (i.e. the mapping)
        l2ids = l2ids[0 != l2ids] # remove background identifiers (=0)
        if 1 == len(l2ids): # one-to-one mapping: if target label not already used, add to final list of object-to-object mappings and mark target label as used
            l2id = l2ids[0]
            if not l2id in used_labels:
                mapping[l1id] = l2id
                used_labels.add(l2id)
        elif 1 < len(l2ids): # one-to-many mapping: store relationship for later processing
            one_to_many.append((l1id, set(l2ids)))
            
    # process one-to-many mappings, always choosing the one with the least labelmap2 correspondences first
    while True:
        one_to_many = [(l1id, l2ids - used_labels) for l1id, l2ids in one_to_many] # remove already used ids from all sets
        one_to_many = [x for x in one_to_many if x[1]] # remove empty sets
        one_to_many = sorted(one_to_many, key=lambda x: len(x[1])) # sort by set length
        if 0 == len(one_to_many):
            break
        l2id = one_to_many[0][1].pop() # select an arbitrary target label id from the shortest set
        mapping[one_to_many[0][0]] = l2id # add to one-to-one mappings 
        used_labels.add(l2id) # mark target label as used
        one_to_many = one_to_many[1:] # delete the processed set from all sets
    
    return labelmap1, labelmap2, n_obj_result, n_obj_reference, mapping

def __combine_windows(w1, w2):
    """
    Joins two windows (defined by tuple of slices) such that their maximum
    combined extend is covered by the new returned window.
    """
    res = []
    for s1, s2 in zip(w1, w2):
        res.append(slice(min(s1.start, s2.start), max(s1.stop, s2.stop)))
    return tuple(res)

def obj_asd(result, reference, voxelspacing=None, connectivity=1):

    sds = list()
    labelmap1, labelmap2, _a, _b, mapping = __distinct_binary_object_correspondences(result, reference, connectivity)
    slicers1 = find_objects(labelmap1)
    slicers2 = find_objects(labelmap2)
    for lid2, lid1 in list(mapping.items()):
        window = __combine_windows(slicers1[lid1 - 1], slicers2[lid2 - 1])
        object1 = labelmap1[window] == lid1
        object2 = labelmap2[window] == lid2
        sds.extend(__surface_distances(object1, object2, voxelspacing, connectivity))
    asd = numpy.mean(sds)
    return asd
def obj_assd(result, reference, voxelspacing=None, connectivity=1):
    """
    Average symmetric surface distance.
    
    Computes the average symmetric surface distance (ASSD) between the binary objects in
    two images.
    
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining what accounts
        for a distinct binary object as well as when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        The decision on the connectivity is important, as it can influence the results
        strongly. If in doubt, leave it as it is.
        
    Returns
    -------
    assd : float
        The average symmetric surface distance between all mutually existing distinct
        binary object(s) in ``result`` and ``reference``. The distance unit is the same as for
        the spacing of elements along each dimension, which is usually given in mm.
        
    See also
    --------
    :func:`obj_asd`
    
    Notes
    -----
    This is a real metric, obtained by calling and averaging
    
    >>> obj_asd(result, reference)
    
    and
    
    >>> obj_asd(reference, result)
    
    The binary images can therefore be supplied in any order.
    """
    assd = numpy.mean( (obj_asd(result, reference, voxelspacing, connectivity), obj_asd(reference, result, voxelspacing, connectivity)) )
    return assd


def evaluate(dataset_val, net, opt):
    
    print("Start Evaluation...")
    net.eval()

    iou_scores = []
    i=0
    k=0
    total = 0
    for idx, (img, mask) in enumerate(dataset_val):
        # Load Data
        img = torch.Tensor(img).float()
        if opt.use_gpu:
            img = img.cuda(non_blocking=True)

        # Predict
        with torch.no_grad():
            pred = net(img)
	        
            y = pred.sigmoid()
            # dice = DiceCoef(return_score_per_channel=False)(y, mask.cuda())
            
            # Convert to Binary
            zeros = torch.zeros(y.size())
            ones = torch.ones(y.size())
            y = y.cpu()

            # make mask
            y = torch.where(y > 0.9, ones, zeros) # threshold 0.99
            dice = DiceCoef(return_score_per_channel=False)(y, mask.cpu())
            # y = Variable(y).cuda()
            pA = SamplewiseAccuracy((y,mask.cpu()))
            meanIoU = MeanIoU((y,mask.cpu()))
            frequencyWeightedIoU =FrequencyWeightedIoU((y,mask.cpu()))
            euclidean_di= euclidean_dis(y,mask.cpu())
            print(euclidean_di)
            iou_score = iou_modified(y, mask.cpu(),opt)
            
            # print(dice)

            # if iou_score < 0.75:
            ###### Plot & Save Figure #########
            import cv2
            
            for iw in range(len(img.cpu().numpy())):
                origin = img.cpu().numpy()[iw,0,:,:] 
                pred = y.cpu().numpy()[iw,0,:,:]
                true = mask.cpu().numpy()[iw,0,:,:]
                pred = pred[:,:,np.newaxis]
                true= true[:,:,np.newaxis]
                # print(true.shape)	
                origin = origin[:,:,np.newaxis]
                origin=origin*255
                # print(origin.shape)
                pred=pred*255
                true = true*255
                cv2.imwrite('/home/vfuser/sungjoo/Centroid-Unet/exp/origin/'+str(k)+'.jpg',origin)
                cv2.imwrite('/home/vfuser/sungjoo/Centroid-Unet/exp/pred/'+str(k)+'.png',pred)
                cv2.imwrite('/home/vfuser/sungjoo/Centroid-Unet/exp/gt/'+str(k)+'.png',true)
                
                # origin=origin *255
                # i+= hd95(pred,true)
                # i+=obj_asd(pred,true)
                # i+=dice.item()
                # i+=iou_score.item()
                # pre, rec ,acc, spe= compute_pre_rec(true,pred)
                # i+=(2*pre*rec)/(pre+rec)
                # print((2*pre*rec)/(pre+rec))
                # print(rec)
                # i+=dice.item()
            #     i+=meanIoU
                # i+=frequencyWeightedIoU
                #i+=pre
                # i+=rec
                i+=euclidean_di
                if euclidean_di>10:
                    cv2.imwrite('/home/vfuser/sungjoo/Centroid-Unet/exp/10>/origin/'+str(k)+'.jpg',origin)
                    cv2.imwrite('/home/vfuser/sungjoo/Centroid-Unet/exp/10>/pred/'+str(k)+'.png',pred)
                    cv2.imwrite('/home/vfuser/sungjoo/Centroid-Unet/exp/10>/gt/'+str(k)+'.png',true)
                # kernel = np.ones((15,15),np.uint8)
                # kernel_3 = np.ones((7,7),np.uint8)

                # mask_C=pred.copy()
                # mask = cv2.morphologyEx(pred,cv2.MORPH_OPEN,kernel_3)
                # mask =cv2.dilate(mask, kernel, iterations=2)
                # mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
                # mask_cp = pred.copy()
                
                
                # ret, thr = cv2.threshold(mask,127,255,0)
                # contours,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                # print('contours길이:',len(contours))
                # if len(contours)==1:
                #     print(dice.item())
                #     i+=dice.item()
                #     k+=1
                k+=1
    print(i)
    total = i/k
    print('avg',total)
                # ###post Processing####
                # import cv2
                # import numpy as np
                # import os
                # import matplotlib.pyplot as plt
                # from glob import glob

                
                # kernel = np.ones((15,15),np.uint8)
                # kernel_3 = np.ones((7,7),np.uint8)

                # mask_C=pred.copy()
                # mask = cv2.morphologyEx(pred,cv2.MORPH_OPEN,kernel_3)
                # mask =cv2.dilate(mask, kernel, iterations=2)
                # mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
                # mask_cp = pred.copy()
                
                
                # ret, thr = cv2.threshold(mask,127,255,0)
                # contours,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                # print('contours길이:',len(contours))
                # if len(contours)==1:

                #     mask_cp = mask_C.copy()
                #     mask_C = cv2.morphologyEx(mask_C,cv2.MORPH_OPEN,kernel_3)
                #     mask_C = cv2.cvtColor(mask_C,cv2.COLOR_BGR2GRAY)
                #     ret, thr = cv2.threshold(mask_C,127,255,0)
                #     contours,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                    
                #     small_area=[]
                #     for cnt in contours:
                #         area = cv2.contourArea(cnt)
                #         area=int(area)
                #         if area <1000:
                #             print(area)
                #             small_area.append(area)
                #     if len(small_area) < 2: 
                #         cv2.imwrite('/home/vfuser/sungjoo/Resize_model/exp/plots_prob/post/'+str(i)+'.png',pred)    
                       
                #         k+=1
                # i+=1





                # cv2.imwrite('/home/vfuser/sungjoo/Resize_model/exp/plots_prob/img/'+str(i)+'.png',origin)
                # cv2.imwrite('/home/vfuser/sungjoo/Resize_model/exp/plots_prob/gt/'+str(i)+'.png',true)
                # cv2.imwrite('/home/vfuser/sungjoo/Resize_model/exp/plots_prob/pred/'+str(i)+'.png',pred)
                
                # pred = pred*255
                # true =true*255
                # hd_sc = obj_asd(pred, true)
                # if dice<1.0 and dice >0.92:
                    
                
    #             i+=dice
    #             j+=1
                    
    # print(j)
    # total=i/j
    # print('total:',total)

                # pred = cv2.resize(pred, (800,1200))
                # if dice > 0.90:
                # i +=1
def compute_pre_rec(gt,mask,mybins=np.arange(0,256)):

    if(len(gt.shape)<2 or len(mask.shape)<2):
        print("ERROR: gt or mask is not matrix!")
        exit()
    if(len(gt.shape)>2): # convert to one channel
        gt = gt[:,:,0]
    if(len(mask.shape)>2): # convert to one channel
        mask = mask[:,:,0]
    if(gt.shape!=mask.shape):
        print("ERROR: The shapes of gt and mask are different!")
        exit()

    gtNum = gt[gt>128].size # pixel number of ground truth foreground regions
    pp = mask[gt>128] # mask predicted pixel values in the ground truth foreground region
    nn = mask[gt<=128] # mask predicted pixel values in the ground truth bacground region

    pp_hist,pp_edges = np.histogram(pp,bins=mybins) #count pixel numbers with values in each interval [0,1),[1,2),...,[mybins[i],mybins[i+1]),...,[254,255)
    nn_hist,nn_edges = np.histogram(nn,bins=mybins)

    pp_hist_flip = np.flipud(pp_hist) # reverse the histogram to the following order: (255,254],...,(mybins[i+1],mybins[i]],...,(2,1],(1,0]
    nn_hist_flip = np.flipud(nn_hist)

    pp_hist_flip_cum = np.cumsum(pp_hist_flip) # accumulate the pixel number in intervals: (255,254],(255,253],...,(255,mybins[i]],...,(255,0]
    nn_hist_flip_cum = np.cumsum(nn_hist_flip)
    tn = int(((mask == 0) * (gt == 0)).sum())
    tp = int(((mask != 0) * (gt != 0)).sum())
    fp = int(((mask != 0) * (gt == 0)).sum())
    fn = int(((mask== 0) * (gt != 0)).sum())
    precision = pp_hist_flip_cum/(pp_hist_flip_cum + nn_hist_flip_cum+1e-8) #TP/(TP+FP)
    recall = pp_hist_flip_cum/(gtNum+1e-8) #TP/(TP+FN)
    acc = tp+tn/(tp+tn+fp+fn)
    specificity= tn/(fp+tn)

    precision[np.isnan(precision)]= 0.0
    recall[np.isnan(recall)] = 0.0

    return np.reshape(precision,(len(precision))),np.reshape(recall,(len(recall))),acc,specificity

def SamplewiseAccuracy(data):
    # Unpack data, assert shapes and get predictions
    _correct_samples = 0
    _total_samples = 0
    outputs, labels = data
    assert outputs.shape[0] == labels.shape[0]
    outputs = outputs.argmax(1)

    # Update numbers of correctly predicted and total samples
    _correct_samples += (outputs == labels).sum(dtype=torch.float)
    _total_samples += torch.numel(outputs)
    return _correct_samples / _total_samples


def MeanAccuracy(data):
    _total_class_samples = {}
    _correct_class_samples = {}
    accuracies = []
    # Unpack data, assert shapes and get predictions
    outputs, labels = data
    assert outputs.shape[0] == labels.shape[0]
    outputs = outputs.argmax(1)

    # Update correctly predicted and total precitions for each class in batch
    for label in torch.unique(labels):
        if not label in self._total_class_samples:
            _correct_class_samples[label] = 0
            _total_class_samples[label] = 0

        # Samples belonging to current class
        class_samples = labels == label

        # Correctly predicted samples and total samples for current class in batch
        correct_samples = (outputs[class_samples] == label).sum(dtype=torch.float)
        total_samples = class_samples.sum(dtype=torch.float)
        _correct_class_samples[label] += correct_samples
        _total_class_samples[label] += total_samples
        
    for label in _total_class_samples:
        correct_samples = _correct_class_samples[label]
        total_samples = _total_class_samples[label]
        accuracies.append(correct_samples / total_samples)
    return torch.mean(torch.tensor(accuracies))



def MeanIoU(data):
    # Unpack data, assert shapes and get predictions
    _class_intersection = {}
    _class_union = {}
    ious = []
    outputs, labels = data
    assert outputs.shape[0] == labels.shape[0]
    outputs = outputs.argmax(1)

    # Update intersection and union for each class in batch
    for label in torch.unique(labels):
        if not label in _class_intersection:
            _class_intersection[label] = 0
            _class_union[label] = 0

        # Intersection and union of current class
        intersection = (
            ((labels == label) & (outputs == label)).sum(dtype=torch.float).item()
        )
        union = (
            ((labels == label) | (outputs == label)).sum(dtype=torch.float).item()
        )
        _class_intersection[label] += intersection
        _class_union[label] += union
    for label in _class_intersection:
        total_intersection = _class_intersection[label]
        total_union = _class_union[label]
        ious.append(total_intersection / total_union)
    return torch.mean(torch.tensor(ious))

    def compute(self):
        ious = []
        for label in self._class_intersection:
            total_intersection = self._class_intersection[label]
            total_union = self._class_union[label]
            ious.append(total_intersection / total_union)
        return torch.mean(torch.tensor(ious))


def FrequencyWeightedIoU(data):
    # Unpack data, assert shapes and get predictions
    _class_intersection = {}
    _class_union = {}
    _class_samples = {}
    _total_samples = 0
    ious = []
    outputs, labels = data
    assert outputs.shape[0] == labels.shape[0]
    outputs = outputs.argmax(1)

    # Update intersection, union, class and total samples
    for label in torch.unique(labels):
        if not label in _class_intersection:
            _class_intersection[label] = 0
            _class_union[label] = 0
            _class_samples[label] = 0

        # Samples belonging to current class
        class_samples = labels == label

        # Total samples, class samples, and intersection and union of current class
        _total_samples += class_samples.sum(dtype=torch.float).item()
        _class_samples[label] += class_samples.sum(dtype=torch.float).item()
        intersection = (
            ((labels == label) & (outputs == label)).sum(dtype=torch.float).item()
        )
        union = (
            ((labels == label) | (outputs == label)).sum(dtype=torch.float).item()
        )
        _class_intersection[label] += intersection
        _class_union[label] += union
    for label in _class_intersection:
        total_samples = _total_samples
        class_samples = _class_samples[label]
        class_intersection = _class_intersection[label]
        class_union = _class_union[label]
        ious.append(class_samples * class_intersection / class_union)
    return torch.tensor(ious).sum().item() / total_samples

def euclidean_dis(y_pred,y_true):
    return torch.sqrt(torch.sum(torch.square(y_pred-y_true)))
