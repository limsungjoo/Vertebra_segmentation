import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import cv2
def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order).contiguous()
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)
def compute_per_channel_dice(input, target, epsilon=1e-5, ignore_index=None, weight=None):
    # assumes that input is a normalized probability
    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)

    # target = target.float()
    # Compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)

    denominator = (input + target).sum(-1)

    return (2. * intersect + epsilon) / (denominator + epsilon)
class DiceCoef(nn.Module):
    """Computes Dice Coefficient
    """

    def __init__(self, epsilon=1e-5, return_score_per_channel=False):
        super(DiceCoef, self).__init__()
        self.epsilon = epsilon
        self.return_score_per_channel = return_score_per_channel

    def forward(self, input, target):
        per_channel_dice = compute_per_channel_dice(input, target, epsilon=self.epsilon)

        if self.return_score_per_channel:
            return per_channel_dice
        else:
            return torch.mean(per_channel_dice)

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()

from datasets.dataset import VertebralDataset
def inference(args, model, test_save_path=None):
    index =0
    i=0
    test_dataset = VertebralDataset(is_Train=False, augmentation=False)
    testloader = DataLoader(test_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=1,
                                  pin_memory=True)
    
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, (img,mask) in tqdm(enumerate(testloader)):
        img= torch.Tensor(img).float()
        img = img.cuda()
        with torch.no_grad():
            pred = model(img)
            pred = pred[:,0,:,:]
            mask = mask[:,0,:,:]
            y = pred.sigmoid()
            # dice = DiceCoef(return_score_per_channel=False)(y, mask.cuda())
            
            # Convert to Binary
            zeros = torch.zeros(y.size())
            ones = torch.ones(y.size())
            y = y.cpu()
            y = torch.where(y > 0.9, ones, zeros) # threshold 0.99
            dice = DiceCoef(return_score_per_channel=False)(y, mask.cpu())

            for iw in range(len(img.cpu().numpy())):
                origin = img.cpu().numpy()[iw,:,:] 
                pred = y.cpu().numpy()[iw,:,:]
                true = mask.cpu().numpy()[iw,:,:]	
                # origin = img.cpu().numpy()[iw,0,:,:] 
                # pred = y.cpu().numpy()[iw,0,:,:]
                # true = mask.cpu().numpy()[iw,0,:,:]	
                pred=pred*255
                true = true*255
                origin=origin *255
                print(origin[0].shape)
                print(index+1)
                # pred = pred[:,:,np.newaxis]
                # origin = origin[0][:,:,np.newaxis]
                # cv2.imwrite('/data/workspace/vfuser/sungjoo/TransUnet/test/finaljpg/origin/'+str(index+1)+'.jpg',origin)
                # cv2.imwrite('/data/workspace/vfuser/sungjoo/TransUnet/test/finaljpg/pred/'+str(index+1)+'.png',pred)
                
                # plt.imshow(origin,'gray')
                # plt.imshow(pred,alpha=0.3,cmap='plasma')
                i+=dice.item()
                print(dice.item())
                # cv2.imwrite('/data/workspace/vfuser/sungjoo/TransUnet/test/Origin/'+str(index+1)+'.jpg',origin)
                # cv2.imwrite('/data/workspace/vfuser/sungjoo/TransUnet/test/Pred/'+str(index+1)+'.png',pred)
                index+=1
                   
    total = i/index
    print(index)
    logging.info('Testing performance in best val model: mean_dice : %f ' % (total))
    return "Testing Finished!"


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': VertebralDataset,
            'volume_path': '../data/Synapse/test_vol_h5',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 1,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    net.load_state_dict(torch.load('/data/workspace/vfuser/sungjoo/TransUnet/train/pth/1024/epoch_149.pth'))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = '../predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)


