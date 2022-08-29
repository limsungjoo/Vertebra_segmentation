import torch
import numpy as np
from torch import nn

def dice_coef(y_pred, y_true):

    # Convert to Binary
    zeros = torch.zeros(y_pred.size())
    ones = torch.ones(y_pred.size())

    y_pred = y_pred.cpu()
    y_pred = torch.where(y_pred > 0.9, ones, zeros)

    if torch.cuda.is_available():
        y_pred = y_pred.cuda()

    y_true = y_true.cpu()
    y_true = torch.where(y_true > 0, ones, zeros)
    
    if torch.cuda.is_available():
        y_true = y_true.cuda()


    # Calculate Dice Coefficient Score
    smooth = 1e-5
    
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    intersection = (y_pred * y_true).sum()

    return (2 * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

def dice_coef_np(y_pred, y_true):
    # Calculate Dice Coefficient Score
    smooth = 1e-5
    
    y_pred = y_pred.reshape((-1))
    y_true = y_true.reshape((-1))

    intersection = np.sum(np.matmul(y_pred, y_true))

    return (2 * intersection + smooth) / (np.sum(y_pred) + np.sum(y_true) + smooth)


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