# model/loss.py

import torch.nn.functional as F

def mel_loss(predicted, target):
    return F.mse_loss(predicted, target)
