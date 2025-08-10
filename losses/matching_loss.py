import torch.nn.functional as F

def matching_loss(x1, x2):
    return F.l1_loss(x1, x2)
