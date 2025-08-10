import torch
import torch.nn.functional as F

def gradient_loss(pred, target):
    def get_grad(x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy

    dx_pred, dy_pred = get_grad(pred)
    dx_target, dy_target = get_grad(target)
    return F.l1_loss(dx_pred, dx_target) + F.l1_loss(dy_pred, dy_target)

def exclusion_loss(T, R):
    return torch.mean(torch.abs(T * R))

def pixel_loss(pred, target):
    return F.l1_loss(pred, target)

def color_loss(pred, target):
    mean_loss = F.l1_loss(pred.mean([2,3]), target.mean([2,3]))
    std_loss = F.l1_loss(pred.std([2,3]), target.std([2,3]))
    return mean_loss + std_loss

def consistency_loss(M, T, R):
    return F.l1_loss(M, T + R)

def total_rr_loss(T_hat, R_hat, T_gt, R_gt, M):
    loss = 0
    loss += gradient_loss(T_hat, T_gt) + gradient_loss(R_hat, R_gt)
    loss += exclusion_loss(T_hat, R_hat)
    loss += pixel_loss(T_hat, T_gt) + pixel_loss(R_hat, R_gt)
    loss += color_loss(T_hat, T_gt) + color_loss(R_hat, R_gt)
    loss += consistency_loss(M, T_hat, R_hat)
    return loss
