import torch
import torch.nn.functional as F

def get_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

def get_alpha_schedule(beta):
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return alpha, alpha_bar

def generate_timesteps(T, batch_size, device):
    return torch.randint(low=0, high=T, size=(batch_size,), device=device)

def q_sample(x_0, t, alpha_bar):
    noise = torch.randn_like(x_0)
    a_bar = alpha_bar[t].view(-1, 1, 1, 1)
    x_t = torch.sqrt(a_bar) * x_0 + torch.sqrt(1 - a_bar) * noise
    return x_t, noise

def predict_x0_from_eps(x_t, eps, t, alpha_bar):
    a_bar = alpha_bar[t].view(-1, 1, 1, 1)
    return (x_t - torch.sqrt(1 - a_bar) * eps) / torch.sqrt(a_bar)

def p_sample(x_t, eps, t, beta, alpha, alpha_bar):
    b = beta[t].view(-1, 1, 1, 1)
    a = alpha[t].view(-1, 1, 1, 1)
    a_bar = alpha_bar[t].view(-1, 1, 1, 1)
    mean = (1 / torch.sqrt(a)) * (x_t - (b / torch.sqrt(1 - a_bar)) * eps)

    noise = torch.randn_like(x_t)
    mask = (t != 0).float().view(-1, 1, 1, 1)
    return mean + mask * torch.sqrt(b) * noise
