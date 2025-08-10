import torch
from torchvision.utils import save_image
from models.dual_unet import DualStreamUNet
from models.diffusion_utils import get_beta_schedule, get_alpha_schedule, p_sample
from dataset import load_image, default_transform
import os
import argparse


def inference(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # schedule
    beta = get_beta_schedule(config["T"]).to(device)
    alpha, alpha_bar = get_alpha_schedule(beta)

    # model
    model = DualStreamUNet(base_ch=config["base_ch"], num_res_blocks=config["res_blocks"])
    model.load_state_dict(torch.load(config["ckpt_path"], map_location=device))
    model.eval().to(device)

    # input
    M = load_image(config["input_path"])
    M = default_transform()(M).unsqueeze(0).to(device)
    B, C, H, W = M.shape

    x_T_T = torch.randn_like(M)
    x_T_R = torch.randn_like(M)

    for t_val in reversed(range(config["T"])):
        t = torch.tensor([t_val] * B, device=device, dtype=torch.long)

        with torch.no_grad():
            eps_T, eps_R = model(x_T_T, x_T_R, M)
            x_T_T = p_sample(x_T_T, eps_T, t, beta, alpha, alpha_bar)
            x_T_R = p_sample(x_T_R, eps_R, t, beta, alpha, alpha_bar)

    save_image(x_T_T, os.path.join(config["save_dir"], "pred_T.png"))
    save_image(x_T_R, os.path.join(config["save_dir"], "pred_R.png"))
    save_image(M, os.path.join(config["save_dir"], "input_M.png"))
