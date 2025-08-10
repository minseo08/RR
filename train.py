from models.dual_unet import DualStreamUNet
from models.diffusion_utils import *
from losses.matching_loss import matching_loss
from losses.reflection_loss import total_rr_loss
from dataset.synthetic_generator import ReflectionRemovalDataset

import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm


def train_loop(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Schedules
    beta = get_beta_schedule(config["T"]).to(device)
    alpha, alpha_bar = get_alpha_schedule(beta)

    # Model
    model = DualStreamUNet(base_ch=config["base_ch"], num_res_blocks=config["res_blocks"])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Data
    dataset = ReflectionRemovalDataset(config["data_root"], mode=config["data_mode"], setting=config["setting"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    # Training loop
    model.train()
    for step in range(config["max_iters"]):
        for batch in loader:
            M = batch["M"].to(device)
            T = batch["T"].to(device)
            R = batch["R"].to(device)
            B = M.size(0)

            # Sample t step
            t = generate_timesteps(config["T"], B, device)

            # Forward process q(x_t|x_0)
            x_t_T, eps_T = q_sample(T, t, alpha_bar)
            x_t_R, eps_R = q_sample(R, t, alpha_bar)

            # Predict eps
            eps_T_hat, eps_R_hat = model(x_t_T, x_t_R, M)

            # Recover x0_hat and x_{t-1}_hat
            x0_hat_T = predict_x0_from_eps(x_t_T, eps_T_hat, t, alpha_bar)
            x0_hat_R = predict_x0_from_eps(x_t_R, eps_R_hat, t, alpha_bar)

            x_t_minus_1_T_hat = p_sample(x_t_T, eps_T_hat, t, beta, alpha, alpha_bar)
            x_t_minus_1_R_hat = p_sample(x_t_R, eps_R_hat, t, beta, alpha, alpha_bar)

            x_t_minus_1_T_gt = p_sample(x_t_T, eps_T, t, beta, alpha, alpha_bar)
            x_t_minus_1_R_gt = p_sample(x_t_R, eps_R, t, beta, alpha, alpha_bar)

            # Loss
            loss_match = matching_loss(x_t_minus_1_T_hat, x_t_minus_1_T_gt)
            loss_match += matching_loss(x_t_minus_1_R_hat, x_t_minus_1_R_gt)

            loss_rr = total_rr_loss(x0_hat_T, x0_hat_R, T, R, M)
            loss = loss_match + config["rr_lambda"] * loss_rr

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if step % config["log_every"] == 0:
            print(f"Step {step}: Loss={loss.item():.4f} | Match={loss_match.item():.4f} | RR={loss_rr.item():.4f}")

        if step % config["save_every"] == 0:
            torch.save(model.state_dict(), os.path.join(config["save_dir"], f"model_step{step}.pth"))
