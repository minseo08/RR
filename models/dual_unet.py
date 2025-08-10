import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class Encoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, num_res_blocks=2):
        super().__init__()
        self.initial = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(base_ch) for _ in range(num_res_blocks)])
        self.down = nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.initial(x))
        x = self.res_blocks(x)
        return F.relu(self.down(x))


class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch, num_res_blocks=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 4, stride=2, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(in_ch // 2) for _ in range(num_res_blocks)])
        self.final = nn.Conv2d(in_ch // 2, out_ch, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.up(x))
        x = self.res_blocks(x)
        return self.final(x)


class DualStreamUNet(nn.Module):
    def __init__(self, base_ch=64, num_res_blocks=2):
        super().__init__()
        self.encoder = Encoder(base_ch=base_ch, num_res_blocks=num_res_blocks)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 2 * 3, base_ch * 4, 3, padding=1),
            nn.ReLU(),
            *[ResidualBlock(base_ch * 4) for _ in range(num_res_blocks)]
        )

        self.decoder_T = Decoder(base_ch * 4, 3, num_res_blocks=num_res_blocks)
        self.decoder_R = Decoder(base_ch * 4, 3, num_res_blocks=num_res_blocks)

    def forward(self, x_t_T, x_t_R, M):
        f_T = self.encoder(x_t_T)
        f_R = self.encoder(x_t_R)
        f_M = self.encoder(M)

        f = torch.cat([f_T, f_R, f_M], dim=1)
        fused = self.bottleneck(f)

        eps_T = self.decoder_T(fused)
        eps_R = self.decoder_R(fused)
        return eps_T, eps_R
