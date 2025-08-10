import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

def load_image(path):
    return Image.open(path).convert("RGB")

def default_transform():
    return T.Compose([
        T.Resize((384, 384)),
        T.ToTensor()
    ])

class ReflectionRemovalDataset(Dataset):
    def __init__(self, root_dir, setting=1, transform=None):
        self.root_dir = root_dir
        self.transform = transform if transform else default_transform()
        self.setting = setting

        self.T_paths = sorted([os.path.join(root_dir, 'T', f) for f in os.listdir(os.path.join(root_dir, 'T'))])
        self.R_paths = sorted([os.path.join(root_dir, 'R', f) for f in os.listdir(os.path.join(root_dir, 'R'))])

    def __len__(self):
        return len(self.T_paths)

    def __getitem__(self, idx):
        T_img = load_image(self.T_paths[idx])
        R_img = load_image(self.R_paths[idx])

        T_tensor = self.transform(T_img)
        R_tensor = self.transform(R_img)

        if self.setting in [1, 3]:
            R_gt = R_tensor
        elif self.setting == 2:
            R_gt = None  # M - T will be used

        M_tensor = T_tensor + R_tensor
        M_tensor = torch.clamp(M_tensor, 0, 1)

        sample = {
            'T': T_tensor,
            'R': R_gt,
            'M': M_tensor
        }

        return sample
