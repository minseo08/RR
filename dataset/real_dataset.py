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

class RealReflectionDataset(Dataset):
    def __init__(self, root_dir, setting=1, transform=None):
        self.root_dir = root_dir
        self.transform = transform if transform else default_transform()
        self.setting = setting

        self.M_paths = sorted([os.path.join(root_dir, 'M', f) for f in os.listdir(os.path.join(root_dir, 'M'))])
        self.T_paths = sorted([os.path.join(root_dir, 'T', f) for f in os.listdir(os.path.join(root_dir, 'T'))])

        if setting == 3:
            self.R_paths = sorted([os.path.join(root_dir, 'R', f) for f in os.listdir(os.path.join(root_dir, 'R'))])
        else:
            self.R_paths = None

    def __len__(self):
        return len(self.M_paths)

    def __getitem__(self, idx):
        M_img = load_image(self.M_paths[idx])
        T_img = load_image(self.T_paths[idx])

        M_tensor = self.transform(M_img)
        T_tensor = self.transform(T_img)

        if self.setting == 3 and self.R_paths:
            R_img = load_image(self.R_paths[idx])
            R_tensor = self.transform(R_img)
        else:
            R_tensor = M_tensor - T_tensor

        sample = {
            'T': T_tensor,
            'R': R_tensor,
            'M': M_tensor
        }

        return sample
