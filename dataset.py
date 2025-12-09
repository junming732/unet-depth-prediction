import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms

class NYUDepthDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        split (str): 'train', 'val', or 'test'.
                     We use the EXACT indices from the UNet benchmark data.py.
        """
        self.root_dir = root_dir
        self.transform = transform

        # 1. Get all sorted filenames (0000.jpg, 0001.jpg, ...)
        # Sorting ensures that index 0 is always the same image.
        self.rgb_files = sorted(os.listdir(os.path.join(root_dir, 'rgb')))

        # 2. Strict Index Splitting (Matching UNet data.py logic)
        # Train: 0-1024 | Val: 1024-1248 | Test: 1248-End
        if split == 'train':
            self.indices = range(0, 1024)
        elif split == 'val':
            self.indices = range(1024, 1248)
        elif split == 'test':
            self.indices = range(1248, len(self.rgb_files))
        else:
            raise ValueError(f"Unknown split: {split}")

        # Filter files list to only include this split
        self.rgb_files = [self.rgb_files[i] for i in self.indices]

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        # A. Load RGB
        img_name = self.rgb_files[idx]
        file_id = os.path.splitext(img_name)[0] # e.g. "0000"

        rgb_path = os.path.join(self.root_dir, 'rgb', img_name)
        image = Image.open(rgb_path).convert('RGB')

        # B. Load Depth (Match ID)
        depth_path = os.path.join(self.root_dir, 'depth', f"{file_id}.npy")
        depth = np.load(depth_path).astype(np.float32)

        # C. Transforms (DINOv3 specific)
        if self.transform:
            image = self.transform(image)

        # Prepare Depth Tensor (1, H, W)
        depth = torch.from_numpy(depth).unsqueeze(0)

        return {'image': image, 'depth': depth}