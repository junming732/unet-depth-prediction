import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# --- Benchmark Settings ---
# The UNet paper uses a tiny 64x64 output. We keep this to match the benchmark exactly.
output_height = 224  # Was 64
output_width = 224  # Was 64

# --- Transforms ---
class TransposeDepthInput(object):
    def __call__(self, depth):
        # 1. Load from Numpy (H, W) -> Tensor (1, H, W)
        depth = torch.from_numpy(depth).float().unsqueeze(0)

        # 2. Resize to benchmark size (64x64)
        depth = torch.nn.functional.interpolate(depth.unsqueeze(0),
                                                size=(output_height, output_width),
                                                mode='bilinear',
                                                align_corners=False)

        # 3. Log transform (Standard for Depth estimation to handle range)
        depth = torch.log(depth.squeeze(0))
        return depth

rgb_data_transforms = transforms.Compose([
    transforms.Resize((output_height, output_width)),
    transforms.ToTensor(),
])

depth_data_transforms = transforms.Compose([
    TransposeDepthInput(),
])

# --- Dataset Class ---
class NYUDataset(Dataset):
    def __init__(self, root_dir, split_type, rgb_transform=None, depth_transform=None):
        """
        root_dir: Path to the 'extracted_data' folder containing 'rgb' and 'depth'
        split_type: 'training', 'validation', or 'test'
        """
        self.root_dir = root_dir
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform

        # 1. Get all filenames (we rely on the 0000.jpg numbering we created earlier)
        # We look at the 'rgb' folder to count files
        self.files = sorted(os.listdir(os.path.join(root_dir, 'rgb')))

        # 2. Replicate the original Split Logic (0-1024 Train, 1024-1248 Val, Rest Test)
        if split_type == "training":
            self.files = self.files[0:1024]
        elif split_type == "validation":
            self.files = self.files[1024:1248]
        elif split_type == "test":
            self.files = self.files[1248:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Filename example: "0000.jpg"
        img_name = self.files[idx]
        file_id = os.path.splitext(img_name)[0] # "0000"

        # A. Load RGB
        rgb_path = os.path.join(self.root_dir, 'rgb', img_name)
        image = Image.open(rgb_path).convert('RGB')

        # B. Load Depth (Match the ID: 0000.jpg -> 0000.npy)
        depth_path = os.path.join(self.root_dir, 'depth', f"{file_id}.npy")
        depth = np.load(depth_path).astype(np.float32) # Shape: (480, 640)

        # C. Apply Transforms
        if self.rgb_transform:
            image = self.rgb_transform(image)

        if self.depth_transform:
            depth = self.depth_transform(depth)

        return {'image': image, 'depth': depth}