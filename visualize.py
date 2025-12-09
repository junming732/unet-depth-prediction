import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import os

# Import components
from dataset import NYUDepthDataset
from tiny_unet import UNet
from model_dino import DINOv3Depth

# --- CONFIGURATION ---
DATA_PATH = '/proj/uppmax2025-2-346/nobackup/private/junming/nyu_depth_v2/extracted_data'
UNET_PATH = 'models/unet_highres/model_10.pth'
# Point this to your BEST High-Res DINO checkpoint
DINO_PATH = 'checkpoints_highres/dinov3_highres_epoch_15.pth'
OUTPUT_FILE = 'comparison_highres.png'

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform: We load at HIGH RES (224) for DINO
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Test Data
test_dataset = NYUDepthDataset(root_dir=DATA_PATH, split='test', transform=transform)
loader = DataLoader(test_dataset, batch_size=5, shuffle=True)

# Load Models
print("Loading models...")
unet = UNet().to(device)
# If UNet trained on CPU/GPU mismatch, map_location fixes it
unet.load_state_dict(torch.load(UNET_PATH, map_location=device))
unet.eval()

# DINO: Init with 224x224
dino = DINOv3Depth(output_size=(224, 224)).to(device)
dino.load_state_dict(torch.load(DINO_PATH, map_location=device))
dino.eval()

def colorize(tensor, cmap='magma'):
    # Normalize for viz
    vmin, vmax = tensor.min(), tensor.max()
    norm = (tensor - vmin) / (vmax - vmin + 1e-5)
    return plt.get_cmap(cmap)(norm.squeeze().cpu().numpy())

# --- Run Inference ---
print("Generating High-Res Visualization...")
batch = next(iter(loader))
images = batch['image'].to(device) # (B, 3, 224, 224)
depths = batch['depth'].to(device) # (B, 1, 480, 640)

# Resize Ground Truth to 224 for comparison
gt_resized = F.interpolate(depths, size=(224, 224), mode='bilinear', align_corners=False)

with torch.no_grad():
    # 1. UNet Inference (The "Compat Mode")
    # UNet expects 64x64. We resize input down, run inference, resize output up.
    img_small = F.interpolate(images, size=(64, 64), mode='bilinear', align_corners=False)
    unet_out = unet(img_small) # Log Depth
    unet_pred = torch.exp(unet_out) # Linear Depth
    # Upscale back to 224 for display
    unet_pred = F.interpolate(unet_pred, size=(224, 224), mode='bilinear', align_corners=False)

    # 2. DINO Inference (Native High-Res)
    dino_out = dino(images) # (B, 1, 224, 224)
    # DINO was trained with Scale Invariant Loss, so output is Log Depth
    dino_pred = torch.exp(dino_out)

# --- Plotting ---
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))
# Cols: RGB | GT | UNet | DINO | DINO Error

for i in range(5):
    # RGB (Un-normalize)
    img = images[i].cpu().permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    axes[i, 0].imshow(np.clip(img, 0, 1))
    axes[i, 0].set_title("Input (224x224)")
    axes[i, 0].axis('off')

    # Ground Truth
    axes[i, 1].imshow(colorize(gt_resized[i]))
    axes[i, 1].set_title("Ground Truth")
    axes[i, 1].axis('off')

    # UNet
    axes[i, 2].imshow(colorize(unet_pred[i]))
    axes[i, 2].set_title("UNet (Upscaled)")
    axes[i, 2].axis('off')

    # DINO
    axes[i, 3].imshow(colorize(dino_pred[i]))
    axes[i, 3].set_title("DINOv3 (Native)")
    axes[i, 3].axis('off')

    # Error Map (DINO vs GT)
    diff = torch.abs(dino_pred[i] - gt_resized[i])
    axes[i, 4].imshow(colorize(diff, cmap='seismic'))
    axes[i, 4].set_title("DINO Error")
    axes[i, 4].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_FILE)
print(f"Saved to {OUTPUT_FILE}")