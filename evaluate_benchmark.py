import torch
import argparse
import os
from torch.utils.data import DataLoader
from data import NYUDataset, rgb_data_transforms, depth_data_transforms
from torchvision import transforms
import torch.nn.functional as F

# Import Models
from tiny_unet import UNet
from model_dino import DINOv3Depth

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, required=True, choices=['unet', 'dinov3'], help='Which model to test')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to the .pth file')
args = parser.parse_args()

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = '/proj/uppmax2025-2-346/nobackup/private/junming/nyu_depth_v2/extracted_data'

# --- THE FIX: Resize to 224x224 (High Res) ---
dino_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # <--- WAS 64, NOW 224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize Model & Loader
if args.model_type == 'unet':
    # UNet keeps using 64x64 (Low Res)
    test_dataset = NYUDataset(DATA_PATH, 'test', rgb_transform=rgb_data_transforms, depth_transform=depth_data_transforms)
    model = UNet().to(device)

elif args.model_type == 'dinov3':
    from dataset import NYUDepthDataset
    # DINO uses 224x224 (High Res)
    test_dataset = NYUDepthDataset(DATA_PATH, split='test', transform=dino_transform)
    model = DINOv3Depth(output_size=(224, 224)).to(device)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load Weights
print(f"Loading {args.model_type} weights from {args.checkpoint}...")
state_dict = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# --- Metrics Calculation ---
def compute_metrics(output, target):
    if args.model_type == 'unet':
        pred = torch.exp(output)
        gt = torch.exp(target)
    else:
        # DINO output is already Log-Depth (from Scale Invariant Training)
        # So we exp() it to get meters.
        pred = torch.exp(output)
        gt = target # Loader gives Linear meters

    # DINO RESIZE FIX:
    # If prediction is 224x224 and GT is 480x640, resize GT to match prediction
    if pred.shape != gt.shape:
        gt = F.interpolate(gt, size=pred.shape[-2:], mode='bilinear', align_corners=False)

    # Avoid div by zero
    gt = gt.clamp(min=1e-3)
    pred = pred.clamp(min=1e-3)

    # RMSE (Linear Meters)
    rmse = torch.sqrt(torch.mean((gt - pred) ** 2))

    # Accuracy (Delta < 1.25)
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()

    return rmse.item(), a1.item()

# Run Test
print(f"Running evaluation for {args.model_type}...")
total_rmse = 0.0
total_a1 = 0.0

with torch.no_grad():
    for batch in test_loader:
        img = batch['image'].to(device)
        depth = batch['depth'].to(device)

        pred = model(img)

        rmse, a1 = compute_metrics(pred, depth)
        total_rmse += rmse
        total_a1 += a1

avg_rmse = total_rmse / len(test_loader)
avg_a1 = total_a1 / len(test_loader)

print("-" * 30)
print(f"FINAL RESULTS FOR {args.model_type.upper()}")
print(f"RMSE (Lower is better):    {avg_rmse:.4f}")
print(f"Accuracy (Higher is better): {avg_a1:.4f}")
print("-" * 30)