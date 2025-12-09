import torch
from torch.utils.data import DataLoader
from data import NYUDataset, rgb_data_transforms, depth_data_transforms
from model import UNet  # (Or DINOv3Depth for your other model)
import argparse
import os

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

# 2. Point to your extracted data
# (Make sure this matches the path in your main.py)
DATA_PATH = '/proj/uppmax2025-2-346/nobackup/private/junming/nyu_depth_v2/extracted_data'

# 3. Load the TEST Split (The part the model has never seen)
test_dataset = NYUDataset(DATA_PATH, 'test',
                          rgb_transform=rgb_data_transforms,
                          depth_transform=depth_data_transforms)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 4. Load the Model
# Change this to 'DINOv3Depth()' when testing your DINO model
model = UNet().to(device)

# Load the weights you saved during training
# (Replace 'model_10.pth' with your best checkpoint)
checkpoint_path = "models/trial/model_10.pth"
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded checkpoint: {checkpoint_path}")
else:
    print("Error: Checkpoint not found!")
    exit()

model.eval()

# 5. The Exam (Calculate Metrics)
def compute_errors(output, target):
    # Unlog data to get meters
    pred = torch.exp(output)
    gt = torch.exp(target)

    # Threshold Accuracy (Higher is better)
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()

    # RMSE (Lower is better)
    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    return a1.item(), rmse.item()

print("\nRunning Final Test...")
total_a1 = 0.0
total_rmse = 0.0

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        images = batch['image'].to(device)
        depths = batch['depth'].to(device)

        preds = model(images)

        a1, rmse = compute_errors(preds, depths)
        total_a1 += a1
        total_rmse += rmse

# 6. Final Report
avg_a1 = total_a1 / len(test_loader)
avg_rmse = total_rmse / len(test_loader)

print("="*30)
print(f"FINAL TEST RESULTS")
print(f"Accuracy (delta < 1.25): {avg_a1:.4f}")
print(f"RMSE Error:              {avg_rmse:.4f}")
print("="*30)