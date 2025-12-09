import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import numpy as np
import time

# Use PyTorch's built-in TensorBoard
from torch.utils.tensorboard import SummaryWriter

from data import NYUDataset, rgb_data_transforms, depth_data_transforms, output_height, output_width
from tiny_unet import UNet

# Training settings
parser = argparse.ArgumentParser(description='PyTorch depth map prediction example')
parser.add_argument('model_folder', type=str, default='trial', help='Folder to save the model')
parser.add_argument('--batch-size', type=int, default=32, help='Input batch size for training')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--log-interval', type=int, default=50, help='Batches to wait before logging')
args = parser.parse_args()

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Loaders ---
# Ensure your .mat file path is correct here
data_path = '/proj/uppmax2025-2-346/nobackup/private/junming/nyu_depth_v2/extracted_data'

train_loader = torch.utils.data.DataLoader(
    NYUDataset(data_path, 'training', rgb_transform=rgb_data_transforms, depth_transform=depth_data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    NYUDataset(data_path, 'validation', rgb_transform=rgb_data_transforms, depth_transform=depth_data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
)

# --- Model Setup ---
model = UNet().to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# --- TensorBoard Logger ---
writer = SummaryWriter(log_dir=f'./logs/{args.model_folder}')

def custom_loss_function(output, target):
    # Modern PyTorch Loss Implementation
    di = target - output
    n = (output_height * output_width)

    di2 = torch.pow(di, 2)
    first_term = torch.sum(di2, dim=(1,2,3)) / n

    second_term = 0.5 * torch.pow(torch.sum(di, dim=(1,2,3)), 2) / (n**2)

    loss = first_term - second_term
    return loss.mean()

# --- Metrics ---
def compute_errors(output, target):
    # Unlog the data to get back to real meters
    pred = torch.exp(output)
    gt = torch.exp(target)

    # Threshold metrics
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    # RMSE
    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    # RMSE log
    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    # Abs Relative Difference
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    return a1, a2, a3, rmse, rmse_log, abs_rel

# --- Training Loop ---
def train(epoch):
    model.train()
    running_loss = 0.0

    for batch_idx, sample in enumerate(train_loader):
        x = sample['image'].to(device)
        y = sample['depth'].to(device)

        optimizer.zero_grad()
        y_hat = model(x)
        loss = custom_loss_function(y_hat, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(x)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

            step = (epoch - 1) * len(train_loader) + batch_idx
            writer.add_scalar('Training/Loss', loss.item(), step)

    return running_loss / len(train_loader)

# --- Validation Loop ---
def validate(epoch):
    model.eval()
    val_loss = 0.0
    a1_sum, a2_sum, a3_sum = 0, 0, 0
    rmse_sum, rmse_log_sum, abs_rel_sum = 0, 0, 0

    with torch.no_grad():
        for sample in val_loader:
            x = sample['image'].to(device)
            y = sample['depth'].to(device)

            y_hat = model(x)
            loss = custom_loss_function(y_hat, y)
            val_loss += loss.item()

            # Metrics
            a1, a2, a3, rmse, rmse_log, abs_rel = compute_errors(y_hat, y)
            a1_sum += a1; a2_sum += a2; a3_sum += a3
            rmse_sum += rmse; rmse_log_sum += rmse_log; abs_rel_sum += abs_rel

    # Averaging
    num_batches = len(val_loader)
    val_loss /= num_batches
    print(f'\nValidation set: Average loss: {val_loss:.4f}, '
          f'delta1: {a1_sum/num_batches:.3f}, RMSE: {rmse_sum/num_batches:.3f}\n')

    # Log to TensorBoard
    writer.add_scalar('Validation/Loss', val_loss, epoch)
    writer.add_scalar('Validation/delta1', a1_sum/num_batches, epoch)
    writer.add_scalar('Validation/RMSE', rmse_sum/num_batches, epoch)

# --- Main Execution ---
folder_name = "models/" + args.model_folder
if not os.path.exists(folder_name): os.makedirs(folder_name, exist_ok=True)

print("********* Training the UNet Model **************")
for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    validate(epoch)

    # Save checkpoint
    torch.save(model.state_dict(), f"{folder_name}/model_{epoch}.pth")

writer.close()