import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os

from dataset import NYUDepthDataset
from model_dino import DINOv3Depth

# --- Configuration ---
# REDUCED BATCH SIZE to 16 because 224x224 takes 10x more memory than 64x64
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 15
DATA_PATH = "/proj/uppmax2025-2-346/nobackup/private/junming/nyu_depth_v2/extracted_data"
SAVE_DIR = "./checkpoints_highres"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Transforms (HIGH RES) ---
dino_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # <--- CHANGED FROM 64
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Data ---
print("Initializing Datasets...")
train_dataset = NYUDepthDataset(root_dir=DATA_PATH, split='train', transform=dino_transform)
val_dataset   = NYUDepthDataset(root_dir=DATA_PATH, split='val', transform=dino_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# --- Model ---
# Initialize with 224x224 output target
model = DINOv3Depth(output_size=(224, 224)).to(device)
optimizer = optim.AdamW(model.decoder.parameters(), lr=LR)

# --- Scale Invariant Loss (Dynamic Size) ---
def scale_invariant_loss(output, target):
    # output/target are already (B, 1, 224, 224)
    di = output - target

    # Calculate n dynamically (224*224 = 50176)
    n = output.shape[-1] * output.shape[-2]

    di2 = torch.pow(di, 2)

    first_term = torch.sum(di2, dim=(1,2,3)) / n
    second_term = 0.5 * torch.pow(torch.sum(di, dim=(1,2,3)), 2) / (n**2)

    loss = first_term - second_term
    return loss.mean()

# --- Training Loop ---
print("Starting High-Res Training...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for i, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        depths = batch['depth'].to(device)

        # Preprocess Targets (Resize to 224 + Log)
        depths = F.interpolate(depths, size=(224, 224), mode='bilinear', align_corners=False)
        depths = torch.log(depths.clamp(min=1e-3))

        optimizer.zero_grad()

        preds = model(images)

        loss = scale_invariant_loss(preds, depths)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            depths = batch['depth'].to(device)

            # Validation resize MUST match training resize
            depths = F.interpolate(depths, size=(224, 224), mode='bilinear', align_corners=False)
            depths = torch.log(depths.clamp(min=1e-3))

            preds = model(images)
            val_loss += scale_invariant_loss(preds, depths).item()

    avg_val = val_loss/len(val_loader)
    print(f"--- Epoch {epoch+1} Val Loss: {avg_val:.4f} ---")

    # Save Model
    torch.save(model.state_dict(), f"{SAVE_DIR}/dinov3_highres_epoch_{epoch+1}.pth")

print("Training Complete.")