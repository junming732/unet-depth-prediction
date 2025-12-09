import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os

# Import your custom modules
from dataset import NYUDepthDataset
from model_dino import DINOv3Depth

# --- Configuration ---
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 10
# Ensure this matches your path
DATA_PATH = "/proj/uppmax2025-2-346/nobackup/private/junming/nyu_depth_v2/extracted_data"
SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Setup Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Transforms for INPUT IMAGE (Resize to 64x64) ---
dino_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Load Data ---
print("Initializing Datasets...")
train_dataset = NYUDepthDataset(root_dir=DATA_PATH, split='train', transform=dino_transform)
val_dataset   = NYUDepthDataset(root_dir=DATA_PATH, split='val', transform=dino_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# --- Initialize Model ---
model = DINOv3Depth(output_size=(64, 64)).to(device)

# --- Loss & Optimizer ---
criterion = nn.L1Loss()
optimizer = optim.AdamW(model.decoder.parameters(), lr=LR)

# --- Training Loop ---
print("Starting Training...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for i, batch in enumerate(train_loader):
        images = batch['image'].to(device) # (B, 3, 64, 64)
        depths = batch['depth'].to(device) # (B, 1, 480, 640) <-- Too Big!

        # --- THE FIX: Preprocess Ground Truth ---
        # 1. Resize Ground Truth to match Model Output (64x64)
        depths = F.interpolate(depths, size=(64, 64), mode='bilinear', align_corners=False)

        # 2. Log Transform (To match UNet benchmark logic)
        # We clamp to 1e-3 to avoid log(0) errors if any holes exist
        depths = torch.log(depths.clamp(min=1e-3))

        optimizer.zero_grad()

        preds = model(images)
        loss = criterion(preds, depths)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            depths = batch['depth'].to(device)

            # Apply same fix to validation data
            depths = F.interpolate(depths, size=(64, 64), mode='bilinear', align_corners=False)
            depths = torch.log(depths.clamp(min=1e-3))

            preds = model(images)
            val_loss += criterion(preds, depths).item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"--- Epoch {epoch+1} Complete. Val Loss: {avg_val_loss:.4f} ---")

    # Save Model
    torch.save(model.state_dict(), f"{SAVE_DIR}/dinov3_da3_epoch_{epoch+1}.pth")

print("Training Complete.")