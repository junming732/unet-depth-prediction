import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOv3Depth(nn.Module):
    def __init__(self, output_size=(64, 64)):
        super().__init__()
        self.output_size = output_size

        # --- 1. The Backbone ---
        print("Loading DINOv3 (DINOv2) backbone...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)

        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.embed_dim = self.backbone.embed_dim

        # --- 2. The Decoder ---
        # We need 4 upsamples to go from feature map (4x4) to Output (64x64)
        # 4 -> 8 -> 16 -> 32 -> 64
        self.decoder = nn.Sequential(
            nn.Conv2d(self.embed_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 4 -> 8

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 8 -> 16

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 16 -> 32

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 32 -> 64

            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        # x input is (B, 3, 64, 64)
        B, C, H, W = x.shape

        # --- THE FIX ---
        # DINO needs multiples of 14.
        # Closest to 64 is 56 (14x4) or 70 (14x5).
        # We choose 56 because our decoder does 16x upsampling (4 -> 64).
        # If we chose 70, we'd get 5x5 features -> 80x80 output (mismatch).
        x_dino = F.interpolate(x, size=(56, 56), mode='bilinear', align_corners=False)

        # 1. Forward pass through DINO
        with torch.no_grad():
            features_dict = self.backbone.forward_features(x_dino)
            patch_tokens = features_dict['x_norm_patchtokens'] # (B, 16, C) because 4*4=16 patches

            # Reshape tokens: (B, 16, 1024) -> (B, 1024, 4, 4)
            # We hardcode 4x4 because we forced input to 56x56
            feature_map = patch_tokens.permute(0, 2, 1).reshape(B, self.embed_dim, 4, 4)

        # 2. Decode
        # 4x4 -> 64x64
        depth = self.decoder(feature_map)

        # 3. Safety Resize (just in case)
        if depth.shape[-2:] != self.output_size:
            depth = F.interpolate(depth, size=self.output_size, mode='bilinear', align_corners=False)

        return depth