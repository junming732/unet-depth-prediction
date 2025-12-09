import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOv3Depth(nn.Module):
    def __init__(self, output_size=(224, 224)):
        super().__init__()
        self.output_size = output_size

        # 1. Backbone
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.embed_dim = self.backbone.embed_dim

        # 2. Decoder (16x16 -> 224x224)
        # We need to upsample 16 -> 32 -> 64 -> 112 -> 224?
        # Actually 224/14 = 16 patches.
        # 16 -> 32 -> 64 -> 128 -> 256 (Too big!)
        # Let's do 3 upsamples (16->32->64->128) then Resize, or 4 upsamples with smaller kernels.
        # Simplest robust way: 4 blocks of 2x upsampling with a final resize.

        self.decoder = nn.Sequential(
            nn.Conv2d(self.embed_dim, 512, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 32

            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 64

            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 128

            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=1.75, mode='bilinear', align_corners=False), # 128 -> 224

            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        # Input is now (B, 3, 224, 224) - Native DINO size!
        B, C, H, W = x.shape

        with torch.no_grad():
            features_dict = self.backbone.forward_features(x)
            patch_tokens = features_dict['x_norm_patchtokens']
            # (B, 256, 1024) -> (B, 1024, 16, 16)
            feature_map = patch_tokens.permute(0, 2, 1).reshape(B, self.embed_dim, 16, 16)

        depth = self.decoder(feature_map)

        # Force exact match
        if depth.shape[-1] != 224:
            depth = F.interpolate(depth, size=(224, 224), mode='bilinear', align_corners=False)

        return depth