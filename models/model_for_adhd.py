import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .cbramod import CBraMod


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        # CBraMod backbone với cấu hình phù hợp cho ADHD EEG data
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )

        # Load pretrained weights nếu có
        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            print(f"Loading pretrained weights from: {param.foundation_dir}")
            self.backbone.load_state_dict(torch.load(param.foundation_dir, map_location=map_location))
            print("Pretrained weights loaded successfully!")
        
        self.backbone.proj_out = nn.Identity()

        # Note: The backbone expects inputs shaped (batch, channels, patch_num, patch_size)
        # For ADHD data we will provide patch_size=200 (repeat single value) and patch_num=1

        # Simple, size-agnostic classifier: average pooled features -> linear
        self.classifier = nn.Sequential(
            Rearrange('b c s d -> b d c s'),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(200, param.num_of_classes),
        )

        # Freeze backbone parameters if specified
        if param.frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        # Expected input x shape from DataLoader: (batch_size, channels, 1) or (batch_size, channels, patch_size)
        # Ensure we provide (batch, channels, patch_num, patch_size) with patch_size=200
        if x.dim() == 2:
            # (batch, channels) -> add time dim
            x = x.unsqueeze(-1)

        batch_size = x.size(0)
        ch = x.size(1)
        last_dim = x.size(2)

        # If last_dim != 200, expand/repeat/interpolate to length 200
        if last_dim != 200:
            if last_dim == 1:
                x = x.repeat(1, 1, 200)  # repeat the single value
            else:
                # interpolate along last dim
                x = x.unsqueeze(1).float()  # (b,1,ch,t)
                x = nn.functional.interpolate(x, size=200, mode='linear', align_corners=False)
                x = x.squeeze(1)

        # Now x shape is (batch, ch, 200)
        # Add patch_num dimension =1 -> (batch, ch, 1, 200)
        x = x.unsqueeze(2)

        # Pass through backbone
        x = self.backbone(x)  # Expected output shape (b, c, s, d)

        # Apply classifier
        x = self.classifier(x)

        return x