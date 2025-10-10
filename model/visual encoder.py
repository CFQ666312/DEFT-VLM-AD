# vision_encoder_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig
from M3D_CLIP.modeling_m3d_clip import ViT

class VisionEncoder(nn.Module):
    """
    Vision encoder with VPT prompts, MMSE token, projection, and cross-attention.
    """

    def __init__(self, pretrained_path, prompt_length=20, k_layers=12, img_size=(128,128,128)):
        super().__init__()
        # 1. Load M3D-CLIP config
        self.config = AutoConfig.from_pretrained("GoodBaiBai88/M3D-CLIP", trust_remote_code=True)
        self.config.img_size = img_size
        self.config.gather_loss = False
        self.hidden_size = self.config.hidden_size

        # 2. Initialize ViT
        self.vision_encoder = ViT(
            in_channels=self.config.in_channels,
            img_size=self.config.img_size,
            patch_size=self.config.patch_size,
            hidden_size=self.config.hidden_size,
            mlp_dim=self.config.mlp_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            pos_embed=self.config.pos_embed,
            dropout_rate=self.config.dropout_rate,
            spatial_dims=self.config.spatial_dims,
            classification=True
        )

        # 3. Load pre-trained weights
        state_dict = torch.load(pretrained_path, map_location="cpu")
        self.vision_encoder.load_state_dict(state_dict)
        self.vision_encoder.eval()

        # 4. Freeze ViT parameters except CLS token
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        self.vision_encoder.cls_token.requires_grad = True

        # 5. MMSE token + head
        self.MMSE_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.MMSE_head = nn.Linear(self.hidden_size, 1)

        # 6. Projection layer
        self.proj_layer = nn.Linear(self.hidden_size, self.hidden_size)

        # 7. Visual prompt tuning (VPT)
        self.k_layers = k_layers
        self.prompt_length = prompt_length
        self.visual_prompts = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.prompt_length, self.hidden_size)) for _ in range(self.k_layers)]
        )

        # 8. Cross-attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=8)

    def forward(self, x):
        """
        Forward pass for MRI batch x (B, C, D, H, W)
        Returns:
            - projected features
            - MMSE prediction
            - cross-attention output
        """
        # ViT features
        feats, _ = self.vision_encoder(x)

        # Projection
        feats = self.proj_layer(feats)
        feats = F.normalize(feats, dim=-1)

        # MMSE prediction
        mmse_input = self.MMSE_token.expand(feats.size(0), -1, -1)
        mmse_pred = self.MMSE_head(mmse_input)

        # Cross-attention (self-attention for illustration)
        attn_output, _ = self.cross_attention(feats, feats, feats)

        return feats, mmse_pred, attn_output


if __name__ == "__main__":
    # Example usage
    dummy_input = torch.randn(2, 1, 128, 128, 128)
    model = VisionEncoder(pretrained_path="path/to/pretrained_ViT.bin")
    feats, mmse, attn = model(dummy_input)
    print("Feature shape:", feats.shape)
    print("MMSE prediction shape:", mmse.shape)
    print("Attention output shape:", attn.shape)
