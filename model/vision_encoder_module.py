# vision_encoder_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig
from M3D_CLIP.modeling_m3d_clip import ViT


class VisionEncoder(nn.Module):
    """
    Vision encoder with Visual Prompt Tuning (VPT), MMSE token, projection, and cross-attention.
    """

    def __init__(self, pretrained_path, prompt_length=20, k_layers=12, img_size=(128,128,128)):
        super().__init__()

        # load config
        self.config = config
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
        for name, param in self.vision_encoder.named_parameters():
            param.requires_grad = False
        self.vision_encoder.cls_token.requires_grad = True

        # 5. MMSE token + head
        self.MMSE_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        nn.init.trunc_normal_(self.MMSE_token, std=0.02)
        self.MMSE_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1)
        )

        # 6. Projection layer
        self.proj_layer = nn.Linear(self.hidden_size, self.hidden_size)

        # 7. Visual Prompt Tuning (VPT)
        self.k_layers = k_layers
        self.prompt_length = prompt_length
        self.visual_prompts = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.prompt_length, self.hidden_size))
            for _ in range(self.k_layers)
        ])
        for p in self.visual_prompts:
            nn.init.trunc_normal_(p, std=0.02)

        # 8. Cross-attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=8)

    def forward(self, x):
        """
        Forward pass for MRI batch (B, C, D, H, W)
        Returns:
            - projected features
            - MMSE prediction
            - self-attention output
        """
        vit = self.vision_encoder
        B = x.size(0)

        # Patch embedding
        x = vit.patch_embedding(x)

        cls_token = vit.cls_token.expand(B, -1, -1)
        mmse_token = self.MMSE_token.expand(B, -1, -1)
        
        # CLS and MMSE tokens precede the patch tokens
        x = torch.cat((cls_token, mmse_token, x), dim=1)

        # Transformer blocks with visual prompts
        for i, blk in enumerate(vit.blocks):
            if i < self.k_layers:
                prompt = self.visual_prompts[i].expand(B, -1, -1)
                if i == 0:
                    # first layer: insert after CLS and MMSE tokens
                    x = torch.cat((x[:, :2], prompt, x[:, 2:]), dim=1)
                else:
                    # subsequent layers: replace previous prompt
                    x = torch.cat((x[:, :2], prompt, x[:, 2 + self.prompt_length:]), dim=1)
            x = blk(x)

        # Layer normalization
        x = vit.norm(x)

        # Projection & normalization
        feats = self.proj_layer(x)
        feats = F.normalize(feats, dim=-1)

        # MMSE prediction (second token)
        mmse_pred = self.MMSE_head(feats[:, 1])

        # Self-attention (MultiheadAttention expects seq_len, batch, embed_dim)
        feats_t = feats.transpose(0, 1)
        attn_output, _ = self.cross_attention(feats_t, feats_t, feats_t)
        attn_output = attn_output.transpose(0, 1)

        return feats, mmse_pred


if __name__ == "__main__":
    dummy_input = torch.randn(2, 1, 128, 128, 128)
    model = VisionEncoder(pretrained_path="path/to/pretrained_ViT.bin")
    feats, mmse, attn = model(dummy_input)
    print("Feature shape:", feats.shape)
    print("MMSE prediction shape:", mmse.shape)
    print("Attention output shape:", attn.shape)
