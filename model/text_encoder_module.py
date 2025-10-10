# text_encoder_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TextEncoder(nn.Module):
    """
    Text encoder using a pre-trained model with learnable prompts and projection.
    """

    def __init__(self, pretrained_model, hidden_size, prompt_length=30):
        """
        pretrained_model: pre-trained text encoder model
        hidden_size: embedding dimension
        prompt_length: number of learnable prompt tokens
        """
        super().__init__()
        self.model = pretrained_model
        self.hidden_size = hidden_size
        self.prompt_length = prompt_length

        # Freeze original model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Learnable prompts
        self.prompts = nn.Parameter(torch.zeros(1, prompt_length, hidden_size))
        nn.init.trunc_normal_(self.prompts, std=0.02)

        # Projection layer to joint embedding space
        self.proj_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: (B, L)
        attention_mask: (B, L)
        Returns:
            - projected text features (B, hidden_size)
        """
        B = input_ids.size(0)

        # Get input embeddings from the model
        embeddings = self.model.embeddings(input_ids=input_ids)  # (B, L, hidden_size)

        # Concatenate learnable prompts at the beginning
        prompts = self.prompts.expand(B, -1, -1)  # (B, prompt_len, hidden_size)
        concat_embeddings = torch.cat([prompts, embeddings], dim=1)  # (B, prompt_len + L, hidden_size)

        # Update attention mask
        if attention_mask is not None:
            prompt_mask = torch.ones(B, self.prompt_length, device=attention_mask.device)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # Pass through encoder
        encoder_outputs = self.model.encoder(concat_embeddings, attention_mask=attention_mask)
        sequence_output = encoder_outputs.last_hidden_state  # (B, prompt_len + L, hidden_size)

        # Pooling (mean over sequence)
        pooled = sequence_output.mean(dim=1)

        # Projection and normalization
        proj_features = self.proj_layer(pooled)
        proj_features = F.normalize(proj_features, dim=-1)

        return proj_features
