# text_encoder_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TextEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model,
        hidden_size,
        output_size=768,
        prompt_length=30
    ):
        super().__init__()

        self.model = pretrained_model
        self.prompt_length = prompt_length

        for param in self.model.parameters():
            param.requires_grad = False

        self.prompts = nn.Parameter(
            torch.zeros(1, prompt_length, hidden_size)
        )
        nn.init.trunc_normal_(self.prompts, std=0.02)

        self.proj_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        B = input_ids.size(0)

        embeddings = self.model.get_input_embeddings()(input_ids)
        prompts = self.prompts.expand(B, -1, -1)

        inputs_embeds = torch.cat(
            [prompts, embeddings],
            dim=1
        )

        prompt_mask = torch.ones(
            B,
            self.prompt_length,
            device=attention_mask.device,
            dtype=attention_mask.dtype
        )
        extended_mask = torch.cat(
            [prompt_mask, attention_mask],
            dim=1
        )

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask
        )

        text_tokens = self.proj_layer(
            outputs.last_hidden_state
        )
        text_tokens = F.normalize(text_tokens, dim=-1)

        return text_tokens, extended_mask
