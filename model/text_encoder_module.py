# text_encoder_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig

class TextEncoder(nn.Module):
    """
    Text encoder using a medical BERT model with learnable prompts and projection.
    """

    def __init__(self, pretrained_model_name="emilyalsentzer/Bio_ClinicalBERT", prompt_length=30):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.hidden_size = self.config.hidden_size

        # Load pre-trained medical BERT
        self.text_encoder = AutoModel.from_pretrained(pretrained_model_name)

        # Freeze original BERT parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Learnable prompts
        self.prompt_length = prompt_length
        self.text_prompts = nn.Parameter(torch.zeros(1, self.prompt_length, self.hidden_size))

        # Projection layer to joint embedding space
        self.proj_layer = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass
        input_ids: (B, L) token ids
        attention_mask: (B, L)
        Returns:
            - projected text features
        """
        B = input_ids.size(0)

        # Expand prompts to batch
        prompts = self.text_prompts.expand(B, -1, -1)

        # Encode input
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # (B, L, hidden_size)

        # Concatenate prompts at the beginning
        concat_embeddings = torch.cat([prompts, token_embeddings], dim=1)  # (B, prompt_len + L, hidden_size)

        # Mean pooling over sequence
        pooled = concat_embeddings.mean(dim=1)

        # Projection
        proj_features = self.proj_layer(pooled)
        proj_features = F.normalize(proj_features, dim=-1)

        return proj_features


if __name__ == "__main__":
    # Demo usage
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    texts = ["Patient shows normal hippocampal volume.", "MRI indicates enlarged ventricles in AD patient."]
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    model = TextEncoder(pretrained_model_name="emilyalsentzer/Bio_ClinicalBERT")
    features = model(encodings["input_ids"], encodings["attention_mask"])
    print("Text features shape:", features.shape)  # (B, hidden_size)
