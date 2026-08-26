# tune_m3d_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from visual_encoder_module import VisionEncoder
from text_encoder_module import TextEncoder
from classifier import Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TuneM3D(nn.Module):
    """
    Unified model integrating VisualEncoder and TextEncoder,
    with cross-attention, MMSE prediction, CLIP-style loss, and classification head.
    """

    def __init__(self, pretrained_visual_path, clip_model=None):
        super().__init__()

        # Encoders
        self.M3D = VisionEncoder(pretrained_visual_path)
        self.text_encoder = TextEncoder(
            pretrained_model=text_model,
            hidden_size=text_hidden_size,
            output_size=768,
            prompt_length=text_prompt_length
        )
        self.clip_model = clip_model.float() if clip_model is not None else None

        # Linear projection for text features
        #self.linear_layer = nn.Linear(1024, 768)

        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=8,
            batch_first=True
        )

        # Classification head
        self.fc = Classifier(latent_size=768, inter_num_ch=64)

        # CLIP logit scale
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

        # MMSE loss weighting
        self.alpha = 0.5

    # -------------------------
    # 1. Medical report generation
    # -------------------------
    def generate_medical_report(self, Hippocampus, Ventricles, WholeBrain, Entorhinal, Fusiform, MidTemp, labels):
        """
        Generate structured medical reports from MRI-derived biomarkers and labels.
        """
        def format_value(value):
            return f"{value:.2f}" if not torch.isnan(value) else "Data unavailable"

        # Label mapping
        label_mapping = {2: "A photo of AD", 1: "A photo of MCI", 0: "A photo of NC"}

        reports = []
        for i in range(Hippocampus.shape[0]):
            label_text = label_mapping.get(labels[i].item(), "Unknown label")
            report = (
                f"{label_text}: "
                f"Hippocampal Vol: {format_value(Hippocampus[i])}, "
                f"Ventricular size: {format_value(Ventricles[i])}, "
                f"Whole brain Vol: {format_value(WholeBrain[i])}, "
                f"Entorhinal Vol: {format_value(Entorhinal[i])}, "
                f"Fusiform Vol: {format_value(Fusiform[i])}, "
                f"Middle temporal Vol: {format_value(MidTemp[i])}"
            )
            reports.append(report)

        return reports

    # -------------------------
    # 2. Visual features
    # -------------------------
    def compute_visual_features(self, img):
        feats, predicted_mmse = self.M3D(img)
        cls_feat = feats[:, 0]
        mmse_feat = feats[:, 1]
        return feats, cls_feat, mmse_feat, predicted_mmse

    # -------------------------
    # 3. Text features
    # -------------------------
    def compute_text_features(self, reports):
        encoding = self.tokenizer(
            reports,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
    
        text_tokens, text_mask = self.text_encoder(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"]
        )
    
        return text_tokens, text_mask

    # -------------------------
    # 4. Cross-attention
    # -------------------------
    def apply_cross_attention(
        self,
        image_feats,
        text_tokens,
        text_mask
    ):
        adapted_image, _ = self.cross_attention(
            query=image_feats,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=~text_mask.bool()
        )
    
        adapted_text, _ = self.cross_attention(
            query=text_tokens,
            key=image_feats,
            value=image_feats
        )
    
        image_feats = image_feats + 0.01 * adapted_image
        text_tokens = text_tokens + adapted_text
    
        return image_feats, text_tokens

    # -------------------------
    # 5. Loss computations
    # -------------------------
    def compute_clip_loss(self, image_cls, text_features):
        logits_per_image = torch.matmul(image_cls, text_features.T) * self.logit_scale
        logits_per_text = torch.matmul(text_features, image_cls.T) * self.logit_scale
        clip_loss = (F.cross_entropy(logits_per_image, torch.arange(len(logits_per_image)).to(device)) +
                     F.cross_entropy(logits_per_text, torch.arange(len(logits_per_text)).to(device))) / 2.0
        return clip_loss, logits_per_image

    def compute_mmse_loss(self, predicted_mmse, MMSE):
        return F.mse_loss(predicted_mmse.squeeze(), MMSE)
      
    '''
    def compute_cls_loss(self, mmse_feat, cls_feat, labels):
        logits1 = self.fc(mmse_feat)
        logits2 = self.fc(cls_feat)
        avg_logits = (logits1 + logits2) / 2
        labels = labels.long()
        cls_loss = nn.CrossEntropyLoss()(avg_logits, labels)
        cls_logits = F.softmax(avg_logits, dim=1)
        return cls_loss, cls_logits
    '''

    # -------------------------
    # 6. Forward
    # -------------------------
    def forward(self, img, Hippocampus, Ventricles, WholeBrain, Entorhinal, Fusiform, MidTemp, MMSE, labels):
        # Visual representations and MMSE prediction
        image_feats, cls_feat, mmse_feat, predicted_mmse = \
            self.compute_visual_features(img)
        
        # Structured textual supervision
        reports = self.generate_medical_report(
            Hippocampus,
            Ventricles,
            WholeBrain,
            Entorhinal,
            Fusiform,
            MidTemp,
            labels
        )
        
        text_tokens, text_mask = self.compute_text_features(reports)
        
        # Cross-modal interaction
        image_feats, text_tokens = self.apply_cross_attention(
            image_feats,
            text_tokens,
            text_mask
        )
        
        # Representations after cross-attention
        cls_feat = image_feats[:, 0]
        mmse_feat = image_feats[:, 1]
        
        mask = text_mask.unsqueeze(-1).to(text_tokens.dtype)
        text_pooled = (
            (text_tokens * mask).sum(dim=1)
            / mask.sum(dim=1).clamp_min(1.0)
        )
        text_pooled = F.normalize(text_pooled, dim=-1)
        
        clip_loss, logits_per_image = self.compute_clip_loss(
            cls_feat,
            text_pooled
        )
        
        mmse_loss = self.compute_mmse_loss(
            predicted_mmse,
            MMSE
        )
        
        cls_loss, cls_logits = self.compute_cls_loss(
            mmse_feat,
            cls_feat,
            labels
        )
        
        total_loss = (
            clip_loss
            + self.alpha * mmse_loss
        )

        return {
            "logits_per_image": logits_per_image,
            "text_embedding": text_features,
            "img_embedding": cls_feat,
            #"CLS_logits": cls_logits,
            "loss": total_loss,
            "clip_loss": clip_loss,
            "mmse_loss": mmse_loss,
            #"CLS_loss": cls_loss,
            "predicted_MMSE": predicted_mmse,
            "reports": reports
        }
