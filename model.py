import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
from config import *

tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)

def mean_pool(hidden, mask):
    mask = mask.unsqueeze(-1).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

class SpectralAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.mlp(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * w

class SmallS2Encoder(nn.Module):
    def __init__(self, in_ch=4, emb_dim=IMG_EMB_DIM):
        super().__init__()
        self.attn = SpectralAttention(in_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, emb_dim)

    def forward(self, x):
        x = self.attn(x)
        z = self.proj(self.net(x).flatten(1))
        return F.normalize(z, dim=-1)

class Where2Vec(nn.Module):
    def __init__(self):
        super().__init__()

        base_text = AutoModel.from_pretrained(TEXT_MODEL)
        lora_cfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=["query", "key", "value"],
            bias="none"
        )

        self.text_encoder = get_peft_model(base_text, lora_cfg)
        self.text_proj = nn.Linear(
            self.text_encoder.config.hidden_size,
            IMG_EMB_DIM
        )

        self.img_encoder = SmallS2Encoder()

    def encode_text(self, texts):
        tok = tokenizer(texts, padding=True, truncation=True,
                        return_tensors="pt").to(DEVICE)
        out = self.text_encoder(**tok)
        pooled = mean_pool(out.last_hidden_state, tok["attention_mask"])
        return F.normalize(self.text_proj(pooled), dim=-1)

    def encode_image(self, images):
        return self.img_encoder(images.to(DEVICE))
