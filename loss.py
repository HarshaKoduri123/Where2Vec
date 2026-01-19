import torch
import torch.nn.functional as F
from config import *

def build_type_mask(types, device):
    B = len(types)
    mask = torch.zeros(B, B, device=device)
    for i in range(B):
        for j in range(B):
            if types[i] == types[j]:
                mask[i, j] = 1
    return mask

def type_aware_clip_loss(zi, zt, type_mask):
    logits = zi @ zt.T / TEMP
    weights = torch.ones_like(logits) - ALPHA_TYPE * type_mask
    weights.fill_diagonal_(1.0)
    logits = logits * weights
    labels = torch.arange(len(zi), device=zi.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) +
        F.cross_entropy(logits.T, labels)
    )
