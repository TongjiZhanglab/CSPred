import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    "Compute scaled dot product attention."
    
    def forward(self, query, key, value, mask = None, dropout = None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1)) ## scaler, to make the model have a more stable gradient

        if mask is not None:
            # mask: attention mask, where <PAD> = 0
            scores = scores.masked_fill(mask == 0, -1e9) ## fill with an -Inf that will be 0 after softmax

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn