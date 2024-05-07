import torch.nn as nn
from performer_pytorch import Performer
from .position import FixedPositionalEmbedding

class PerformerBlock(nn.Module):
    def __init__(self, dec_dim, depth, heads, dim_head, max_seq_len, emb_dropout):
        super().__init__()
        assert dec_dim % heads == 0
        
        self.dropout = nn.Dropout(emb_dropout)
        
        self.pos_emb = FixedPositionalEmbedding(dec_dim, max_seq_len)
        self.layer_pos_emb = FixedPositionalEmbedding(dim_head, max_seq_len)
        
        self.performer = Performer(dim=dec_dim, depth=depth, heads=heads, dim_head=dim_head)
        self.norm = nn.LayerNorm(dec_dim)
        
    def forward(self, x):
        x += self.pos_emb(x)
        x = self.dropout(x)
        
        # performer layers
        layer_pos_emb = self.layer_pos_emb(x)
        x = self.performer(x, pos_emb = layer_pos_emb)
        
        # norm
        x = self.norm(x)
        
        return x