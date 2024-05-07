import numpy as np
import torch
import torch.nn as nn
from .embedding import Embedding
from .transformer import TransformerBlock
from .performer.performer_block import PerformerBlock
from .preprocess import Preprocess

class Model(nn.Module):
    def __init__(self, total_samples_num, token_num = 100, 
                 hidden_dim = 256, dec_dim = 256, 
                 n_layers = 8, dec_depth = 4, 
                 attn_heads = 4, dec_heads = 2,
                 dropout = 0.1, out = True):
        super().__init__()
        
        self.out = out
        # self.total_cells_num = total_cells_num
        self.total_samples_num = total_samples_num
        self.vocab_size = total_samples_num + 1 # <PAD>: 0
        
        self.hidden_dim = hidden_dim
        self.dec_dim = dec_dim
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.feed_forward_hidden = hidden_dim * 4
        self.dec_depth = dec_depth
        self.dec_heads = dec_heads
        
        self.embedding = Embedding(self.vocab_size, token_num, self.hidden_dim, dropout)
        self.filtered_emb = nn.Embedding(2, self.hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.attn_heads)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, attn_heads, hidden_dim * 4, dropout) for _ in range(n_layers)])
        self.performer_blocks = PerformerBlock(dec_dim, self.dec_depth, self.dec_heads, 
                                               int(dec_dim//self.dec_heads), self.total_samples_num, dropout)
        self.fc = nn.Linear(self.hidden_dim, self.dec_dim)
        # self.to_out =nn.Linear(dec_dim, 1)
        
        
    def forward(self, mtx, raw_to_filtered_map, nans_positions, masked_positions, samples, alpha):
        """
        :parameter mtx: a filtered and padded matrix, where expression of <PAD> is -100
        :parameter raw_to_filtered_map: a dictionary of raw to filtered indeices map
        :parameter nan_masked_positions: a matrix where 0 represents the NaNs and 1 represents the masked positions
        :parameter samples: a sample tensor where index of <PAD> is 0
        :parameter alpha: scaling mixture factor
        """
                
        # embedding
        # [batch_size, seq_len, hidden_dim]
        padding_value = -100
        x, exp_emb, bin_weights, sample_emb = self.embedding(samples, mtx, alpha, padding_value)
        
        # attetion mask, <PAD> = False
        # [batch_size, 1, seq_len, seq_len]
        atten_mask = (mtx != padding_value).unsqueeze(1).repeat(1, mtx.size(1), 1).unsqueeze(1) 
        
        # encoding
        for transformer in self.transformer_blocks:
            x = transformer(x, atten_mask)
        
        # extending
        nans_index = (torch.tensor([0])).to(self.filtered_emb.weight.device)
        masked_index = (torch.tensor([1])).to(self.filtered_emb.weight.device)
        nan_emb = self.filtered_emb(nans_index)
        masked_emb = self.filtered_emb(masked_index)
        x = self.extending(x, raw_to_filtered_map, nans_positions, masked_positions, nan_emb, masked_emb)
        x = self.fc(x)
        
        # decoding
        x = self.performer_blocks(x)
        # self.dec_emb = x
        
        # output
        # if self.out:
        #     x = self.to_out(x)
        #     x = x.squeeze()
        
        return x, exp_emb, bin_weights, sample_emb
      

    def extending(self, encoder_emb, raw_to_filtered_map, nans_positions, masked_positions, nan_emb, masked_emb):
        extended_emb = (torch.zeros(nans_positions.shape[0], nans_positions.shape[1], encoder_emb.size(2))).to(nan_emb.device)
        for (i, j), (m, n) in raw_to_filtered_map.items():
            extended_emb[i, j, :] = encoder_emb[m, n]

        # nans = torch.tensor(nan_masked_positions == 0)
        # masked = torch.tensor(nan_masked_positions == 1)

        extended_emb[nans_positions] = nan_emb
        extended_emb[masked_positions] = masked_emb
         
        return extended_emb
 