import torch.nn as nn
import numpy as np
import torch

class SampleEmbedding(nn.Embedding):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__(vocab_size, hidden_dim, padding_idx=0) ## padding_idx不会参与更新

# class GeneEmbedding(nn.Module):
#     def __init__(self, hidden_dim):
#         super().__init__()
#         gene2vec_weight = np.load('../../ProtTrans/ProtT5-XL-UniRef50/new/site2vec_pretrain_128.npy')
#         gene2vec_weight = np.concatenate((np.zeros((1, gene2vec_weight.shape[1])),gene2vec_weight), axis=0)
#         gene2vec_weight = torch.from_numpy(gene2vec_weight).type(torch.float32)
        
#         self.emb = nn.Embedding.from_pretrained(gene2vec_weight)
#         self.emb.weight.requires_grad = True

        
#     def forward(self, genes):
#         x = self.emb(genes)
#         return x
