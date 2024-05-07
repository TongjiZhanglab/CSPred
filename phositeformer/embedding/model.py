import torch.nn as nn
from .sample import SampleEmbedding
from .expression import ExpressionEmbedding

class Embedding(nn.Module):
    def __init__(self, vocab_size, token_num, hidden_dim, dropout):
        super().__init__()
        self.sample = SampleEmbedding(vocab_size, hidden_dim)
        self.expression = ExpressionEmbedding(token_num, hidden_dim)
        self.dropout = nn.Dropout(p = dropout)
        
    def forward(self, samples, matrix, alpha, padding_value):
        autodis, bin_weights = self.expression(matrix, alpha, padding_value)
        semb = self.sample(samples)
        x = semb + autodis
        return self.dropout(x), autodis, bin_weights, semb