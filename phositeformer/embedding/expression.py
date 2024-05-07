import torch.nn as nn
from .auto_discretization import AutoDiscretizationBlock

class ExpressionEmbedding(nn.Module):
    def __init__(self, token_num, hidden_dim):
        super().__init__()
        self.discretize = AutoDiscretizationBlock(token_num, hidden_dim)
    
    def forward(self, matrix, alpha, padding_value):
        mask = (matrix != padding_value)
        masked_matrix = matrix * mask ## convert -1 to 0
        # masked_matrix = masked_matrix.unsqueeze(-1)

        output, bin_weights = self.discretize(masked_matrix, alpha)
        output = output * mask.unsqueeze(-1)
        
        return output, bin_weights