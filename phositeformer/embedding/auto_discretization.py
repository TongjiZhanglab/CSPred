import torch
import torch.nn as nn

class AutoDiscretizationBlock(nn.Module):
    """
    AutoDis for numerical features

    :param h: number of bins
    :param dim: hidden dimension
    :param alf: parameter for controling the residual connection
    """
    def __init__(self, token_num, hidden_dim):
        super(AutoDiscretizationBlock, self).__init__()
        self.h = token_num
        self.dim = hidden_dim
        
        self.fc1 = nn.Linear(1, self.h)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(self.h, self.h)
        self.softmax = nn.Softmax(dim=-1)
        # self.meta_emb = nn.Parameter(torch.randn(self.dim, self.h), requires_grad=True)
        self.meta_emb = nn.Embedding(self.h, self.dim)
        
    def forward(self, X, alpha):
        shapes = X.shape                                        ## [batch, len]
        X = X.reshape(-1,1)                                     ## [batch*len, 1]
        X = X.to(self.fc1.weight.dtype)
        
        xh = self.relu(self.fc1(X))                             ## [batch*len, h]
        xh = self.fc2(xh) + alpha * xh                          ## [batch*len, h]
        xh = self.softmax(xh)                                   ## [batch*len, h]
        xh = xh.unsqueeze(1)                                    ## [batch*len, 1, h]
        
        y = torch.reshape(X, [-1, 1, 1])                        ## [batch*len, 1, 1]
        y = y.repeat(1, self.dim, self.h)                       ## [batch*len, d, h]
        ME = self.meta_emb.weight.t() * y                       ## [batch*len, d, h]
        
        autodis = torch.sum(ME * xh, axis=-1)                   ## [batch*len, d]
        autodis = autodis.reshape(shapes[0],shapes[1],self.dim) ## [batch, len, d]
        
        return autodis, xh
    





