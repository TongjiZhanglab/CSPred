import torch
import torch.nn as nn

class CSPred(nn.Module):
    def __init__(self, input_dim_seq, input_dim_phos, dropout_rate):
        super().__init__() 
        input_dim = input_dim_seq + input_dim_phos
        
        self.feature_upsampling = nn.Linear(input_dim_seq, input_dim)
        
        self.classification_head = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.LayerNorm(input_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim//2, input_dim//4),
            nn.LayerNorm(input_dim//4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim//4, input_dim//8),
            nn.LayerNorm(input_dim//8),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim//8, 1),
        )

        self.classification_head_phos = nn.Sequential(
            nn.Linear(input_dim_phos, input_dim_phos//2),
            nn.LayerNorm(input_dim_phos//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim_phos//2, input_dim_phos//2),
            nn.LayerNorm(input_dim_phos//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim_phos//2, 1),
        )
        
        
    def forward(self, x_seq=None, x_phos=None, both_feature=None):
        if both_feature is not None:
            both_feature = both_feature.to(int)
            
            x = torch.cat((x_seq, x_phos), dim=1)
            
            tmp_x = x[both_feature==0, :x_seq.size(1)].clone()
            tmp_x = self.feature_upsampling(tmp_x)
            
            x[both_feature==0,] = tmp_x

            x = self.classification_head(x)
            
        else:
            if x_phos is None:
                x = self.classification_head(x_seq)
            if x_seq is None:
                x = self.classification_head_phos(x_phos)
                
        return x

