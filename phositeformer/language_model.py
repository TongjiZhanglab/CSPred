import torch.nn as nn

# from .model import Model


class ModelLM(nn.Module):
    def __init__(self, model):
        """
        :param model: model which should be trained
        """

        super().__init__()
        self.model = model
        self.mask_lm = MaskedLanguageModel(self.model.dec_dim)

    def forward(self, mtx, raw_to_filtered_map, nans_positions, masked_positions, samples, alpha):
        x, exp_emb, bin_weights, sample_emb = self.model(mtx, raw_to_filtered_map, nans_positions, masked_positions, samples, alpha)
        return self.mask_lm(x), exp_emb, bin_weights, sample_emb
   
    
class MaskedLanguageModel(nn.Module):
    """
    predicting origin expression from masked input
    """

    def __init__(self, hidden):
        """
        :param hidden: output size of model
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.linear(x).squeeze()
    