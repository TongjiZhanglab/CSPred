import torch
import numpy as np

class Preprocess(object):
    def __init__(self, fill_value, mask_ratio = 0.15):
        super().__init__()
        self.fill_value = fill_value
        self.mask_ratio = mask_ratio
        
        
    def wrapper(self, matrix, sid):
        ## 0. NaN filling
        nans_positions = matrix == self.fill_value
        filled_matrix = matrix.clone()

        ## 1. masking
        mtx, masked_positions = self.random_mask(filled_matrix, self.fill_value, self.mask_ratio)
            
        ## 2. filtering and padding
        mtx, raw_to_filtered_map = self.filter_and_padding(mtx, self.fill_value)
        
        ## 3. samples tokenization
        samples = self.sample_tokenization(mtx, raw_to_filtered_map, sid)
        
        self.raw_mtx = filled_matrix
        self.filtered_mtx = mtx
        self.map = raw_to_filtered_map
        self.nans_positions = nans_positions
        self.masked_positions = masked_positions
        self.sample_tokens = samples
        
        return filled_matrix, mtx, raw_to_filtered_map, nans_positions, masked_positions, samples
        

    def random_mask(self, filled_matrix, fill_value, mask_ratio):
        mtx = filled_matrix.clone()
        masked_positions = torch.zeros_like(mtx, dtype=torch.bool)

        # for nrow in range(mtx.size(0)):
        na_positions = torch.nonzero(mtx == fill_value)
        non_na_positions = torch.nonzero(mtx != fill_value)
        # na_positions = torch.nonzero(mtx[nrow,] == fill_value)
        # non_na_positions = torch.nonzero(mtx[nrow,] != fill_value)
        
        masked_non_na = non_na_positions[np.random.choice(range(len(non_na_positions)), size = int(len(non_na_positions) * mask_ratio), replace = False)]

        if len(na_positions) != 0:
            ## ensure an equal count of mask between NaN and non-NaN values
            ## there are more NaN values in phosphoproteome datasets
            pct = len(non_na_positions) / len(na_positions) 

            is_replace = False if len(na_positions) >= int(len(masked_non_na) * pct) else True
            masked_na = na_positions[np.random.choice(range(len(na_positions)), size = int(len(masked_non_na) * pct), replace = is_replace)]

        else:
            masked_na = na_positions[np.random.choice(range(len(na_positions)), size = 0)]

        masked = torch.cat([masked_na, masked_non_na], dim = 0)
        masked_positions[masked[:,0], masked[:,1]] = True
        # masked_positions[nrow, masked] = True

        ## fill masked positions with the -100
        mtx[masked_positions] = -100
        
        return mtx, masked_positions


    def filter_and_padding(self, mtx, fill_value):        
        ## NaN and masked positons
        mask = (mtx == fill_value) | (mtx == torch.tensor(-100))

        raw_to_filtered_map = {}
        filtered_matrix = np.full((mtx.size(0), mtx.size(1)), -100).astype(np.float64)
        filtered_matrix = torch.tensor(filtered_matrix)
        m = 0
        for i in range(mtx.size(0)):
            n = 0
            for j in range(mtx.size(1)):
                if not mask[i, j]:
                    filtered_matrix[m, n] = mtx[i, j].item()
                    raw_to_filtered_map[tuple([i,j])]=tuple([m,n])
                    n += 1
            m += 1

        filtered_matrix = filtered_matrix[:,:(filtered_matrix != -100).sum(axis=1).max().item()]
        
        return filtered_matrix, raw_to_filtered_map
    
    
    def sample_tokenization(self, mtx, raw_to_filtered_map, sid):
        samples = torch.zeros(mtx.size())
        for key,value in raw_to_filtered_map.items():
            samples[tuple(value)] = sid[key[0]][key[1]]
        samples = samples.int()
        
        return samples