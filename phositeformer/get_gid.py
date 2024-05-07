import numpy as np

def get_gid_in_vocab(vocab, genes):
    """
    parameter vocab: vocab of sites, index of sites start with 1.
    parameter genes: actually sites, e.g. Q9Y6R4:S1501.
    """
    
    return np.array([vocab[i] for i in genes])