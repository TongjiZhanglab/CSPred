### import packages ###
import sys
sys.path.append("./cspred/")
from cspred import *

import numpy as np
import json
import random
import os
import torch

import warnings
warnings.filterwarnings('ignore')

### set random seed ###
def set_seeds(seed_value=42, cuda_deterministic=False):
    """Set seeds for reproducibility."""
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

set_seeds(seed_value=42, cuda_deterministic=False)

### load data ###
activesite = np.load("data/activesite.npy").tolist()
activesite_in_aloop = np.load("data/activesite_in_aloop.npy").tolist()
activesite_not_in_aloop = np.load("data/activesite_not_in_aloop.npy").tolist()

kinases_cptac= np.load("data/kinases_cptac.npy").tolist()
kinases_all = np.load("data/kinases_all.npy").tolist()


with open("embs/t5_emb_cspred.json") as f:
    t5_emb=json.load(f)
with open("embs/psf_emb_ep10_cspred.json") as f:
    sample_emb=json.load(f)

not_in_aloop_weight = 4

### train ###
fewshot(activesite, activesite_not_in_aloop, 
        kinases_cptac, kinases_all,
        not_in_aloop_weight, t5_emb, sample_emb, 
        nfold = 5, min_delta = 0, patience = 30, batch_size = 64, 
        lr = 1e-04, weight_decay = 0.01, dropout_rate = 0.1,
        path = './', device = 'cuda:0')
