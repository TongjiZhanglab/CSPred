### import packages ###
import sys
sys.path.append("./phositeformer/")
from phositeformer import *

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import json
import random
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
matrix_train = torch.load('data/train_data.pt')
matrix_val = torch.load('data/valid_data.pt')
matrix = torch.concat([matrix_train, matrix_val])

with open("data/vocab.json","r") as f:
    vocab=json.load(f)
samples = list(vocab.keys())

### create dataloader ###
train_dataloader = create_dataloader(vocab, samples, matrix_train, batch_size=32)
val_dataloader = create_dataloader(vocab, samples, matrix_val, batch_size=32,shuffle=False)

### model initilization ###
model = Model(total_samples_num = matrix.size(1), token_num = 100, 
              hidden_dim = 256, dec_dim = 256, 
              n_layers = 4, dec_depth = 2, 
              attn_heads = 4, dec_heads = 2,
              dropout = 0.1, out = True)
model_pretrain = ModelTrainer(torch.min(matrix), model, train_dataloader, val_dataloader,
                 lr = 1e-04, betas = (0.9, 0.999), weight_decay = 0.01, warmup_steps = int(2264*0.15)*50,
                 with_cuda = True, cuda_devices = "cuda:1", use_ckp=False)

### train ###
OUTPUT_PATH = "./results/phositeformer/"
writer = SummaryWriter(OUTPUT_PATH + "log/")

train_step_lr = []
train_step_loss = []
train_step_acc = []
train_step_corr = []
val_epoch_loss = []
val_epoch_acc = []
val_epoch_corr = []

for epoch in range(50):
    ##########################
    ###                    ###
    ###  1. training loop  ###
    ###                    ###
    ##########################
    
    model_pretrain.train(logfile="./results/phositeformer/log/logfile.txt", epoch, alpha=0.5)
    
    num_step = len(model_pretrain.step_loss)
    for step in range(num_step):
        writer.add_scalar('Train/lr', model_pretrain.step_lr[step], epoch * num_step + step)
        writer.add_scalar('Train/loss', model_pretrain.step_loss[step], epoch * num_step + step)
        writer.add_scalar('Train/acc', model_pretrain.step_acc[step], epoch * num_step + step)
        writer.add_scalar('Train/corr', model_pretrain.step_corr[step], epoch * num_step + step)

    train_step_loss += model_pretrain.step_loss
    train_step_lr += model_pretrain.step_lr
    train_step_acc += model_pretrain.step_acc
    train_step_corr += model_pretrain.step_corr
    
    
    ##########################
    ###                    ###
    ### 2. validation loop ###
    ###                    ###
    ##########################
    
    model_pretrain.test(logfile="./results/phositeformer/log/logfile.txt", epoch, alpha=0.5)
    
    writer.add_scalar('Validation/loss', np.mean(model_pretrain.step_loss), epoch)
    writer.add_scalar('Validation/acc', np.mean(model_pretrain.step_acc), epoch)
    writer.add_scalar('Validation/corr', np.mean(model_pretrain.step_corr), epoch)
    
    val_epoch_loss.append(np.mean(model_pretrain.step_loss))
    val_epoch_acc.append(np.mean(model_pretrain.step_acc))
    val_epoch_corr.append(np.mean(model_pretrain.step_corr))
    
    
    ##########################
    ###                    ###
    ###   3. model  save   ###
    ###                    ###
    ##########################
    path = model_pretrain.save(epoch, OUTPUT_PATH + "model/model_trained.model")

### report save ###
np.save(OUTPUT_PATH + "report/train_step_lr.npy", train_step_lr)
np.save(OUTPUT_PATH + "report/train_step_loss.npy", train_step_loss)
np.save(OUTPUT_PATH + "report/train_step_acc.npy", train_step_acc)
np.save(OUTPUT_PATH + "report/train_step_corr.npy", train_step_corr)

np.save(OUTPUT_PATH + "report/val_epoch_loss.npy", val_epoch_loss)
np.save(OUTPUT_PATH + "report/val_epoch_acc.npy", val_epoch_acc)
np.save(OUTPUT_PATH + "report/val_epoch_corr.npy", val_epoch_corr)