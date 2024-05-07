import torch
import torch.nn as nn
from torch.optim import AdamW
import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error
from .preprocess import Preprocess

# from torch.cuda.amp import autocast as autocast

class ModelTest(object):
    def __init__(self, model, test_dataloader, with_cuda = False, cuda_devices = None
                 ):
        """
        :parameter model: the model you want to train
        :paramrter test_dataloader: test dataset data loader [can be None]
        :parameter with_cuda: traning with cuda, defaults to False
        :parameter cuda_devices: cuda devices, defaults to None
        """
        
        super().__init__()
            
        # Setup cuda device for training, argument with_cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device(cuda_devices if cuda_condition else "cpu")
        
        # Save the model every epoch
        self.model = model.to(self.device)
        
        # Preprocess object
        self.preprocess = Preprocess()
        
        # Distributed GPU training if CUDA can detect more than 1 GPU
        # if with_cuda and torch.cuda.device_count() > 1:
        #     print("Using %d GPUS for BERT" % torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
        
        # Setting the test data loader
        self.test_data = test_dataloader
        
        # Using Mean Square Error Loss function as the foundation for predicting the masked expression
        self.criterion = nn.MSELoss(reduction = "sum")        
        
            
    def test(self, alpha):
        self.iteration(alpha, self.test_data)
    
    
    def iteration(self, alpha, data_loader):
        """ 
        Loop over the data_loader for training.
        
        :parameter epoch: current epoch index
        :parameter data_loader: torch.utils.data.DataLoader for iteration
        """
        
        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                            desc="Test done",
                            total=len(data_loader),
                            bar_format="{l_bar}{r_bar}")
        
        step_loss = []
        step_acc = []
        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            # data = {key: value.to(self.device) for key, value in data.items()}
            
            # 0. preprocess
            filled_matrix, mtx, raw_to_filtered_map, nan_masked_positions, genes = self.preprocess.wrapper(data["matrix"])
            mtx = mtx.to(self.device)
            genes = genes.to(self.device)
            # raw_to_filtered_map = {key: value.to(self.device) for key, value in raw_to_filtered_map.items()}
            # nan_masked_positions = nan_masked_positions.to(self.device)
            
            # 1. forward the model
            with torch.no_grad():
                x = self.model.forward(mtx, raw_to_filtered_map, nan_masked_positions, genes, alpha)
        
            # 2. delete caches
            del mtx, genes
            torch.cuda.empty_cache()
            
            # 3. modified MSELoss of predicting masked expression
            masked_positions = nan_masked_positions == 1
            unmasked_lengths = np.sum(~masked_positions, axis = 1)
            max_length = unmasked_lengths.max()
            batch_size = len(filled_matrix)
            
            ## modified factor
            modified_factor = (self.model.total_genes_num - max_length) * batch_size
            
            ## send masked_positions and filled_matrix into the device(GPU or cpu)
            masked_positions = torch.tensor(masked_positions).to(self.device)
            filled_matrix = filled_matrix.to(self.device)
            
            ## modified MSE Loss
            loss = self.criterion(x[masked_positions], filled_matrix[masked_positions]) / modified_factor
            step_loss.append(loss.item())
            
            # 4. prediction accuracy
            pred = x[masked_positions].clone().detach().cpu().numpy()
            true = filled_matrix[masked_positions].clone().detach().cpu().numpy()
            rmse = np.sqrt(mean_squared_error(true, pred))
            pred_acc = 1 - rmse/true.mean()
            step_acc.append(pred_acc)
            
            # 5. delete caches
            del x, loss, masked_positions, filled_matrix
            torch.cuda.empty_cache()
        
        self.step_loss = step_loss
        self.step_acc = step_acc