import torch
import torch.nn as nn
from torch.optim import AdamW
# from torch.utils.data import DataLoader
from ..language_model import ModelLM
from .optim_schedule import ScheduledOptim
import tqdm
import numpy as np
from ..preprocess import Preprocess
import logging
import os


# from torch.cuda.amp import autocast as autocast

class ModelTrainer(object):
    def __init__(self, fill_value, net, train_dataloader, test_dataloader = None,
                 lr = 1e-04, betas = (0.9, 0.999), weight_decay = 0.01, warmup_steps = 10000,
                 with_cuda = False, cuda_devices = None, use_ckp = False, ckp_path = None, ckp_state = None
                 ):
        """
        :parameter model: the model you want to train
        :paramrter train_dataloader: train dataset data loader
        :paramrter test_dataloader: test dataset data loader [can be None]
        :parameter lr: learning rate of optimizer, defaults to 1e-04
        :parameter betas: AdamW optimizer betas, defaults to (0.9, 0.999)
        :parameter weight_decay: AdamW optimizer weight decay parameter, defaults to 0.01
        :parameter warmup_steps: AdamW optimizer scheduler warmup steps parameter, defaults to 10000
        :parameter with_cuda: traning with cuda, defaults to False
        :parameter cuda_devices: cuda devices, defaults to None
        """
        
        super().__init__()
        self.lr = lr
        self.warmup_steps = warmup_steps
            
        # Setup cuda device for training, argument with_cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device(cuda_devices if cuda_condition else "cpu")

        # This model will be saved every epoch
        self.net = net
        # Initialize the Language Model, with model
        self.model = ModelLM(net).to(self.device)
        
        if use_ckp:
            net_ckp = torch.load(os.path.join(ckp_path, f"model_trained.model.ep{ckp_state}"))
            model_ckp = torch.load(os.path.join(ckp_path, f"model_trained.modelLM.ep{ckp_state}"))
            
            self.net = net_ckp
            self.model = ModelLM(net_ckp).to(self.device)
            self.model.mask_lm.load_state_dict(model_ckp.mask_lm.state_dict())

        # Preprocess object
        self.preprocess = Preprocess(fill_value)
        
        # Distributed GPU training if CUDA can detect more than 1 GPU
        # if with_cuda and torch.cuda.device_count() > 1:
        #     print("Using %d GPUS for BERT" % torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
        
        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        
        # Setting the AdamW optimizer with hyper-param
        self.optim = AdamW(self.model.parameters(), lr = lr, betas = betas, weight_decay = weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, init_lr = lr, n_warmup_steps = warmup_steps)
        
        # Using Mean Square Error Loss function as the foundation for predicting the masked expression
        self.criterion = nn.MSELoss(reduction = "sum")
        
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        
        
    def train(self, logfile, epoch, alpha):
        self.iteration(logfile, epoch, alpha, self.train_data)
        
        
    def test(self, logfile, epoch, alpha):
        self.iteration(logfile, epoch, alpha, self.test_data, train = False)
    
    
    def iteration(self, logfile, epoch, alpha, data_loader, train = True):
        """ 
        Loop over the data_loader for training.
        
        :parameter epoch: current epoch index
        :parameter data_loader: torch.utils.data.DataLoader for iteration
        :parameter train: boolean value of is train or test
        """
        
        str_code = "train" if train else "test"
        
        # Initialize the logging module
        logging.basicConfig(filename = logfile, level = logging.INFO, format = '%(asctime)s - %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')
        logging.info("This is a {} loop on epoch {}.".format(str_code, epoch))
        hyperparam_message = "LR: {}, WarmupSteps: {}".format(self.lr, self.warmup_steps)
        logging.info(hyperparam_message)
        
        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                            desc="EP_%s:%d" % (str_code, epoch),
                            total=len(data_loader),
                            bar_format="{l_bar}{r_bar}")
        
        step_loss = []
        step_lr = []
        step_acc = []
        step_corr = []
        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            # data = {key: value.to(self.device) for key, value in data.items()}
            
            # 0. preprocess
            filled_matrix, mtx, raw_to_filtered_map, nans_positions, masked_positions, samples = \
                self.preprocess.wrapper(data["matrix"], data["samples"])
            mtx = mtx.to(self.device)
            samples = samples.to(self.device)
            nans_positions = nans_positions.to(self.device)
            masked_positions = masked_positions.to(self.device)
            # raw_to_filtered_map = {key: value.to(self.device) for key, value in raw_to_filtered_map.items()}
            # nan_masked_positions = nan_masked_positions.to(self.device)
            
            # 1-1. forward the model
            if train:
                self.model.train()
                x, _, _, _ = self.model.forward(mtx, raw_to_filtered_map, nans_positions, masked_positions, samples, alpha)

            else:
                self.model.eval()
                with torch.no_grad():
                    x, _, _, _ = self.model.forward(mtx, raw_to_filtered_map, nans_positions, masked_positions, samples, alpha)

            # 1-2. delete caches
            del mtx, samples, nans_positions
            torch.cuda.empty_cache()
            
            # 2. modified MSELoss of predicting masked expression
            # masked_positions = nan_masked_positions == 1
            masked_positions = masked_positions.cpu().numpy()
            unmasked_lengths = np.sum(~masked_positions, axis = 1)
            max_length = unmasked_lengths.max()
            batch_size = len(filled_matrix)
            
            ## modified factor
            modified_factor = (self.net.total_samples_num - max_length + 1) * batch_size
            
            ## send masked_positions and filled_matrix into the device(GPU or cpu)
            masked_positions = torch.tensor(masked_positions).to(self.device)
            filled_matrix = filled_matrix.to(self.device)
            
            ## modified MSE Loss
            # print(modified_factor)
            loss = self.criterion(x[masked_positions], filled_matrix[masked_positions]) / modified_factor
            step_loss.append(loss.item())
            # print(step_loss)

            # 3-1. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), int(1e2))
                self.optim_schedule.step_and_update_lr()
            
            step_lr.append(self.optim_schedule._optimizer.param_groups[0]["lr"])
            
            # 3-2. check if weight matrix has been updated
            # print(self.net.filtered_emb.weight[0,:5])
            # print(self.model.mask_lm.linear.weight[0,:5])

            pred = x[masked_positions].clone().detach().cpu().numpy()
            true = filled_matrix[masked_positions].clone().detach().cpu().numpy()
            
            # 4-1. prediction accuracy
            # rmse = np.sqrt(mean_squared_error(true, pred))
            mae = abs(pred - true).mean()
            pred_acc = 1 - mae/true.mean()
            step_acc.append(pred_acc)
            
            # 4-2. prediction correlation
            pred_corr = np.corrcoef(pred, true)[0,1]
            step_corr.append(pred_corr)
            
            # Logging
            post_fix = {
            "epoch": epoch,
            "iter": i,
            "loss": loss.item(),
            "acc": pred_acc,
            "corr": pred_corr,
            "avg_loss": np.mean(step_loss),
            "avg_acc": np.mean(step_acc),
            "avg_corr": np.mean(step_corr)
            }

            # Convert the post_fix dictionary to a formatted string
            log_message = "Epoch: {}, Iter: {}, Loss: {:.4f}, Acc: {:.4f}, Corr: {:.4f}, Avg Loss: {:.4f}, Avg Acc: {:.4f}, Avg Corr: {:.4f}".format(
                post_fix["epoch"], post_fix["iter"], post_fix["loss"], post_fix["acc"], post_fix["corr"], post_fix["avg_loss"], post_fix["avg_acc"], post_fix["avg_corr"]
            )

            # Write the log message to the log file
            logging.info(log_message)
            logging.shutdown()
            
            # 6. delete caches
            del x, loss, masked_positions, filled_matrix
            torch.cuda.empty_cache()
        
        # print("EP_%d:%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter))
        print("EP_%d:%s, avg_loss=" % (epoch, str_code), np.mean(step_loss))
        self.step_loss = step_loss
        self.step_lr = step_lr
        self.step_acc = step_acc
        self.step_corr = step_corr
    
    def save(self, epoch, file_path="output/model_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        output_path2 = file_path + "LM.ep%d" % epoch

        torch.save(self.net.cpu(), output_path)
        torch.save(self.model.cpu(), output_path2)
        
        self.net.to(self.device)
        self.model.to(self.device)
        
        print("EP:%d Model Saved on:" % epoch, output_path)
        
        return output_path
    
            
            
            
            


            

            
            
        
        