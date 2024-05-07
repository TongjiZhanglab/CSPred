import numpy as np
import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader

def get_sid_in_vocab(vocab, samples):
    """
    parameter vocab: vocab of samples, index of samples start with 1.
    """
    
    return np.array([vocab[i] for i in samples])


def create_dataloader(vocab, samples, matrix, batch_size=32, shuffle=True):
    sid = get_sid_in_vocab(vocab, samples)

    dataset = Dataset.from_dict({'matrix': matrix,
                                 'samples':torch.tensor(sid).repeat(matrix.shape[0]).reshape(matrix.shape)})
    dataset_dict = DatasetDict({'data': dataset})
    dataset_dict.set_format("torch")
    dataloader = DataLoader(dataset_dict['data'], shuffle=shuffle, batch_size=batch_size)
    
    return dataloader
