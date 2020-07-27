import numpy as np
import torch
import os
from torch.utils.data import DataLoader, TensorDataset


def split_trainval(root_path, split=0.2):
    images = np.load(os.path.join(root_path, 'xtrain.npy'))
    labels = np.load(os.path.join(root_path, 'ytrain.npy'))

    idxes = np.arange(0, labels.shape[0])
    np.random.seed(322)
    np.random.shuffle(idxes)

    val_idx = int(labels.shape[0]*split)
    train_labels, val_labels = labels[val_idx:], labels[:val_idx]
    train_data, val_data = images[val_idx:], images[:val_idx]

    train_dataset = TensorDataset(train_data,val_labels)
    train_dataloader = DataLoader(train_dataset)

    val_dataset = TensorDataset(val_data,train_labels)
    val_dataloader = DataLoader(val_dataset)

    return train_dataloader, val_dataloader