import numpy as np
import torch
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision.transforms as transforms
from PIL import Image

import cv2


def split_trainval(root_path, bs=32, split=0.2):
    images = np.load(os.path.join(root_path, 'xtrain.npy'))
    labels = np.load(os.path.join(root_path, 'ytrain.npy'))

    idxes = np.arange(0, labels.shape[0])
    np.random.seed(322)
    np.random.shuffle(idxes)

    val_idx = int(labels.shape[0]*split)
    train_labels, val_labels = labels[val_idx:], labels[:val_idx]
    train_data, val_data = images[val_idx:], images[:val_idx]

    print(train_data.shape, train_labels.shape)

    train_dataset = NpyData(train_data, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

    val_dataset = NpyData(val_data, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

    return train_dataloader, val_dataloader


def get_test(root_path, bs=1):
    images = np.load(os.path.join(root_path, 'xtest.npy'))
    print(images.shape)
    test_dataset = NpyTestData(images)

    return test_dataset


class NpyTestData(Dataset):
    def __init__(self, data_npy):
        self.data_npy = data_npy
        self.transforms = transforms.Compose([transforms.Resize(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.449],
                                                                   std= [0.226])])

    def __getitem__(self, idx):
        img = self.data_npy[idx]

        img = Image.fromarray(np.uint8(img.squeeze()), 'L')
        img = self.transforms(img)

        return img.unsqueeze(0)

    def __len__(self):
        return self.labels_npy.shape[0]


class NpyData(Dataset):
    def __init__(self, data_npy, labels_npy):
        self.data_npy = data_npy
        self.labels_npy = labels_npy
        self.transforms = transforms.Compose([transforms.Resize(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.449],
                                                                   std= [0.226]),
                                              ])

    def __getitem__(self, idx):
        img = self.data_npy[idx]
        label = self.labels_npy[idx]
        img = img.squeeze()
        img = Image.fromarray(np.uint8(img), 'L')
        img = self.transforms(img)

        return img, torch.Tensor(label)

    def __len__(self):
        return self.labels_npy.shape[0]
