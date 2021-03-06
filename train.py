import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import numpy as np
import cv2
import argparse
import os

from time import time, sleep

from lib.model import Model
from lib.dataset import split_trainval


def train(opts):
    # Select device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Define model
    model = Model().to(device)
    
    # Define dataloaders
    train_loader, val_loader = split_trainval(opts.data, opts.bs)
    
    # Define loss
    loss_criter = nn.L1Loss().to(device)

    # Define optimizer
    optimizer = Adam(model.parameters(), lr=opts.lr, weight_decay=1e-6)
    scheduler = StepLR(optimizer, step_size=int(opts.epoch/2), gamma=0.1)

    # Training loop
    for epoch in range(opts.epoch):
        # Train cycle
        running_loss = 0.0
        model.train()

        for batch_num, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = loss_criter(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            print(f'epoch num {epoch:02d} batch num {batch_num:04d} train loss {loss:02.04f}', end='\r')

        epoch_loss = running_loss / len(train_loader.dataset)

        # Val cycle
        running_loss = 0.0
        model.eval()
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                outputs = model(inputs)
                loss = loss_criter(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
        epoch_val_loss = running_loss / len(val_loader.dataset)
        print(f'\n\nepoch num {epoch:02d} train loss {epoch_loss:02.04f} val loss {epoch_val_loss:02.04f}')

        scheduler.step()
        if (epoch + 1) % opts.save_every == 0:
            torch.save(model.state_dict(), os.path.join(opts.output, f'checkpoint_size{opts.size}_e{epoch+1}of{opts.epoch}_lr{opts.lr:.01E}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='DATA',
                        type=str, required=True)
    parser.add_argument('--val_split', help='validation data proportion',
                        default=0.2, type=float)
    parser.add_argument('--lr', help='LR',
                        default=1e-4, type=float)
    parser.add_argument('--size', help='Input image size',
                        default=224, type=int)
    parser.add_argument('--epoch', help='Train duration',
                        default=30, type=int)
    parser.add_argument('--bs', help='BS',
                        default=64, type=int)
    parser.add_argument('--save_every', help='Save every N epoch',
                        default=10, type=int)
    parser.add_argument('--output', help='snapshot fld',
                        default='logs', type=str)

    args = parser.parse_args()
    train(args)
