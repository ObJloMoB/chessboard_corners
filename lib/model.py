import torch
import torch.nn as nn
from torchvision.models import resnet50


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = resnet50(pretrained=True)
        self.model.conv1 = nn.Conv2d(in_channels=1, 
                                     out_channels=64, 
                                     kernel_size=(7, 7), 
                                     stride=(2, 2), 
                                     padding=(3, 3), 
                                     bias=False)
        self.model.fc = nn.Linear(2048, 8)

        # print(self.model)

    def forward(self, x):
        ret = self.model(x)
        return ret