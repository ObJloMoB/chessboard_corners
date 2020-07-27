import torch
import torch.nn as nn
from torchvision.models import resnet50


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(2048, 8)

    def forward(self, x):
        ret = self.model(x)
        return ret