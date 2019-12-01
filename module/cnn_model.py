import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet_model import ResidualNet

class cnn_model(nn.Module):

    def __init__(self,num_class):
        self.base_model = ResidualNet("ImageNet",18,1000,None)

    def forward(self,input):
        pass