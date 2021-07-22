import torch
import torch.nn as nn
import timm
from .resmasking. import *

class XunFeiNet(nn.Module):
    def __init__(self):
        super(XunFeiNet, self).__init__()
        self.resnet50 = ResMasking50("")
        
    def forward(self, img):        
        out = self.resnet50(img)
        
        return out
