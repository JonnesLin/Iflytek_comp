import torch
import torch.nn as nn
import timm

class XunFeiNet(nn.Module):
    def __init__(self):
        super(XunFeiNet, self).__init__()
        self.resnet34 = timm.create_model('efficientnet_b4', pretrained=True, num_classes=7, drop_rate=0.2)
        
    def forward(self, img):        
        out = self.resnet34(img)
        
        return out