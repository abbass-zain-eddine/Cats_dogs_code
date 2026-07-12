from torch import nn
import torch.nn.functional as F
from torchvision import models

class CNNResnet(nn.Module):
    def __init__(self):
        super(CNNResnet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        new_head = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 2),
        )
        self.model.fc = new_head

        for param in self.model.parameters():
                param.requires_grad = False
        for param in self.model.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x