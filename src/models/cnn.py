import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalNetwork(nn.Module): #Building our own convolutional neural network
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 40, 3)
        self.conv2 = nn.Conv2d(40, 20, 3)
        self.fc1 = nn.Linear(20*54*54, 128)  #building with 120 neurons
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self,X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = nn.Flatten()(X) #X=X.view(-1,20*54*54)# similar to Flatten
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return F.log_softmax(X, dim=1)