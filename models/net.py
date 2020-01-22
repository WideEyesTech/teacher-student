"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    """

    def __init__(self, params):
        """
        """
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 5)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32,64,3)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64,128,3)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128,256,3)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(256,512,1)
        self.pool5 = nn.MaxPool2d(2, 2)

        #Linear Layer
        self.fc1 = nn.Linear(512*2*2, 256)
        self.fc2 = nn.Linear(256, 30)

        # Dropouts
        self.drop1 = nn.Dropout(p = 0.1)
        self.drop2 = nn.Dropout(p = 0.2)
        self.drop3 = nn.Dropout(p = 0.25)
        self.drop4 = nn.Dropout(p = 0.25)
        self.drop5 = nn.Dropout(p = 0.3)
        self.drop6 = nn.Dropout(p = 0.4)

    def forward(self, x):
        
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.drop2(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.drop3(x)
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.drop4(x)
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.drop5(x)

        x = x.view(1, x.size(1)*2*2)
        x = F.relu(self.fc1(x))
        x = self.drop6(x)
        x = self.fc2(x)

        return x

def loss_fn(outputs, labels):
    return nn.SmoothL1Loss(reduction='mean')(outputs, labels)


def accuracy(outputs, labels):
    """
    """
    return 1/float(labels.size) * (np.sum(labels - outputs)/float(labels.size))


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}