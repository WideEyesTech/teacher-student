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
        
        self.conv1 = nn.Conv2d(3, 32, 5)

        # output size = (W-F)/S +1 = (224-5)/1 + 1 = 220
        self.pool1 = nn.MaxPool2d(2, 2)
        # 220/2 = 110  the output Tensor for one image, will have the dimensions: (32, 110, 110)

        self.conv2 = nn.Conv2d(32,64,3)
        # output size = (W-F)/S +1 = (110-3)/1 + 1 = 108
        self.pool2 = nn.MaxPool2d(2, 2)
        #108/2=54   the output Tensor for one image, will have the dimensions: (64, 54, 54)

        self.conv3 = nn.Conv2d(64,128,3)
        # output size = (W-F)/S +1 = (54-3)/1 + 1 = 52
        self.pool3 = nn.MaxPool2d(2, 2)
        #52/2=26    the output Tensor for one image, will have the dimensions: (128, 26, 26)

        self.conv4 = nn.Conv2d(128,256,3)
        # output size = (W-F)/S +1 = (26-3)/1 + 1 = 24
        self.pool4 = nn.MaxPool2d(2, 2)
        #24/2=12   the output Tensor for one image, will have the dimensions: (256, 12, 12)

        self.conv5 = nn.Conv2d(256,512,1)
        # output size = (W-F)/S +1 = (12-1)/1 + 1 = 12
        self.pool5 = nn.MaxPool2d(2, 2)
        #12/2=6    the output Tensor for one image, will have the dimensions: (512, 6, 6)

        #Linear Layer
        self.fc1 = nn.Linear(512*6*6, 1024)
        self.fc2 = nn.Linear(1024, 136)

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
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop6(x)
        x = self.fc2(x)

        return x

def loss_fn(outputs, labels):
    return nn.SmoothL1Loss()(outputs, labels)


def accuracy(outputs, labels):
    """
    """
    return np.sum(labels - outputs)/float(labels.size)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}