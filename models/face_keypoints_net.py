"""Defines the neural network, losss function and metrics"""

from utils.params import Params

import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# Load the config from json file
CONFIG_PATH = "./config.json"
assert os.path.isfile(
    CONFIG_PATH), "No json configuration file found at {}".format(CONFIG_PATH)
CONFIG = Params(CONFIG_PATH)

ACTIVE_DATASET = CONFIG.DATASETS[CONFIG.DATASETS["active"]]


class Net(nn.Module):
    """
    """

    def __init__(self, params):
        """
        """
        super(Net, self).__init__()

        self.params = params

        self.conv1 = nn.Conv2d(1, 32, 5)

        self.pool1 = nn.MaxPool2d(3, 2)

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(3, 2)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool3 = nn.MaxPool2d(3, 2)

        self.conv4 = nn.Conv2d(128, 256, 3)
        self.pool4 = nn.MaxPool2d(3, 2)

        self.conv5 = nn.Conv2d(256, 512, 1)
        self.pool5 = nn.MaxPool2d(3, 2)

        # Linear Layer
        self.fc1 = nn.Linear(512 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 30)  # 15 landmarks * 2 (x,y) = 30

        # Dropouts
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.25)
        self.drop4 = nn.Dropout(p=0.25)
        self.drop5 = nn.Dropout(p=0.3)
        self.drop6 = nn.Dropout(p=0.4)

    def forward(self, x):
        apply_drop = ACTIVE_DATASET["dropout"]
        x = self.pool1(F.relu(self.conv1(x)))
        if apply_drop:
            x = self.drop1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        if apply_drop:
            x = self.drop2(x)
        x = self.pool3(F.relu(self.conv3(x)))
        if apply_drop:
            x = self.drop3(x)
        x = self.pool4(F.relu(self.conv4(x)))
        if apply_drop:
            x = self.drop4(x)
        x = self.pool5(F.relu(self.conv5(x)))
        if apply_drop:
            x = self.drop5(x)
        bsize = x.shape[0]
        x = x.view(bsize, -1)
        x = F.relu(self.fc1(x))
        if apply_drop:
            x = self.drop6(x)
        x = self.fc2(x)

        return x


def loss_fn(outputs, labels):
    """"""
    if ACTIVE_DATASET["loss_fn"] == "MSE":
        return nn.MSELoss()(outputs, labels)
    
    if ACTIVE_DATASET["loss_fn"] == "SL1":
        return nn.SmoothL1Loss()(outputs, labels)


def MSE(outputs, labels):
    """
    """
    return np.mean(np.sqrt(np.sum((labels - outputs)**2, 1)))


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'MSE': MSE,
    # could add more metrics such as accuracy for each token type
}
