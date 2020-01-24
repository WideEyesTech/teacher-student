""""""
import matplotlib.pyplot as plt
import numpy as np
import torch


class ShowFacialKeypoints():
    """"""

    def __init__(self, image, label):
        """"""
        self.image = image
        self.label = label

    def show(self):
        """"""
        if torch.is_tensor(self.label):
            self.fromTorchTensor()

        if isinstance(self.label, list) or isinstance(self.image, list):
            self.fromList()

        if len(self.image.shape) > 2 or len(self.label.shape) > 2:
            self.getDataFromBatch()

        if len(self.label.shape) == 1:
            self.labelToXY()
            
        plt.imshow(self.image)
        x = self.label[:, 0]
        y = self.label[:, 1]

        plt.scatter(x, y, c='red')
        plt.show()

    def fromTorchTensor(self):
        """"""
        self.image = self.image.data.cpu().numpy()
        self.label = self.label.data.cpu().numpy()

    def fromList(self):
        """"""
        self.label = np.array(self.label)
        self.image = np.array(self.image)

    def labelToXY(self):
        """"""

        self.label = self.label.reshape(-1, 2)

    def getDataFromBatch(self):

        if len(self.image.shape) == 3:
            self.image = self.image[0, :, :]

        if len(self.image.shape) == 4:
            self.image = self.image[0, 0, :, :]

        self.label = self.label[0, :]
