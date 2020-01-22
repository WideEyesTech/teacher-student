"""Basic data loader module

Load random augmented data from a root folder
"""
# TODO improve docs
# TODO make path absolutes by creating config file

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.albumentation import Albumentation
from dataset.maininput import handleinput

import torch
import torchvision
from torch.utils.data import Dataset

from PIL import Image

sys.path.insert(0, 'utils/')


class MainDataset(Dataset):
    """Test MainDataset main class

    """

    def __init__(self, root_dir, filenames_root, labels_root, labels_type='labels', env='train', images_size=0):
        """Test MainDataset main class
        """

        # Image info
        self.images_size = images_size
        self.labels_type = labels_type

        # Load images
        self.images = np.load(root_dir)

        self.labels_root = pd.read_csv(labels_root)
        # Remove rows with missing values
        self.labels_root = pd.DataFrame.dropna(
            self.labels_root, axis=0, how='any', thresh=None, subset=None, inplace=False)
        
        # Align images data with labels data
        mask = np.delete(np.array(range(len(self.images[0][0]))), self.labels_root.index, None)
        self.images = np.delete(self.images, mask, axis=2)

        assert len(self.labels_root) == len(self.images[0][0])

        images = []
        # Transform each matrix into an image
        for i in range(len(self.images[0][0])):
            img = self.images[:,:,i]
            img = Image.fromarray(img)

            images.append(np.array(img))

        self.images = images
    
        # Env (train/test/validation)
        self.env = env

    # Method __len__ must be override in Pytorch
    def __len__(self):
        return len(self.labels_root)

    # Method __getitem__ must be override in Pytorch
    def __getitem__(self, idx):
        # Check if argument is correct
        if not isinstance(idx, int):
            if isinstance(idx, float):
                print('Converting {} to int({})'.format(idx, abs(idx)))
                idx = abs(idx)
            elif isinstance(idx, str):
                try:
                    idx = int(idx)
                except ValueError:
                    print('Expected type "int" argument')
                    return None

        # Create Output object
        output = {
            'image': self.images[idx],
            self.labels_type: self.labels_root.iloc[idx, :].as_matrix().reshape(-1, 2).astype('uint8')
        }
        
        if self.env != 'test':
            # Transform Output with albumentation
            aug = Albumentation('kp', [self.images_size, self.images_size]).fast_aug()

            if self.labels_type == 'labels':
                output['image'] = aug(output['image'])
            else:
                output = aug(**output)

        
        # plt.figure()
        # plt.imshow(output['image'], cmap='gray')
        # x = []
        # y = []
        # for i,j in output[self.labels_type]:
        #     x.append(i)
        #     y.append(j)
        # plt.scatter(x, y)
        # plt.show()

        # Normalize and generate tensor from output['image']
        output['image'] = torchvision.transforms.ToTensor()(output['image'])
        output['image'] = torchvision.transforms.Normalize(
            (0.5,), (0.5,))(output['image'])
        

        return output['image'], torch.Tensor(output[self.labels_type]).view(len(output[self.labels_type]*2))


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        description='DataLoader')
    PARSER.add_argument('src', type=str, help='path to get the images')
    PARSER.add_argument('filenames', type=str,
                        help='path to get images filenames')
    PARSER.add_argument('labelnames', type=str,
                        help='path to get images ground truth filenames')
    PARSER.add_argument('--getimage', '-gi', type=int,
                        help='Get only one image')
    PARSER.add_argument('--batchsize', '-b', help='Batch size', type=int,
                        default=1)
    PARSER.add_argument('--env', '-e', help='Whether env is train, validation or test',
                        choices=['train', 'validation', 'test'], default='train')
    PARSER.add_argument('--labeltype', '-lt', help='Label type',
                        choices=['labels', 'keypoints', 'bboxes'], default='labels')
    PARSER.add_argument('--resize', '-r',
                        help='Resize images H x W, Ex: 500 200; 760 230 ....',
                        type=int, nargs='+')

    ARGS = PARSER.parse_args()

    OUTPUT = handleinput(ARGS, MainDataset)

    print(OUTPUT)
