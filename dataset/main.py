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
from utils.albumentation import Albumentation
from dataset.maininput import handleinput

import torch
import torchvision
from torch.utils.data import Dataset

from PIL import Image

sys.path.insert(0, 'utils/')


class MainDataset(Dataset):
    """Test MainDataset main class

    Args:
        MainDataset: Pytorch MainDataset module

    Attributes:
        root_dir (str): Root to dir where images are saved.
        images_size: Size to resize images, 0 means no resizing
    """

    def __init__(self, root_dir, filenames_root, labels_root, labels_type='labels', env='train', images_size=0):
        """Test MainDataset main class

        Args:
            root_dir: Root to dir where images are saved.
            images_size: Size to resize images, 0 means no resizing.
            env: Whether if env is train test or validation
        """
        # Image info
        self.images_size = images_size
        self.labels_type = labels_type
        # Dir info
        self.root_dir = root_dir
        if os.path.isfile(filenames_root):
            self.image_names = [l.strip().replace('\n', '')
                                for l in open(filenames_root).readlines()]

        self.labels_root = pd.read_csv(labels_root)
        # Env (train/test/validation)
        self.env = env

    # Method __len__ must be override in Pytorch
    def __len__(self):
        return len(self.image_names)

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
            'image': np.array(Image.open(os.path.join(
                self.root_dir, self.image_names[idx]))),
            self.labels_type: self.labels_root.iloc[idx, 1:].as_matrix().reshape(-1, 2).astype('uint8')
        }

        if self.env != 'test':
            # Transform Output with albumentation
            aug = Albumentation('kp', [200, 200]).fast_aug()

            if self.labels_type == 'labels':
                output['image'] = aug(output['image'])
            else:
                output = aug(**output)

        # Normalize and generate tensor from output['image']
        output['image'] = torchvision.transforms.ToTensor()(output['image'])
        output['image'] = torchvision.transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(output['image'])

        return output['image'], torch.Tensor(output[self.labels_type])


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
