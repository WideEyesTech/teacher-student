"""Basic Datset

Load random augmented data from a root folder
"""
# TODO improve documentation

from utils.params import Params
from utils.albumentation import Albumentation
import argparse
import sys
import os
import pandas as pd
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

from PIL import Image

sys.path.insert(0, 'utils/')


# Load the config from json file
config_path = os.path.join("./config.json")
assert os.path.isfile(
    config_path), "No json configuration file found at {}".format(config_path)
config = Params(config_path)


class FaceKeypointsDataset(Dataset):
    """Test FaceKeypointsDataset face_keypoints_dataset class

    """

    def __init__(self, env='train'):
        """Test FaceKeypointsDataset main class
        """

        # Env (train/test/validation)
        self.env = env

        # Load images
        self.images = pd.read_csv(
            config.DATASETS["face_keypoints_dataset"]["paths"][self.env]["images"])['Image']

        # Load labels
        self.labels = pd.read_csv(
            config.DATASETS["face_keypoints_dataset"]["paths"][self.env]["labels"])

        self.labels = self.labels.loc[:,
                                      self.labels.columns != 'Image']

        # Remove rows with missing values
        self.labels = pd.DataFrame.dropna(
            self.labels, axis=0, how='any', thresh=None, subset=None, inplace=False)
        mask = np.delete(np.array(range(len(self.images.index))),
                         self.labels.index, None)
        self.images = self.images.drop(mask)

        # Assert last step has been made propertly
        assert len(self.labels) == len(self.images)

        images = []
        # Transform each matrix into an image (1, 96, 96)
        for i in range(self.images.shape[0]):
            img = np.fromstring(self.images.iloc[i], sep=' ').astype(
                np.float32).reshape(-1, 96)
            img = Image.fromarray(img)

            images.append(np.array(img))

        self.images = images

        # Split train/val data
        if self.env == 'train':
            self.images = self.images[: config.DATASETS["face_keypoints_dataset"]
                                      ["batches"]["train"]]
            self.labels = self.labels[: config.DATASETS["face_keypoints_dataset"]
                                      ["batches"]["train"]]
        elif self.env == 'val':
            self.images = self.images[config.DATASETS["face_keypoints_dataset"]
                                      ["batches"]["train"]:]
            self.labels = self.labels[config.DATASETS["face_keypoints_dataset"]
                                      ["batches"]["train"]:]

    # Method __len__ must be override in Pytorch
    def __len__(self):
        return len(self.labels)

    # Method __getitem__ must be override in Pytorch
    def __getitem__(self, idx):

        # Create Output object
        landmarks = self.labels.iloc[idx, :].as_matrix().reshape(-1, 2)
        output = {
            'image': self.images[idx],
            'keypoints': landmarks.astype(np.float32)
        }

        # Do not apply data augmentation when testing
        if self.env != 'test':
            # Transform Output with albumentation
            aug = Albumentation(
                'kp').fast_aug()
            if 'keypoints' == 'labels':
                output['image'] = aug(output['image'])
            else:
                output = aug(**output)

        # Normalize and generate tensor from output['image']
        output['image'] = torchvision.transforms.ToTensor()(output['image'])
        output['keypoints'] = np.array(
            output['keypoints']) / self.images[idx].shape[:2] - 0.5

        return output['image'], torch.Tensor(output['keypoints'].reshape(output['keypoints'].shape[0]*2))


def handleinput(args):
    """Add logic to user input

    Args:
        args: Parsed arguments from user input
    """

    # Check all params are correct
    # src param
    if not os.path.exists(args.src):  # If does not exist
        raise ValueError

    # Load dataset
    data_set = FaceKeypointsDataset(args.env)

    # Return the Dataloader or single image depending on args
    return data_set.__getitem__(
        args.getimage) if args.getimage else DataLoader(
            data_set, batch_size=args.batchsize)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        description='DataLoader')
    PARSER.add_argument('src', type=str, help='path to images')
    PARSER.add_argument('labels', type=str,
                        help='path to labels')
    PARSER.add_argument('--getimage', '-gi', type=int,
                        help='Get only one image')
    PARSER.add_argument('--env', '-e', help='Whether env is train, validation or test',
                        choices=['train', 'validation', 'test'], default='train')

    ARGS = PARSER.parse_args()

    OUTPUT = handleinput(ARGS)

    print(OUTPUT)
