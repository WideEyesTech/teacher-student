"""Basic Datset

Load random augmented data from a root folder
"""
# TODO improve documentation

from utils.params import Params
from utils.albumentation import Albumentation
from utils.show_results import ShowFacialKeypoints

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
CONFIG_PATH = os.path.join("./config.json")
assert os.path.isfile(
    CONFIG_PATH), "No json configuration file found at {}".format(CONFIG_PATH)
CONFIG = Params(CONFIG_PATH)

ACTIVE_DATASET = CONFIG.DATASETS[CONFIG.DATASETS["active"]]


class FaceKeypointsDataset(Dataset):
    """Test FaceKeypointsDataset face_keypoints_dataset class

    """

    def __init__(self, env='train', data_aug_prob=0.5):
        """Test FaceKeypointsDataset main class
        """

        # Env (train/test/validation)
        self.env = env

        # Data augmentation probability
        self.data_aug_prob = data_aug_prob

        # Load images
        self.images = pd.read_csv(
            ACTIVE_DATASET["paths"][self.env]["images"])['Image']

        # Load labels
        self.labels = pd.read_csv(
            ACTIVE_DATASET["paths"][self.env]["labels"])

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
            self.images = self.images[: ACTIVE_DATASET
                                      ["batches"]["train"]]
            self.labels = self.labels[: ACTIVE_DATASET
                                      ["batches"]["train"]]
        elif self.env == 'val':
            self.images = self.images[ACTIVE_DATASET
                                      ["batches"]["train"]:]
            self.labels = self.labels[ACTIVE_DATASET
                                      ["batches"]["train"]:]

        # Show image with labels if enabled
        if ACTIVE_DATASET["show_preview"]:
            ShowFacialKeypoints(self.images, self.labels).show()



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
            # Transform output with albumentation
            aug = Albumentation(
                'kp', {"format": "xy", "remove_invisible": False}, "spatial_level", self.data_aug_prob).transform()

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
