"""Basic data loader module

Load random augmented data from a root folder
"""
# TODO improve docs
# TODO make path absolutes by creating config file

import argparse
import os

# import torch
import torchvision

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class BasicDataLoader(Dataset):
    """Test Dataset main class

    Args:
        Dataset: Pytorch Dataset module

    Attributes:
        root_dir (str): Root to dir where images are saved.
        images_size: Size to resize images, 0 means no resizing
        files_list: List of files contained in root_dir. Only files with specific
        format will be considered.
        color_mode: Images color mode
    """

    def __init__(self, root_dir, image_filenames, images_size=0, color_mode='RGB'):
        """Test Dataset main class

        Args:
            root_dir: Root to dir where images are saved.
            images_size: Size to resize images, 0 means no resizing.
            color_mode: Images color mode
        """
        # Image info
        self.images_size = images_size
        self.color_mode = color_mode
        # Dir info
        self.root_dir = root_dir
        self.image_filenames = image_filenames
        self.gt_filenames = []

    # Method __len__ must be override in Pytorch

    def __len__(self):
        return len(self.files_list)

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

        # Open and show Image
        img = Image.open(os.path.join(
            self.root_dir, self.image_filenames[idx]))

        # Transforms
        # TODO make transforms optional
        transforms = []

        # Resize or not
        if self.images_size != 0:
            transforms.append(torchvision.transforms.Resize(self.images_size))
            transforms.append(
                torchvision.transforms.RandomCrop(self.images_size))

        # More transforms for data augmentation
        transforms.append(torchvision.transforms.RandomHorizontalFlip(p=0.5))
        transforms.append(torchvision.transforms.RandomPerspective(
            distortion_scale=0.5, p=0.2, interpolation=3))

        # Transform image to torch tensor
        transforms.append(torchvision.transforms.ToTensor())

        # Normalize image
        if self.color_mode == 'RGB':
            transforms.append(torchvision.transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        elif self.color_mode == 'L':
            transforms.append(torchvision.transforms.Normalize((0.5,), (0.5,)))

        return {
            'image': torchvision.transforms.Compose(transforms)(img),
            'label': self.gt_filenames[idx]}


def handle_user_input(args):
    """Add logic to user input

    Args:
        args: Parsed arguments from user input
    """

    # Check all params are correct
    # src param
    if not os.path.exists(args.src):  # If does not exist
        raise ValueError

    # Resize param
    if args.resize:
        data_set = BasicDataLoader(args.src, args.filenames, args.resize)
    # No resizing
    else:
        data_set = BasicDataLoader(args.src, args.filenames)

    return data_set.__getitem__(
        args.getimage) if args.getimage else DataLoader(
            data_set, batch_size=args.batchsize)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        description='Training over datsets creation')
    PARSER.add_argument('src', type=str, help='path to get images')
    PARSER.add_argument('filenames', type=str,
                        help='path to get images filenames')
    PARSER.add_argument('--getimage', '-gi', type=int,
                        help='Get only one image')
    PARSER.add_argument('--batchsize', '-b', help='Batch size', type=int,
                        default=1)
    PARSER.add_argument('--resize', '-r',
                        help='Resize images',
                        type=int, nargs='+')

    ARGS = PARSER.parse_args()

    OUTPUT = handle_user_input(ARGS)

    print(OUTPUT)
