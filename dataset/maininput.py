"""
"""
import torchvision
from torch.utils.data import DataLoader
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

def handleinput(args, dataloader):
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
        data_set = dataloader(
            args.src, args.filenames, args.labelnames, args.labels_type, args.env, args.resize)
    # No resizing
    else:
        data_set = dataloader(
            args.src, args.filenames, args.labelnames, args.labels_type, args.env)

    return data_set.__getitem__(
        args.getimage) if args.getimage else DataLoader(
            data_set, batch_size=args.batchsize)
