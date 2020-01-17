"""
"""
import torchvision
from torch.utils.data import Dataset, DataLoader
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
            args.src, args.filenames, args.gt_filenames, args.resize)
    # No resizing
    else:
        data_set = dataloader(
            args.src, args.filenames, args.gt_filenames,)

    for s in data_set:
        def vis_points(image, points, diameter=15):
            image = np.array(torchvision.transforms.ToPILImage()
                             (image).convert('RGB'))
            im = image.copy()

            for (x, y) in points:
                cv2.circle(im, (int(x), int(y)), 1, (0, 255, 0), -1)

            plt.imshow(im)
            plt.show()

        vis_points(s['image'], s['keypoints'])

        break

    return data_set.__getitem__(
        args.getimage) if args.getimage else DataLoader(
            data_set, batch_size=args.batchsize)
