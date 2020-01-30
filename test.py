import torch
import argparse
import os
import cv2
import numpy as np
from torch.utils.data import DataLoader

from utils.params import Params
from utils.checkpoints import CheckPoints
from utils.show_results import ShowFacialKeypoints

# Load the config from json file
CONFIG_PATH = "./config.json"
assert os.path.isfile(
    CONFIG_PATH), "No json configuration file found at {}".format(CONFIG_PATH)
CONFIG = Params(CONFIG_PATH)

# Import active datasets
if CONFIG.DATASETS["active"] == "face_keypoints_dataset":
    import models.face_keypoints_net as net
    from dataset.face_keypoints_dataset import FaceKeypointsDataset as data_loader

ACTIVE_DATASET = CONFIG.DATASETS[CONFIG.DATASETS["active"]]


def handleInput(args):
    # Load model
    model = net.Net({})
    # Load weights
    model = CheckPoints().load(args.modelpath, model)

    model.eval()

    # Test only one image
    if args.image:
        # Load Image
        image = cv2.imread(args.image, 0)
        # Resize
        image = cv2.resize(
            image, (ACTIVE_DATASET["images"]["size"], ACTIVE_DATASET["images"]["size"]))
        # Normalize
        image = (image / abs(np.max(image)) - 0.5)*2

        # Create tensor and unsquezee
        image = torch.Tensor(image)
        image = image.unsqueeze(0)
        image = image.unsqueeze(0)

        # Predictions
        pred = model(image)

        ShowFacialKeypoints(
                    image, (pred + 0.5)*ACTIVE_DATASET["images"]["size"]).show()
    else:
        # Load params
        params_path = os.path.join(args.params, 'params.json')
        assert os.path.isfile(
            params_path), "No json configuration file found at {}".format(params_path)
        params = Params(params_path)

        dataloader = DataLoader(data_loader(
            'test'), batch_size=params.batch_size)

        # move to GPU if available
        for i, images_batch in enumerate(dataloader):
            if torch.cuda.is_available():
                images_batch = images_batch.cuda(non_blocking=True)
                model.cuda()

            # compute model output and loss
            output_batch = model(images_batch)

            # Show last result from Epoch
            if i == dataloader.__len__() - 1:
                ShowFacialKeypoints(
                    images_batch, (output_batch + 0.5)*ACTIVE_DATASET["images"]["size"]).show()


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        description='Predict image with model')
    PARSER.add_argument('modelpath',
                        help="pre trained model")
    PARSER.add_argument('--image', '-i',
                        help="image for inference")
    PARSER.add_argument('--params', '-p',
                        help="params")
    ARGS = PARSER.parse_args()

    handleInput(ARGS)
