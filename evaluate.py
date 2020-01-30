"""Evaluates the model"""
from utils.checkpoints import CheckPoints
from utils.params import Params
from utils.logger import Logger
from utils.json import DictToJson

import argparse
import logging
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import models.face_keypoints_net as net
import dataset.face_keypoints_dataset as data_loader


class Evaluate():
    """"""

    def __init__(self, model, loss_fn, dataloader, metrics, params):
        """Evaluate the model on `num_steps` batches.
        Args:
            model: (torch.nn.Module) the neural network
            loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
            dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
            metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
            params: (Params) hyperparameters
            num_steps: (int) number of batches to train on, each of size params.batch_size
        """

        self.model = model
        self.loss_fn = loss_fn
        self.dataloader = dataloader
        self.metrics = metrics
        self.params = params

    def __call__(self):
        """"""

        # set model to evaluation mode
        self.model.eval()

        # summary for current eval loop
        summ = []

        # compute metrics over the dataset
        for (data_batch, labels_batch) in self.dataloader:
            # move to GPU if available
            if self.params.cuda:
                data_batch, labels_batch = data_batch.cuda(
                    non_blocking=True), labels_batch.cuda(non_blocking=True)

            # compute model output
            output_batch = self.model(data_batch)
            loss = self.loss_fn(output_batch, labels_batch)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # compute all metrics on this batch
            summary_batch = {metric: self.metrics[metric](output_batch, labels_batch)
                             for metric in self.metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

        # compute mean of all metrics in summary
        metrics_mean = {metric: np.mean([x[metric]
                                         for x in summ]) for metric in summ[0]}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                    for k, v in metrics_mean.items())
        logging.info("- Eval metrics : " + metrics_string)
        return metrics_mean


def handleinput(args):
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()  # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Get the logger
    Logger()(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
    model = net.Net(params).cuda() if params.cuda else net.Net(params)

    loss_fn = net.loss_fn
    metrics = net.metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    CheckPoints().load(os.path.join(
        args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = Evaluate(model, loss_fn, test_dl, metrics, params)()
    save_path = os.path.join(
        args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    DictToJson()(test_metrics, save_path)


if __name__ == '__face_keypoints_dataset__':
    # TODO evaluate script
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument()

    ARGS = PARSER.parse_args()
    handleinput(ARGS)
