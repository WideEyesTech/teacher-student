"""Training loop
"""
from dataset.main import MainDataset
from utils.average import RunningAverage
from utils.checkpoints import CheckPoints
from utils.json import DictToJson
from utils.params import Params
from utils.logger import Logger

import argparse
import logging
import os
import json

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

import models.net as net
from dataset.main import MainDataset as data_loader
from evaluate import Evaluate


class Train():
    """Training loop main class
    """

    def __init__(self, model, optimizer, loss_fn, dataloader, metrics, params):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.dataloader = dataloader
        self.metrics = metrics
        self.params = params

        return

    def __call__(self):
        """Start training loop
        """

        # Set model in train mode
        self.model.train()

        # summary for current training loop and a running average object for loss
        summ = []
        loss_avg = RunningAverage()

        # Use tqdm for progress bar
        with tqdm(total=len(self.dataloader)) as t:
            for i, (train_batch, labels_batch) in enumerate(self.dataloader):
                # move to GPU if available
                if self.params.cuda:
                    train_batch, labels_batch = train_batch.cuda(
                        non_blocking=True), labels_batch.cuda(non_blocking=True)
                # convert to torch Variables
                train_batch, labels_batch = Variable(
                    train_batch), Variable(labels_batch)

                # compute model output and loss
                output_batch = self.model(train_batch)
                loss = self.loss_fn(output_batch, labels_batch)

                # clear previous gradients, compute gradients of all variables wrt loss
                self.optimizer.zero_grad()
                loss.backward()

                # performs updates using calculated gradients
                self.optimizer.step()

                # Evaluate summaries only once in a while
                if i % self.params.save_summary_steps == 0:
                    # extract data from torch Variable, move to cpu, convert to numpy arrays
                    output_batch = output_batch.data.cpu().numpy()
                    labels_batch = labels_batch.data.cpu().numpy()

                    # compute all metrics on this batch
                    summary_batch = {metric: self.metrics[metric](output_batch, labels_batch)
                                     for metric in self.metrics}
                    summary_batch['loss'] = loss.item()
                    summ.append(summary_batch)

                # update the average loss
                loss_avg.update(loss.item())

                t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
                t.update()

        # compute mean of all metrics in summary
        metrics_mean = {metric: np.mean([x[metric]
                                         for x in summ]) for metric in summ[0]}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                    for k, v in metrics_mean.items())
        logging.info("- Train metrics: " + metrics_string)


class TrainAndEval():
    """
    """

    def __init__(self, model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir):
        """Train the model and evaluate every epoch.
        Args:
            model: (torch.nn.Module) the neural network
            train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
            val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
            optimizer: (torch.optim) optimizer for parameters of model
            loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
            metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
            params: (Params) hyperparameters
            model_dir: (string) directory containing config, weights and log
        """
        self.model = model
        self.params = params
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.params = params
        self.model_dir = model_dir

    def __call__(self):

        best_val_acc = 0.0

        for epoch in range(self.params.num_epochs):
            # Run one epoch
            logging.info("Epoch {}/{}".format(epoch +
                                              1, self.params.num_epochs))

            # compute number of batches in one epoch (one full pass over the training set)
            Train(self.model, self.optimizer, self.loss_fn,
                  self.train_dataloader, self.metrics, self.params)

            # Evaluate for one epoch on validation set
            val_metrics = Evaluate(
                self.model, self.loss_fn, self.val_dataloader, self.metrics, self.params)()

            val_acc = val_metrics['accuracy']
            is_best = val_acc >= best_val_acc

            # Save weights
            CheckPoints().save({'epoch': epoch + 1,
                                'state_dict': self.model.state_dict(),
                                'optim_dict': self.optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=self.model_dir)

            # If best_eval, best_save_path
            if is_best:
                logging.info("- Found new best accuracy")
                best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
                best_json_path = os.path.join(
                    self.model_dir, "metrics_val_best_weights.json")
                DictToJson(val_metrics, best_json_path)

            # Save latest val metrics in a json file in the model directory
            last_json_path = os.path.join(
                self.model_dir, "metrics_val_last_weights.json")
            DictToJson(val_metrics, last_json_path)


def handleinput(args):
    """Handle user input
    """

    # Load the parameters from json file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Load paths info
    with open('config.json') as f:
        paths = json.load(f)['PATHS']

    # Load dataset info
    with open('config.json') as f:
        dataset_info = json.load(f)['DATASET']

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    Logger()(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    train_dl = DataLoader(data_loader(
        paths['images'], paths['filenames'], paths['labelnames'], dataset_info['labels']['type']), batch_size=params.batch_size)
    val_dl = DataLoader(data_loader(
        paths['images'], paths['filenames'], paths['labelnames'], dataset_info['labels']['type'], 'val'), batch_size=params.batch_size)

    logging.info("- done.")

    # Define the model and optimizer
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    loss_fn = net.loss_fn
    metrics = net.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    TrainAndEval(model, train_dl, val_dl, optimizer,
                 loss_fn, metrics, params, args.model_dir)()


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        description='Training over datsets creation')
    # TODO add default data root
    PARSER.add_argument('--model_dir', default='experiments/base_model',
                        help="Directory containing the model hyperparams")

    ARGS = PARSER.parse_args()
    handleinput(ARGS)
