import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from resnet import ResNet18

logger = logging.getLogger()
logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')


def get_args():
    parser = argparse.ArgumentParser(description='Train the network')
    parser.add_argument('--gpu', '-g', action='store_true', default=False, help="Whether the GPU should be used or not")
    parser.add_argument('--epochs', '-e', metavar='INT', default=20, help="Amount of epochs")
    parser.add_argument('--optimizer', '-o', metavar='STRING', default="sgd", help="The optimizer you want to use")
    parser.add_argument('--job_id', '-j', metavar='STRING', default="", help="Job_id used for saving files")
    parser.add_argument('--root', '-d', metavar='STRING', default="./data/", help="Path to data")
    parser.add_argument('--output', '-p', metavar='STRING', default="./output/", help="Path for output")
    parser.add_argument('--split', '-s', metavar='FLOAT', default=0.9,
                        help="percentage of test set used for validation set")
    parser.add_argument('--seed', '-r', metavar='INT', default=1337, help="Random seed for shuffle")
    return parser.parse_args()


class DeepLearningComparison:

    def __init__(self):
        self.args = get_args()
        logger.info(self.args)
        self.train_loader = None
        self.test_loader = None
        self.valid_loader = None
        self.net = None
        self.criterion = None
        self.optimizer = None
        self.train_loss = np.array([])
        self.val_accuracy = np.array([])
        self.val_loss = np.array([])

    def load_data(self):
        logger.info("Start loading data")
        if not os.path.exists(self.args.root):
            os.makedirs(self.args.root)

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        # Loading the training data
        train_set = torchvision.datasets.CIFAR10(self.args.root, train=True, transform=transform,
                                                 target_transform=None, download=True)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=512, shuffle=True, num_workers=2)

        # Loading the test data
        valid_set = torchvision.datasets.CIFAR10(self.args.root, train=False, transform=transform,
                                                 target_transform=None, download=True)

        # Calculate the random split between validation and test
        num_test = len(valid_set)
        indices = list(range(num_test))
        split = int(np.floor(float(self.args.split) * num_test))

        # Seed for reproduction
        np.random.seed(int(self.args.seed))
        # Shuffle indices
        np.random.shuffle(indices)
        # Perform the actual split
        valid_idx, test_idx = indices[split:], indices[:split]
        # Setup samplers for validation and test
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)
        test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_idx)

        self.valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=512, num_workers=2, sampler=valid_sampler)
        self.test_loader = torch.utils.data.DataLoader(valid_set, batch_size=512, num_workers=2, sampler=test_sampler)

        logger.info("Loading data was successful")

    def load_network(self):
        logger.info("Start loading network, loss function and optimizer")

        # Load a network
        # self.net = VGG('VGG11')
        self.net = ResNet18()

        # Move network to GPU if needed
        if self.args.gpu:
            self.net.to('cuda')

        # Define the loss function and the optimizer
        self.criterion = nn.CrossEntropyLoss()

        if self.args.optimizer.lower() == 'adadelta':
            logger.info("Selected adadelta as optimizer")
            self.optimizer = optim.Adadelta(self.net.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
        elif self.args.optimizer.lower() == 'adagrad':
            logger.info("Selected adagrad as optimizer")
            self.optimizer = optim.Adagrad(self.net.parameters(), lr=0.01, lr_decay=0, weight_decay=0,
                                           initial_accumulator_value=0)
        elif self.args.optimizer.lower() == 'adam':
            logger.info("Selected adam as optimizer")
            self.optimizer = optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                        amsgrad=False)
        elif self.args.optimizer.lower() == 'sparseadam':
            logger.info("Selected sparseadam as optimizer")
            self.optimizer = optim.SparseAdam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
        elif self.args.optimizer.lower() == 'adamax':
            logger.info("Selected adamax as optimizer")
            self.optimizer = optim.Adamax(self.net.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08,
                                          weight_decay=0)
        elif self.args.optimizer.lower() == 'asgd':
            logger.info("Selected asgd as optimizer")
            self.optimizer = optim.ASGD(self.net.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0,
                                        weight_decay=0)
        elif self.args.optimizer.lower() == 'lbfgs':
            logger.info("Selected lbfgs as optimizer")
            self.optimizer = optim.LBFGS(self.net.parameters(), lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-05,
                                         tolerance_change=1e-09, history_size=100, line_search_fn=None)
        elif self.args.optimizer.lower() == 'rmsprop':
            logger.info("Selected rmsprop as optimizer")
            self.optimizer = optim.RMSprop(self.net.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0,
                                           momentum=0, centered=False)
        elif self.args.optimizer.lower() == 'rprop':
            logger.info("Selected rprop as optimizer")
            self.optimizer = optim.Rprop(self.net.parameters(), lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
        elif self.args.optimizer.lower() == 'sgd':
            logger.info("Selected sgd as optimizer")
            self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0, dampening=0, weight_decay=0,
                                       nesterov=False)
        else:
            logger.info("Unknown optimizer given, SGD is chosen instead.")
            self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

        logger.info("Loading network, loss function and %s optimizer was successful", self.args.optimizer)

    def train_network(self):
        logger.info("Start training network")

        self.net.train()
        epoch_loss = 0.0
        for i, data in enumerate(self.train_loader, 0):
            # Getting the inputs
            inputs, labels = data

            # Move inputs to GPU if needed
            if self.args.gpu:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forwards
            outputs = self.net(inputs)

            # Backwards
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # Save loss
            epoch_loss += loss.item()

        self.train_loss = np.append(self.train_loss, epoch_loss)
        logger.info("Training networks was successful")

    def eval_network(self, test):
        logger.info("Start evaluating the network")

        self.net.eval()
        total = 0
        correct = 0
        epoch_loss = 0.0

        if test:
            loader = self.test_loader
        else:
            loader = self.valid_loader

        with torch.no_grad():
            for i, data in enumerate(loader, 0):
                inputs, labels = data

                # Move inputs to GPU if needed
                if self.args.gpu:
                    inputs = inputs.to('cuda')
                    labels = labels.to('cuda')

                # Predict
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                _, predicted = outputs.max(1)

                # Count amount of correct labels
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Save loss
                epoch_loss += loss.item()

        if test:
            logger.info("Test loss: %f", epoch_loss)
        else:
            self.val_loss = np.append(self.val_loss, epoch_loss)

        # Calculate accuracy and log
        accuracy = 100. * correct / total
        if test:
            logger.info("Test successful, result: %f %%", accuracy)
        else:
            self.val_accuracy = np.append(self.val_accuracy, accuracy)
            logger.info("Validation successful, result: %f %%", accuracy)

    def save_output(self):
        if not os.path.exists(self.args.output):
            os.makedirs(self.args.output)

        np.savetxt(os.path.join(self.args.output, str(self.args.job_id) + '_train_loss.csv'),
                   self.train_loss, fmt='%10.5f')
        np.savetxt(os.path.join(self.args.output, str(self.args.job_id) + '_val_loss.csv'),
                   self.val_loss, fmt='%10.5f')
        np.savetxt(os.path.join(self.args.output, str(self.args.job_id) + '_accuracy.csv'),
                   self.val_accuracy, fmt='%10.5f')

    def run(self):
        logger.info("Start the run")
        self.load_data()
        self.load_network()

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10, verbose=True)

        for epoch in range(int(self.args.epochs)):
            logger.info("Starting on training/validation %d", epoch)
            self.train_network()

            logger.info("[%d] loss: %.3f", epoch, self.train_loss[-1])
            self.eval_network(test=False)
            scheduler.step(self.val_loss[-1])

        self.eval_network(test=True)
        self.save_output()


def main():
    comparison = DeepLearningComparison()
    comparison.run()


if __name__ == '__main__':
    main()
