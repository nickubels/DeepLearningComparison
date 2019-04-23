import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from vgg import VGG

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def get_args():
    parser = argparse.ArgumentParser(description='Train the network')
    parser.add_argument('--gpu', '-g', action='store_true', default=False, help="Whether the GPU should be used or not")
    parser.add_argument('--epochs', '-e', metavar='INT', default=20, help="Amount of epochs")
    parser.add_argument('--optimizer', '-o', metavar='STRING', default="", help="The optimizer you want to use")
    parser.add_argument('--job_id', '-j', metavar='STRING', default="", help="Job_id used for saving files")
    parser.add_argument('--root', '-d', metavar='STRING', default="./data/", help="Path to data")
    return parser.parse_args()


class DeepLearningComparison:

    def __init__(self):
        self.args = get_args()
        self.train_loader = None
        self.test_loader = None
        self.net = None
        self.criterion = None
        self.optimizer = None

    def load_data(self):
        logger.info("Start loading data")
        if not os.path.exists(self.args.root):
            os.makedirs(self.args.root)

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # Loading the training data
        train_set = torchvision.datasets.CIFAR10(self.args.root, train=True, transform=transform,
                                                 target_transform=None, download=True)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)

        # Loading the test data
        test_set = torchvision.datasets.CIFAR10(self.args.root, train=False, transform=transform,
                                                target_transform=None, download=True)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True, num_workers=2)
        logger.info("Loading data was successful")

    def load_network(self):
        logger.info("Start loading network, loss function and optimizer")
        # Load a network
        self.net = VGG('VGG19')

        # Move network to GPU if needed
        if self.args.gpu:
            self.net.to('cuda')

        # Define the loss function and the optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        logger.info("Loading network, loss function and optimizer was successful")

    def train_network(self):
        logger.info("Start training network")
        running_loss = 0.0
        for epoch in range(int(self.args.epochs)):
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

                if i % 50 == 49:  # print every 50 mini-batches
                    logging.info("[%d, %5d] loss: %.3f", epoch + 1, i + 1, epoch_loss / 50)
                epoch_loss = 0.0
        logger.info("Training networks was successful")

    def eval_network(self):
        logger.info("Start evaluating the network")
        net.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader, 0):
                inputs, labels = data

                # Move inputs to GPU if needed
                if self.args.gpu:
                    inputs = inputs.to('cuda')
                    labels = labels.to('cuda')

                # Predict
                outputs = self.net(inputs)
                _, predicted = outputs.max(1)

                # Count amount of correct labels
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        # Calculate accuracy and log
        accuracy = 100.*correct/total
        logger.info("Evaluation succesful, result: ")
        logger.info(accuracy)

    def run(self):
        logger.info("This is a test")
        self.load_data()
        self.load_network()
        self.train_network()
        self.eval_network


if __name__ == '__main__':
    logger.info("This is a test")
    comparison = DeepLearningComparison()
    comparison.run()
