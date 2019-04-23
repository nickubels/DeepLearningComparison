import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os
from vgg import VGG


def get_args():
    parser = argparse.ArgumentParser(description='Train the network')
    parser.add_argument('--gpu', '-g' ,action='store_true', default=False, help="Whether the GPU should be used or not")
    parser.add_argument('--epochs', '-e' ,metavar='INT', default=20, help="Amount of epochs")
    parser.add_argument('--optimizer', '-o' ,metavar='STRING', default="", help="The optimizer you want to use")
    parser.add_argument('--jobid' ,'-j' ,metavar='STRING', default="" , help="Jobid used for saving files")
    return parser.parse_args()


class DeepLearningComparison:
    def __init__(self):
        self.args = get_args()

        data_path = './data/'
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])

        # Loading the training data
        trainset = torchvision.datasets.CIFAR10(data_path, train=True, transform=transform, target_transform=None, download=True)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

        # Loading the test data
        testset = torchvision.datasets.CIFAR10(data_path, train=False, transform=transform, target_transform=None, download=True)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

        # Load a network
        self.net = VGG('VGG19')

        # Define the loss function and the optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

    def train_network(self):
        for epoch in range(self.args.epochs):
            epoch_loss = 0.0
            for i,data in enumerate(self.trainloader, 0):
                # Getting the inputs
                inputs, labels = data

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

                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


if __name__ == '__main__':
    comparison = DeepLearningComparison()
    comparison.train_network()
