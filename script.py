import argparse
import torch
import torchvision
import torch.nn as nn
import os

def get_args():
    parser = argparse.ArgumentParser(description='Train the network')
    parser.add_argument('--gpu', '-g',action='store_true', default=False,
        help="Whether the GPU should be used or not")
    parser.add_argument('--epochs', '-e',metavar='INT', default=20,
        help="Amount of epochs")
    parser.add_argument('--optimizer', '-o',metavar='STRING', default="",
        help="The optimizer you want to use")
    parser.add_argument('--jobid','-j',metavar='STRING', default="",help="Jobid used for saving files")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    data_path = './data/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Loading the training data
    trainset = torchvision.datasets.FashionMNIST(data_path, train=True, transform=None, target_transform=None, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    # Loading the test data
    testset = torchvision.datasets.FashionMNIST(data_path, train=False, transform=None, target_transform=None, download=True)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
