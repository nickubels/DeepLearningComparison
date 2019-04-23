import argparse
import torch
import torch.nn as nn

def get_args():
    parser = argparse.ArgumentParser(description='Train the network')
    parser.add_argument('--gpu', '-g',action='store_true', default=False,
        help="Wether the GPU should be used or not")
    parser.add_argument('--epochs', '-e',metavar='INT', default=20,
        help="Amount of epochs")
    parser.add_argument('--optimizer', '-o',metavar='STRING', default="",
        help="The optimizer you want to use")
    parser.add_argument('--jobid','-j',metavar='STRING', default="",help="Jobid used for saving files")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()