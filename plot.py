import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description='Plot the results')
    parser.add_argument('--input_path', '-i', metavar='STRING', default='output', help="Where we find the input")
    parser.add_argument('--output_path', '-o', metavar='STRING', default='output', help="Where we store the output")
    return parser.parse_args()


class Plotter(object):
    """Plotter"""
    def __init__(self):
        self.args = get_args()
        self.fileNames = os.listdir(self.args.input_path)
        self.accuracy_files = [file for file in self.fileNames if 'accuracy.csv' in file]
        self.validation_files = [file for file in self.fileNames if 'val_loss.csv' in file]
        self.train_files = [file for file in self.fileNames if 'train_loss.csv' in file]

    def plot_accuracy(self):
        ### Loop over all files
        for file in self.accuracy_files:

            ### Read .csv file and append to list
            df = pd.read_csv(os.path.join(self.args.input_path, file))

            ### Create line for every file
            plt.plot(df)

        ### Generate the plot
        plt.show()

    def plot_validation(self):
        ### Loop over all files
        for file in self.validation_files:

            ### Read .csv file and append to list
            df = pd.read_csv(os.path.join(self.args.input_path, file))

            ### Create line for every file
            plt.plot(df)

        ### Generate the plot
        plt.show()

    def plot_training(self):
        ### Loop over all files
        for file in self.train_files:

            ### Read .csv file and append to list
            df = pd.read_csv(os.path.join(self.args.input_path, file))

            ### Create line for every file
            plt.plot(df)

        ### Generate the plot
        plt.show()

def main():
    plotter = Plotter()
    plotter.plot_accuracy()
    plotter.plot_training()
    plotter.plot_validation()


if __name__ == '__main__':
    main()
                