import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description='Plot the results')
    parser.add_argument('--input_path', '-i', metavar='STRING', default='plots/output', help="Where we find the input")
    parser.add_argument('--log_path', '-l', metavar='STRING', default='plots/logs', help="Where we find the logs")
    parser.add_argument('--output_path', '-o', metavar='STRING', default='plots', help="Where we store the output")
    return parser.parse_args()


class Plotter(object):
    """Plotter"""
    def __init__(self):
        self.args = get_args()
        self.fileNames = os.listdir(self.args.input_path)
        self.accuracy_files = [file for file in self.fileNames if 'accuracy.csv' in file]
        self.validation_files = [file for file in self.fileNames if 'val_loss.csv' in file]
        self.train_files = [file for file in self.fileNames if 'train_loss.csv' in file]
        self.labels = {}

    def plot_accuracy(self):
        # Loop over all files
        for file in self.accuracy_files:
            # Obtain label
            label = self.labels.get(file.split('_')[0], None)

            # Read .csv file and append to list
            df = pd.read_csv(os.path.join(self.args.input_path, file))

            # Create line for every file
            plt.plot(df, label=label)

        # Generate the plot
        plt.legend()
        plt.ylabel('Accuracy in %')
        plt.xlabel('Epochs')
        # plt.show()
        plt.savefig(os.path.join(self.args.output_path, 'accuracy.png'))

    def plot_validation(self):
        # Loop over all files
        for file in self.validation_files:
            # Obtain label
            label = self.labels.get(file.split('_')[0], None)

            # Read .csv file and append to list
            df = pd.read_csv(os.path.join(self.args.input_path, file))

            # Create line for every file
            plt.plot(df, label=label)

        # Generate the plot
        plt.legend()
        plt.ylabel('Validation loss')
        plt.xlabel('Epochs')
        # plt.show()
        plt.savefig(os.path.join(self.args.output_path, 'validation.png'))

    def plot_training(self):
        # Loop over all files
        for file in self.train_files:
            # Obtain label
            label = self.labels.get(file.split('_')[0], None)

            # Read .csv file and append to list
            df = pd.read_csv(os.path.join(self.args.input_path, file))

            # Create line for every file
            plt.plot(df, label=label)

        # Generate the plot
        plt.legend()
        plt.ylabel('Training loss')
        plt.xlabel('Epochs')
        # plt.show()
        plt.savefig(os.path.join(self.args.output_path, 'train.png'))

    def obtain_labels(self):
        files = os.listdir(self.args.log_path)

        for log_file in files:
            with open(os.path.join(self.args.log_path, log_file)) as f:
                first_line = f.readline()
                first_line = first_line.split()
                job_id = [s for s in first_line if "job_id" in s]
                job_id = job_id[0][8:-2]
                optimizer = [s for s in first_line if "optimizer" in s]
                optimizer = optimizer[0][11:-2]
                self.labels[job_id] = optimizer


def main():
    plotter = Plotter()
    plotter.obtain_labels()
    plotter.plot_accuracy()
    plotter.plot_training()
    plotter.plot_validation()


if __name__ == '__main__':
    main()
