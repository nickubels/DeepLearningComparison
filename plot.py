import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description='Plot the results')
    parser.add_argument('--input_path', '-i', metavar='STRING', default='plots/output', help="Where we find the input")
    parser.add_argument('--log_path', '-l', metavar='STRING', default='plots/logs', help="Where we find the logs")
    parser.add_argument('--output_path', '-o', metavar='STRING', default='plots/plots', help="Where we store the output")
    return parser.parse_args()


class Plotter(object):
    """Plotter"""
    def __init__(self):
        self.args = get_args()
        self.fileNames = os.listdir(self.args.input_path)
        self.labels = {}

    def plot_accuracy(self):
        # Loop over all files
        plt.figure(figsize=(25, 10))
        plt.rc('font', size=18)  # controls default text sizes
        plt.rc('axes', titlesize=18)  # fontsize of the axes title
        # plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('legend', fontsize=18)  # legend fontsize

        for label, file in self.labels:
            # Read .csv file and append to list
            df = pd.read_csv(os.path.join(self.args.input_path, file + '_accuracy.csv'))

        # Create line for every file
            plt.plot(df, label=label)

        # Generate the plot
        plt.legend()
        plt.ylabel('Accuracy in %')
        plt.xlabel('Epochs')
        # plt.show()
        plt.savefig(os.path.join(self.args.output_path, 'accuracy.png'))
        plt.close()

    def plot_validation(self):
        # Loop over all files
        plt.figure(figsize=(25, 10))
        plt.rc('font', size=18)  # controls default text sizes
        plt.rc('axes', titlesize=18)  # fontsize of the axes title
        # plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('legend', fontsize=18)  # legend fontsize

        for label, file in self.labels:
            # Read .csv file and append to list
            df = pd.read_csv(os.path.join(self.args.input_path, file + '_val_loss.csv'))

            # Create line for every file
            plt.plot(df, label=label)

        # Generate the plot
        plt.legend()
        plt.ylabel('Validation loss')
        plt.xlabel('Epochs')
        # plt.show()
        plt.savefig(os.path.join(self.args.output_path, 'validation.png'))
        plt.close()

    def plot_training(self):
        # Loop over all files
        plt.figure(figsize=(25, 10))
        plt.rc('font', size=18)  # controls default text sizes
        plt.rc('axes', titlesize=18)  # fontsize of the axes title
        # plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('legend', fontsize=18)  # legend fontsize

        for label, file in self.labels:
            # Read .csv file and append to list
            df = pd.read_csv(os.path.join(self.args.input_path, file + '_train_loss.csv'))

            # Create line for every file
            plt.plot(df, label=label)

        # Generate the plot
        plt.legend()
        plt.ylabel('Training loss')
        plt.xlabel('Epochs')
        # plt.show()
        plt.savefig(os.path.join(self.args.output_path, 'train.png'))
        plt.close()

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
                # self.labels[job_id] = optimizer
                self.labels[optimizer] = job_id
        self.labels = sorted(self.labels.items(), key=lambda s: s[0])


def main():
    plotter = Plotter()
    plotter.obtain_labels()
    plotter.plot_accuracy()
    plotter.plot_training()
    plotter.plot_validation()


if __name__ == '__main__':
    main()
