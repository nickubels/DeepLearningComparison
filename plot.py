import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description='Plot the results')
    parser.add_argument('--input_path', '-i', metavar='STRING', default='plots/output', help="Where we find the input")
    parser.add_argument('--log_path', '-l', metavar='STRING', default='plots/logs', help="Where we find the logs")
    parser.add_argument('--output_path', '-o', metavar='STRING', default='plots/plots',
                        help="Where we store the output")
    return parser.parse_args()


class Plotter(object):
    """Plotter"""

    def __init__(self):
        self.args = get_args()
        self.fileNames = os.listdir(self.args.input_path)
        self.labels = {}

    def plot_accuracy(self):
        """
        This function plots accuracy over epochs.
        """
        # Set the figure size and its font sizes
        plt.figure(figsize=(25, 10))
        plt.rc('font', size=26)  # controls default text sizes
        plt.rc('axes', titlesize=26)  # fontsize of the axes title
        plt.rc('legend', fontsize=26)  # legend fontsize

        # Loop over all files
        for label, file in self.labels:
            # Read .csv file and plot the content
            df = pd.read_csv(os.path.join(self.args.input_path, file + '_accuracy.csv'), header=None)
            plt.plot(df, label=label)

        # Generate the plot
        leg = plt.legend()
        plt.ylabel('Accuracy in %')
        plt.xlabel('Epochs')

        # bulk-set the properties of all lines
        leg_lines = leg.get_lines()
        plt.setp(leg_lines, linewidth=4)

        plt.savefig(os.path.join(self.args.output_path, 'accuracy.png'), bbox_inches='tight')
        plt.close()

    def plot_validation(self):
        """
        This function plots the validation loss over epochs.
        """
        # Set the figure size and its font sizes
        plt.figure(figsize=(25, 10))
        plt.rc('font', size=26)  # controls default text sizes
        plt.rc('axes', titlesize=26)  # fontsize of the axes title
        plt.rc('legend', fontsize=26)  # legend fontsize

        # Loop over all files
        for label, file in self.labels:
            # Read .csv file and plot its content
            df = pd.read_csv(os.path.join(self.args.input_path, file + '_val_loss.csv'), header=None)
            plt.plot(df, label=label)

        # Generate the plot
        leg = plt.legend()
        plt.ylabel('Validation loss')
        plt.xlabel('Epochs')

        # bulk-set the properties of all lines
        leg_lines = leg.get_lines()
        plt.setp(leg_lines, linewidth=4)

        plt.savefig(os.path.join(self.args.output_path, 'validation.png'), bbox_inches='tight')
        plt.close()

    def plot_training(self):
        """
        This functions plots training loss over epochs.
        """
        # Set the figure size and its font sizes
        plt.figure(figsize=(25, 10))
        plt.rc('font', size=26)  # controls default text sizes
        plt.rc('axes', titlesize=26)  # fontsize of the axes title
        plt.rc('legend', fontsize=26)  # legend fontsize

        # Loop over all files
        for label, file in self.labels:
            # Read .csv file and plot its content
            df = pd.read_csv(os.path.join(self.args.input_path, file + '_train_loss.csv'), header=None)
            plt.plot(df, label=label)

        # Generate the plot
        leg = plt.legend()
        plt.ylabel('Training loss')
        plt.xlabel('Epochs')

        # bulk-set the properties of all lines
        leg_lines = leg.get_lines()
        plt.setp(leg_lines, linewidth=4)

        plt.savefig(os.path.join(self.args.output_path, 'train.png'), bbox_inches='tight')
        plt.close()

    def obtain_labels(self):
        """
        This function creates a dict with as key an optimizer and as value the corresponding file.
        """
        # First get all the logging files
        files = os.listdir(self.args.log_path)

        # From each logging file, get the optimizer and corresponding job_id == file
        for log_file in files:
            with open(os.path.join(self.args.log_path, log_file)) as f:
                # We know the information is in the first line
                first_line = f.readline()
                first_line = first_line.split()

                # Get the job_id
                job_id = [s for s in first_line if "job_id" in s]
                job_id = job_id[0][8:-2]

                # Get the optimizer
                optimizer = [s for s in first_line if "optimizer" in s]
                optimizer = optimizer[0][11:-2]

                # Create a key for the optimizer
                self.labels[optimizer] = job_id

        # Sort the dictionary based on keys, which makes the legend nicer
        self.labels = sorted(self.labels.items(), key=lambda s: s[0])

        # Create map for plots
        os.makedirs(self.args.output_path, exist_ok=True)


def main():

    plotter = Plotter()
    plotter.obtain_labels()
    plotter.plot_accuracy()
    plotter.plot_training()
    plotter.plot_validation()


if __name__ == '__main__':
    main()
