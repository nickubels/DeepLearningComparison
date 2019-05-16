import argparse
import os
import math
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description='Calculate the early stopping epoch')
    parser.add_argument('--input_path', '-i', metavar='STRING', default='datadl/output', help="Where we find the input")
    parser.add_argument('--log_path', '-l', metavar='STRING', default='datadl/logs', help="Where we find the logs")
    parser.add_argument('--output_path', '-o', metavar='STRING', default='plots/tables',
                        help="Where we store the output")
    parser.add_argument('--n', '-n', metavar='INT', default=10,
                        help="Number of epochs over which we find for a decrease in validation loss")
    return parser.parse_args()


class EarlyStopping(object):
    """Plotter"""

    def __init__(self):
        self.args = get_args()
        self.fileNames = os.listdir(self.args.input_path)
        self.labels = {}

    def get_early_stopping_epoch(self, validation):
        """
        This function finds the epoch in which you could early stop,
        because there has not been any validation loss decrease in the last n epochs.
        """
        cnt = 0
        last = math.inf

        # Loop over the validation loss to see if we could stop earlier
        for idx, row in validation.iterrows():
            cnt = cnt + 1

            # Check if there has been a decrease with the previous epoch
            if row[0] < last:
                cnt = 0
                last = row[0]

            # If in the last n epoch there wasn't any decrease, we can early stop
            if cnt == self.args.n:
                return idx

        # Early stopping wasn't possible
        return idx

    def get_table(self):
        """
        This function saves a table in which each row is a different optimizer, and the columns include the
        earling stopping epoch, the corresponding accuracy and the accuracy after 100 epochs.
        """
        # First define the information we need in our table
        early_epoch = []
        early_acc = []
        acc = []
        labels = []

        # Loop over all the optimizers to find the information we want
        for label, file in self.labels:
            # Load the validation loss and accuracy for the selected optimizer
            df_val = pd.read_csv(os.path.join(self.args.input_path, file + '_val_loss.csv'), header=None)
            df_acc = pd.read_csv(os.path.join(self.args.input_path, file + '_accuracy.csv'), header=None)

            # Get the epoch in which we can early stop for the selected optimizer and the corresponding accuracies
            epoch = self.get_early_stopping_epoch(df_val)
            early_epoch.append(epoch)
            early_acc.append(df_acc.iloc[epoch][0])
            acc.append(df_acc.iloc[-1][0])

            # Also store the optimizer we used
            labels.append(label)

        # Create a dataframe from the data we collected
        data = {'Early Stopping Epoch': early_epoch,
                'Early Stopping Accuracy (%)': early_acc,
                'Accuracy (%)': acc}
        df = pd.DataFrame(data, index=labels)

        # Store the dataframe as csv table
        df.to_csv(os.path.join(self.args.output_path, 'accuracy_with_early.csv'))

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


def main():
    plotter = EarlyStopping()
    plotter.obtain_labels()
    plotter.get_table()


if __name__ == '__main__':
    main()
