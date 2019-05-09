import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description='Calculate the early stopping epoch')
    parser.add_argument('--input_path', '-i', metavar='STRING', default='output', help="Where we find the input")
    parser.add_argument('--log_path', '-l', metavar='STRING', default='logs', help="Where we find the logs")
    # parser.add_argument('--output_path', '-o', metavar='STRING', default='plots/plots', help="Where we store the output")
    return parser.parse_args()


class EarlyStopping(object):
    """Plotter"""
    def __init__(self):
        self.args = get_args()
        self.fileNames = os.listdir(self.args.input_path)
        self.labels = {}

    def calc_es(self):
        for label, file in self.labels:
            df_train = pd.read_csv(os.path.join(self.args.input_path, file + '_train_loss.csv'),header = None)
            df_val = pd.read_csv(os.path.join(self.args.input_path, file + '_val_loss.csv'),header = None)
            for index, row in df_train.iterrows():
                if index > 4:
                    continue
                    # print(df_val.iloc[index])
                    # if df_val[index]:
                print(index, row)
            # for epoch in df_train:
            #     print(epoch)


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
    plotter = EarlyStopping()
    plotter.obtain_labels()
    plotter.calc_es()




if __name__ == '__main__':
    main()
