from __future__ import print_function

import numpy as np
import pandas as pd


class Utils:

    @staticmethod
    def read_data(path, cols_names):
        df = pd.read_csv(path, sep='\t', header=None, names=cols_names)
        return df

    @staticmethod
    def to_one_hot(labels, n_classes):
        labels = np.eye(n_classes)[labels.reshape(-1)]
        return labels

    @staticmethod
    def to_label(data):
        new_labels = []
        for row in data:
            new_labels.append(np.argmax(row))

        return np.array(new_labels)
