#!python3

import numpy as np
import pandas as pd
from sklearn import datasets


def get_mnist_data(frac=1):
    # grab the mnist dataset via sklearn and create a pandas dataframe
    data = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
    df = pd.DataFrame(data[0])
    df['target'] = data[1]
    
    # return rows corresponding to frac percentage of all rows
    n_rows = len(df)
    n = int(frac * n_rows)
    return df.head(n)


if __name__ == "__main__":
    # df = get_mnist_data(frac=1)
    df = get_mnist_data(frac=1./7.)
    df.to_csv('input/mnist_784.csv', index=False)
