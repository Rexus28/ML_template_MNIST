#!python3

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


def save_split(X, y, filename):
    # save the split of X data and y labels as a csv
    df = pd.DataFrame(X)
    df['target'] = y
    df.to_csv(filename, index=False)


def stratified_train_test_split(df, train_file, test_file):
    X = df.drop('target', axis=1).values
    y = df['target'].values
    # create train and test splits for data X and labels y, including
    # stratifying the split for the labels to preserve the same ratio of
    # labels in both train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        stratify=y)
    # save the splits to csvs
    save_split(X_train, y_train, train_file)
    save_split(X_test, y_test, test_file)

    
def create_folds(train_file, bin_targets=False):
    # read the training file
    df = pd.read_csv(train_file)
    
    # add column for the fold number
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df["target"].values
    
    if bin_targets is True:
        num_bins = int(np.floor(1 + np.log2(len(df))))
        df.loc[:, "bins"] = pd.cut(df["target"], bins=num_bins, labels=False)
        y = df["bins"].values
    
    # create the stratifield folds and iterate through them to label the df
    kf = StratifiedKFold(n_splits=5)
    for i, (train_indx, test_indx) in enumerate(kf.split(X=df, y=y)):
        df.loc[test_indx, 'kfold'] = i
    
    if bin_targets is True:
        df = df.drop("bins", axis=1)
    df.to_csv(train_file, index=False)


if __name__ == "__main__":
    filename = "input/mnist_784.csv"
    df = pd.read_csv(filename)
    stratified_train_test_split(df, "input/train.csv", "input/test.csv")
    create_folds("input/train.csv")
