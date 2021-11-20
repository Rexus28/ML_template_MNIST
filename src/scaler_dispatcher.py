#!python3

import pandas as pd
from sklearn import preprocessing

def reconstruct_df(df, data_df, scaled_data):
    df_scaled = pd.DataFrame(scaled_data, columns=data_df.columns)
    df_scaled["target"] = df["target"]
    df_scaled["kfold"] = df["kfold"]
    return df_scaled


def min_max_scale(df):
    data_df = df.drop(["target", "kfold"], axis=1)
    data = data_df.values
    scaled_data = (data - data.min()) / (data.max() - data.min())
    return reconstruct_df(df, data_df, scaled_data)


def standard_scale(df):
    data_df = df.drop(["target", "kfold"], axis=1)
    data = data_df.values
    m = data.mean(axis=0)
    s = data.std(axis=0) + 1e-8
    scaled_data = (data - m) / s
    return reconstruct_df(df, data_df, scaled_data)


def no_scale(df):
    return df


scalers = {
    "minmax": min_max_scale,
    "standard": standard_scale,
    "none": no_scale,
}
