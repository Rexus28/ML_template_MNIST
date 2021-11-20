#!python3

import csv
import time
import tqdm
import train
import joblib
import argparse
import itertools
import pandas as pd
from sklearn import metrics
from model_dispatcher import models
from scaler_dispatcher import scalers

MODEL_LIST = list(models.keys())
SCALER_LIST = list(scalers.keys())


def run_fold(df, fold, model, results_dict, writer):
    # load the training and validation data for this fold
    X_train, y_train, X_valid, y_valid = train.load_fold(df, fold)

    # get and train the model
    clf = models[model]
    start_time = time.time()
    clf.fit(X_train, y_train)
    results_dict["train_time"] = time.time() - start_time
    
    # assess the model on the validation data
    start_time = time.time()
    y_pred = clf.predict(X_valid)
    results_dict["test_time"] = time.time() - start_time
    evaluate_model_sweep(y_valid, y_pred, results_dict)
    writer.writerow(results_dict)


def evaluate_model_sweep(y_true, y_pred, results_dict):
    results_dict["acc"] = metrics.accuracy_score(y_true, y_pred)
    results_dict["f1"] = metrics.f1_score(y_true, y_pred, average='weighted')
    results_dict["mcc"] = metrics.matthews_corrcoef(y_true, y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="input/train.csv")
    parser.add_argument("--output", type=str, default="experiment.csv")
    
    args = parser.parse_args()
    # read the training file
    df = pd.read_csv(args.input)

    csv_file = open(args.output, 'w')
    fieldnames = ["model", "scaler", "fold", "acc", "f1", "mcc", "train_time", "test_time"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    
    # MODEL_LIST = ["decision_tree", "random_forest_5", "naive_bayes"]
    # SCALER_LIST = ["none"]
    folds = range(5)
    for m, s, f in tqdm.tqdm(itertools.product(MODEL_LIST, SCALER_LIST, folds)):
        print(f"running {m}, {s} scaler, fold {f}")
        results_dict = {"model": m, "scaler": s, "fold": f, "acc": -1,
                        "f1": -1, "mcc": -1, "train_time": -1, "test_time": -1}
        # scale the data
        scaler = scalers[s]
        run_fold(scaler(df), f, m, results_dict, writer)

    csv_file.close()
