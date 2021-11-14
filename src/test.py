#!python3

import os
import joblib
import argparse
import pandas as pd
from sklearn import metrics
from model_dispatcher import models
from train import evaluate_model, evaluate_model_sweep

MODEL_LIST = list(models.keys())


def run_test(fold, model, metric):
    model_file = f"models/{model}_{fold}.bin"
    if not os.path.exists(model_file):
        raise ValueError(f"Saved {model} model file does not exist. Please "
                         "train a model before testing.")
    clf = joblib.load(model_file)
    
    # read the training file
    df = pd.read_csv("input/test.csv")
    
    # now split the X and y data
    X_test = df.drop(["target"], axis=1).values
    y_test = df["target"].values

    # assess the model on the validation data
    y_pred = clf.predict(X_test)
    print(f"{model}: fold = {fold}")
    if metric == "sweep":
        evaluate_model_sweep(y_test, y_pred)
    else:
        evaluate_model(y_test, y_pred, metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None,
                        help="Pick a specific fold for training. If not"
                             " specified all folds are used.")
    parser.add_argument("--model", type=str, default="decision_tree",
                        choices=MODEL_LIST)
    parser.add_argument("--metric", type=str, default="accuracy_score")
   
    args = parser.parse_args()
    fold = args.fold
    if fold is None:
        for i in range(5):
            run_test(i, model=args.model, metric=args.metric)
    else:
        run_test(fold, model=args.model, metric=args.metric)