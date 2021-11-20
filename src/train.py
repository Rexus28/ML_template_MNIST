#!python3

import joblib
import argparse
import pandas as pd
from sklearn import metrics
from model_dispatcher import models
from scaler_dispatcher import scalers

MODEL_LIST = list(models.keys())
SCALER_LIST = list(scalers.keys())


def load_fold(df, fold):
    # split the train and validation data
    train = df.query(f"kfold != {fold}").reset_index(drop=True)
    valid = df.query(f"kfold == {fold}").reset_index(drop=True)
    # now split the X and y data
    X_train = train.drop(["target", "kfold"], axis=1).values
    y_train = train["target"].values
    X_valid = valid.drop(["target", "kfold"], axis=1).values
    y_valid = valid["target"].values
    return X_train, y_train, X_valid, y_valid


def run_fold(df, fold, model, metric, save=False):
    # load the training and validation data for this fold
    X_train, y_train, X_valid, y_valid = load_fold(df, fold)
    # get and train the model
    clf = models[model]
    clf.fit(X_train, y_train)
    # assess the model on the validation data
    y_pred = clf.predict(X_valid)
    print(f"{model}: fold = {fold}")
    if metric == "sweep":
        evaluate_model_sweep(y_valid, y_pred)
    else:
        evaluate_model(y_valid, y_pred, metric)
    
    # save the model in specified
    if save is True:
        joblib.dump(clf, f"models/{model}_{fold}.bin")


def evaluate_model(y_true, y_pred, metric):
    metric_func = metrics.__dict__.get(metric)
    if metric == "f1_score":
        score = metric_func(y_true, y_pred, average="weighted")
    else:
        score = metric_func(y_true, y_pred)
    print(f"  {metric} = {score:.4f}")

    
def evaluate_model_sweep(y_true, y_pred):
    print(f"  Accuracy = {metrics.accuracy_score(y_true, y_pred):.4f}")
    print(f"  F1 score = {metrics.f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"  MCC Score = {metrics.matthews_corrcoef(y_true, y_pred):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="input/train.csv")
    parser.add_argument("--fold", type=int, default=None,
                        help="Pick a specific fold for training. If not"
                             " specified all folds are used.")
    parser.add_argument("--model", type=str, default="decision_tree",
                        choices=MODEL_LIST)
    parser.add_argument("--metric", type=str, default="accuracy_score")
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--scaler", type=str, default="none",
                        choices=SCALER_LIST)
   
    args = parser.parse_args()
    fold = args.fold
    # read the training file
    df = pd.read_csv(args.input)
    # scale the data
    scaler = scalers[args.scaler]
    df = scaler(df)

    if fold is None:
        for i in range(5):
            run_fold(df, i, model=args.model, metric=args.metric, save=args.save)
    else:
        run_fold(df, fold, model=args.model, metric=args.metric, save=args.save)
