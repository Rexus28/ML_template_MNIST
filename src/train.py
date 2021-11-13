#!python3

import joblib
import argparse
import pandas as pd
from sklearn import metrics
from model_dispatcher import models

MODEL_LIST = list(models.keys())

def run_fold(fold, model, metric, save=False, train_file="input/train.csv"):
    # read the training file
    df = pd.read_csv(train_file)
    
    # split the train and validation data
    train = df.query(f"kfold != {fold}").reset_index(drop=True)
    valid = df.query(f"kfold == {fold}").reset_index(drop=True)
    
    # now split the X and y data
    X_train = train.drop(["target", "kfold"], axis=1).values
    y_train = train["target"].values
    X_valid = valid.drop(["target", "kfold"], axis=1).values
    y_valid = valid["target"].values
    
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
    parser.add_argument("--fold", type=int, default=None,
                        help="Pick a specific fold for training. If not"
                             " specified all folds are used.")
    parser.add_argument("--model", type=str, default="decision_tree",
                        choices=MODEL_LIST)
    parser.add_argument("--metric", type=str, default="accuracy_score")
    parser.add_argument("--save", action='store_true')
   
    args = parser.parse_args()
    fold = args.fold
    if fold is None:
        for i in range(5):
            run_fold(i, model=args.model, metric=args.metric, save=args.save)
    else:
        run_fold(fold, model=args.model, metric=args.metric, save=args.save)
