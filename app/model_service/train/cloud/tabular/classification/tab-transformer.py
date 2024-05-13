import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, \
    classification_report, log_loss, matthews_corrcoef, balanced_accuracy_score
from bayes_opt import BayesianOptimization
import os
import subprocess
import sys
import argparse
import requests
import time
import torch
from tabpfn import TabPFNClassifier

if __name__ == "__main__":
    start = time.time()
    DATA_CENTER_URL = os.environ.get("DATA_CENTER")
    BUCKET_NAME = os.environ.get("BUCKET_NAME")
    PROJECT_NAME = os.environ.get("PROJECT_NAME")
    ALGORITHM = "Tab-Transformer"
    try:
        parser = argparse.ArgumentParser()

        parser.add_argument("--data-path", type=str,
                            required=True, help="Path to training data")

        parser.add_argument("--target", type=str,
                            default="Survived", help="Name of the target column")

        parser.add_argument("--ensemble", type=int,
                            default=3, help="Number of ensemble configurations")

        args = parser.parse_args()

        titanic_data = pd.read_csv(args.data_path)

        X = titanic_data.drop(columns=[args.target])
        y = titanic_data[args.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        classifier = TabPFNClassifier(device="cuda", N_ensemble_configurations=args.ensemble) if torch.cuda.is_available() else TabPFNClassifier(device="cpu", N_ensemble_configurations=args.ensemble)

        classifier.fit(X_train, y_train, overwrite_warning= True)

        y_val, y_prob = classifier.predict(X_test, return_winning_probability=True)

        accuracy = accuracy_score(y_test, y_val)

        mcc = matthews_corrcoef(y_test, y_val)

        roc_auc = roc_auc_score(y_val, y_val)
        balanced_accuracy = balanced_accuracy_score(y_val, y_val)
        f1 = f1_score(y_val, y_val)
        precision = precision_score(y_val, y_val)
        recall = recall_score(y_val, y_val)

        response_body = {
            "Algorithm": "Tabular Transformer",
            "Accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "mcc": mcc,
            "roc_auc": roc_auc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "time": time.time() - start,
        }
        response = requests.post(DATA_CENTER_URL, json=response_body)


    except Exception as e:
        response = requests.post(DATA_CENTER_URL, json={"error": str(e)})
        print(e)
