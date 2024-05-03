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

if __name__ == "__main__":
    start = time.time()
    DATA_CENTER_URL = os.environ.get("DATA_CENTER")
    BUCKET_NAME = os.environ.get("BUCKET_NAME")
    PROJECT_NAME = os.environ.get("PROJECT_NAME")
    ALGORITHM = "KNN"


    def knn_cv(n_neighbors, n_cv=5):
        val = cross_val_score(
            KNeighborsClassifier(n_neighbors=int(n_neighbors)),
            X_train, y_train, scoring='accuracy', cv=n_cv
        ).mean()
        return val


    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--data-path", type=str,
                            required=True, help="Path to training data")

        parser.add_argument("--target", type=str,
                            default="Survived", help="Name of the target column")

        parser.add_argument("--model-path", type=str, default="models",
                            help="Path to save model")

        parser.add_argument("--test-size", type=float,
                            default=0.2, help="Size of the test dataset")

        parser.add_argument("--random-state", type=int, default=42,
                            help="Random state for splitting data")

        parser.add_argument("--n-iter", type=int,
                            default=10, help="Number of iterations for Bayesian Optimization")

        parser.add_argument("--n-neighbors", type=int,
                            default=5, help="Number of neighbors for KNN")

        parser.add_argument("--n-cv", type=int,
                            default=5, help="Number of cross-validation folds")

        args = parser.parse_args()

        if DATA_CENTER_URL is None:
            raise Exception("Data Center URL is not provided")

        titanic_data = pd.read_csv(args.data_path)

        X = titanic_data.drop(columns=[args.target])
        y = titanic_data[args.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        optimizer = BayesianOptimization(
            f=lambda n_neighbors: knn_cv(n_neighbors, n_cv=args.n_cv),
            pbounds={'n_neighbors': (1, args.n_neighbors)},
            random_state=args.random_state,
            verbose=2
        )

        optimizer.maximize(init_points=2, n_iter=args.n_iter)

        print(f"Best parameters: {optimizer.max['params']}")

        best_n_neighbors = int(optimizer.max['params']['n_neighbors'])
        best_model = KNeighborsClassifier(n_neighbors=best_n_neighbors)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        responses_body = {
            "Algorithm": "KNN",
            "Accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "mcc": mcc,
            "roc_auc": roc_auc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "time": time.time() - start,
        }
        response = requests.post(DATA_CENTER_URL, json=responses_body)
    except Exception as e:
        response = requests.post(DATA_CENTER_URL, json={"Error": str(e)})
        sys.exit(1)
