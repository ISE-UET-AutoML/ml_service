import sys

import optuna
import pandas as pd
import requests
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, \
    classification_report, log_loss, matthews_corrcoef, balanced_accuracy_score
import time
import os
import argparse

if __name__ == "__main__":
    start = time.time()

    DATA_CENTER_URL = os.environ.get("DATA_CENTER")
    BUCKET_NAME = os.environ.get("BUCKET_NAME")
    PROJECT_NAME = os.environ.get("PROJECT_NAME")
    ALGORITHM = "Random-Forest"

    try:
        parser = argparse.ArgumentParser()

        parser.add_argument("--data-path", type=str,
                            required=True, help="Path to training data")

        parser.add_argument("--target", type=str,
                            default="Survived", help="Name of the target column")

        args = parser.parse_args()

        titanic_data = pd.read_csv(args.data_path)

        X = titanic_data.drop(columns=[args.target])
        y = titanic_data[args.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 14)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 14)
            max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])  # Adjusted line

            classifier = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42
            )

            score = cross_val_score(classifier, X_train, y_train, n_jobs=-1, cv=3, scoring='accuracy').mean()
            return score


        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)

        print('Number of finished trials:', len(study.trials))
        print('Best trial:', study.best_trial.params)

        # Train the model with the best parameters
        best_params = study.best_trial.params
        best_model = RandomForestClassifier(**best_params, random_state=42)
        best_model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        responses_body = {
            "Algorithm": "Random Forest",
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
        requests.post(DATA_CENTER_URL, json={"error": str(e)})
        print(e)
        sys.exit(1)
