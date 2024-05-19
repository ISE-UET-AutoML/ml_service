import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, \
    classification_report, log_loss, matthews_corrcoef, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import argparse
from sklearn import ensemble
from pathlib import Path
import os
import numpy as np
from datetime import datetime
import requests
import time
import joblib
import subprocess
import xgboost
import sys

if __name__ == "__main__":
    start = time.time()
    DATA_CENTER_URL = os.environ.get("DATA_CENTER")
    BUCKET_NAME = os.environ.get("BUCKET_NAME")
    PROJECT_NAME = os.environ.get("PROJECT_NAME")
    ALGORITHM = "XGBoost"
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--data-path", type=str,
                            required=True, help="Path to training data")
        parser.add_argument("--target", type=str,
                            default="Survived", help="Name of the target column")

        parser.add_argument("--model-path", type=str, default="models",
                            help="Path to save model"
                            )

        parser.add_argument("--test-size", type=float,
                            default=0.2, help="Size of the test dataset")
        parser.add_argument("--random-state", type=int, default=42,
                            help="Random state for splitting data")
        parser.add_argument("--n-estimators", type=int,
                            default=100, help="Number of trees in the forest")
        parser.add_argument("--learning-rate", type=float,
                            default=1.0, help="Learning rate")
        parser.add_argument("--max-depth", type=int, default=1,
                            help="Maximum depth of the trees")

        args = parser.parse_args()

        if DATA_CENTER_URL is None:
            raise Exception("Data Center URL is not provided")

        train_data = pd.read_csv(args.data_path)

        X = train_data.drop(columns=[args.target])

        y = train_data[args.target]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state)


        model = xgboost.XGBClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            random_state=args.random_state
        )

        clf = Pipeline(steps=[
            ('model', model)]
        )

        clf.fit(X_train, y_train)

        preds = clf.predict(X_val)

        score = accuracy_score(y_val, preds)
        print(score)
        mcc = matthews_corrcoef(y_val, preds)
        roc_auc = roc_auc_score(y_val, preds)
        balanced_accuracy = balanced_accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds)
        precision = precision_score(y_val, preds)
        recall = recall_score(y_val, preds)
        
        response_body = {
            "Algorithm": "XGBoost",
            "Accuracy": score,
            "balanced_accuracy": balanced_accuracy,
            "mcc": mcc,
            "roc_auc": roc_auc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "time": time.time() - start,
        }
        
        response = requests.post(DATA_CENTER_URL, json=response_body)
        
        now = datetime.now()
        
        created_time = datetime.now().strftime("%Y-%b-%d-%H:%M:%S")
        
        model_path = Path(args.model_path) / created_time
        
        model_path.mkdir(parents=True, exist_ok=True)
        
        saved_model_path = model_path / "model.joblib"
        
        joblib.dump(clf, saved_model_path)
        
        s3_command = [
            "aws", "s3", "cp",
            str(model_path.parent),
            f"s3://{BUCKET_NAME}/{PROJECT_NAME}/models/{ALGORITHM}/{created_time}/",
            "--recursive"
        ]
        
        # Execute the S3 upload command
        subprocess.run(s3_command, check=True)

        test_data = pd.read_csv(
            "/home/xuananle/Documents/AutoML-Platform/src/mlservice/app/test/otto_test_transformed.csv")

        num_rows = test_data.shape[0]


        probabilities = model.predict_proba(test_data)


        # Create a submission DataFrame
        submission = pd.DataFrame(probabilities,
                                  columns=[f'Class_{i + 1}' for i in range(probabilities.shape[1])])

        submission['id'] = test_data['id']
        submission = submission[['id'] + [f'Class_{i + 1}' for i in range(probabilities.shape[1])]]

        # Save the submission file

        submission.to_csv('my_kaggle_submission.csv', index=False)

    except Exception as e:
        print(e)
        response = requests.post(DATA_CENTER_URL, json={"Error": str(e)})
        sys.exit(1)
