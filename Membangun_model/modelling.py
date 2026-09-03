from pathlib import Path
import json
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

BASE = Path(__file__).resolve().parent
DATA = BASE / "breast_cancer_preprocessing"
train, test = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
X_train, y_train = train.drop(columns="diagnosis"), train["diagnosis"]
X_test, y_test = test.drop(columns="diagnosis"), test["diagnosis"]
mlflow.set_tracking_uri((BASE / "mlruns").as_uri())
mlflow.set_experiment("Breast Cancer Classification - Reza Harahap")
mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)
with mlflow.start_run(run_name="random_forest_baseline"):
    model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print({"accuracy":accuracy_score(y_test,pred), "precision":precision_score(y_test,pred),
           "recall":recall_score(y_test,pred), "f1":f1_score(y_test,pred),
           "roc_auc":roc_auc_score(y_test,proba)})
