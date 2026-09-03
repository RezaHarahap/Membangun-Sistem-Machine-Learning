from pathlib import Path
import json
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import GridSearchCV

BASE = Path(__file__).resolve().parent
DATA = BASE / "breast_cancer_preprocessing"
ART = BASE / "additional_artifacts"; ART.mkdir(exist_ok=True)
train, test = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
X_train, y_train = train.drop(columns="diagnosis"), train["diagnosis"]
X_test, y_test = test.drop(columns="diagnosis"), test["diagnosis"]
search = GridSearchCV(RandomForestClassifier(random_state=42),
    {"n_estimators":[100,150], "max_depth":[None,8], "min_samples_split":[2,5]},
    scoring="f1", cv=3, n_jobs=-1)
search.fit(X_train, y_train)
model = search.best_estimator_
pred, proba = model.predict(X_test), model.predict_proba(X_test)[:,1]
metrics = {"accuracy":accuracy_score(y_test,pred),
    "precision":precision_score(y_test,pred), "recall":recall_score(y_test,pred),
    "f1":f1_score(y_test,pred), "roc_auc":roc_auc_score(y_test,proba),
    "cv_best_f1":search.best_score_}
sns.heatmap(confusion_matrix(y_test,pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix"); plt.savefig(ART/"confusion_matrix.png", bbox_inches="tight"); plt.close()
RocCurveDisplay.from_predictions(y_test,proba); plt.savefig(ART/"roc_curve.png", bbox_inches="tight"); plt.close()
(ART/"metrics_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
mlflow.set_tracking_uri((BASE / "mlruns").as_uri())
mlflow.set_experiment("Breast Cancer Classification - Reza Harahap")
with mlflow.start_run(run_name="random_forest_tuned_manual_logging"):
    mlflow.log_params(search.best_params_); mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model", input_example=X_test.head(3), registered_model_name=None)
    mlflow.log_artifacts(str(ART), artifact_path="evaluation")
    mlflow.log_dict({"feature_names":list(X_train.columns)}, "feature_schema.json")
    print("Best parameters:", search.best_params_); print("Metrics:", metrics)
