import json, time, requests
from pathlib import Path
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram, start_http_server
REQUESTS=Counter("model_requests_total","Total inference requests")
ERRORS=Counter("model_errors_total","Total inference errors")
LATENCY=Histogram("model_request_latency_seconds","Inference latency")
POSITIVE=Counter("model_positive_predictions_total","Benign predictions")
CONFIDENCE=Gauge("model_prediction_confidence","Latest confidence")
MODEL_UP=Gauge("model_up","Model service availability")
DATA=Path(__file__).resolve().parents[1]/"Membangun_model"/"breast_cancer_preprocessing"/"test.csv"
def predict(row):
    started=time.perf_counter(); REQUESTS.inc()
    try:
        payload={"dataframe_split":{"columns":list(row.columns),"data":row.values.tolist()}}
        response=requests.post("http://127.0.0.1:5001/invocations",json=payload,timeout=10)
        response.raise_for_status(); result=response.json(); value=int(result["predictions"][0])
        if value==1: POSITIVE.inc()
        CONFIDENCE.set(1.0); MODEL_UP.set(1); return result
    except Exception: ERRORS.inc(); MODEL_UP.set(0); raise
    finally: LATENCY.observe(time.perf_counter()-started)
if __name__=="__main__":
    start_http_server(8000)
    test=pd.read_csv(DATA).drop(columns="diagnosis")
    for i in range(30):
        print(predict(test.iloc[[i % len(test)]])); time.sleep(.5)
    print("Exporter tetap aktif; tekan Ctrl+C untuk berhenti.")
    while True: time.sleep(1)
