# api.py: FastAPI backend for model inference and log insights
# Start locally with: python -m uvicorn api:app --reload

import os

import re

import string

import joblib

import nltk

import pandas as pd

from fastapi import FastAPI, HTTPException

from pydantic import BaseModel

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

# The LangGraph assistant (also loads the incident classifier and anomaly detector)
import agent

# Paths relative to this file so it works locally and in Docker
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")

DATA_DIR = os.path.join(BASE_DIR, "data")

# NLTK resources for log cleaning (quiet if already downloaded)
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(resource, quiet=True)

# Saved models and feature lists from Tasks 2 and 3
config = joblib.load(os.path.join(MODEL_DIR, "model_config.joblib"))

forecasters = {}
for target in config["forecast_targets"]:
    forecasters[target] = joblib.load(os.path.join(MODEL_DIR, "forecast_" + target + ".joblib"))

log_vectorizer = joblib.load(os.path.join(MODEL_DIR, "log_vectorizer.joblib"))

log_kmeans = joblib.load(os.path.join(MODEL_DIR, "log_kmeans.joblib"))

log_map = joblib.load(os.path.join(MODEL_DIR, "log_cluster_map.joblib"))

# Scored metrics (with lag features, incident, and anomaly columns) and clustered logs
infra = pd.read_csv(os.path.join(DATA_DIR, "scored_infrastructure.csv"), parse_dates=["timestamp"])

infra = infra.sort_values("timestamp").reset_index(drop=True)

logs = pd.read_csv(os.path.join(DATA_DIR, "clustered_logs.csv"), parse_dates=["timestamp"])

# Forecast every minute that has a full hour of history, once at startup
ready = infra.dropna(subset=config["forecast_features"]).index

for target in config["forecast_targets"]:
    infra.loc[ready, target + "_forecast"] = forecasters[target].predict(infra.loc[ready, config["forecast_features"]])

# Columns the dashboard needs for KPI cards and charts
kpi_cols = ["cpu_utilization", "memory_usage", "db_connection_pool_usage", "network_throughput_mbps",
            "application_error_rate", "neft_success_rate", "imps_success_rate", "rtgs_success_rate",
            "avg_transaction_processing_time_ms", "auth_success_rate", "transaction_volume_per_hour",
            "service_uptime", "hour", "day_type", "incident", "anomaly"]

forecast_cols = []
for target in config["forecast_targets"]:
    forecast_cols.append(target + "_forecast")

app = FastAPI(title="Banking Operations Observability API")


# Request bodies
class ScoreRequest(BaseModel):
    metrics: dict


class LogRequest(BaseModel):
    message: str


class AnalyzeRequest(BaseModel):
    question: str
    timestamp: str


# Convert a DataFrame to JSON-safe records (NaN becomes null)
def to_records(df):
    return df.astype(object).where(df.notnull(), None).to_dict(orient="records")


# Latest minute at or before the requested timestamp
def find_minute(timestamp):
    try:
        ts = pd.to_datetime(timestamp)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Timestamp format not recognized. Use YYYY-MM-DD HH:MM.")
    rows = infra[infra["timestamp"] <= ts]
    if rows.empty:
        raise HTTPException(status_code=404, detail="No data at or before that timestamp.")
    return rows.iloc[-1]


# Same cleaning steps as Section 13
stop_words = set(stopwords.words("english")) - {"not", "no"}

lemmatizer = WordNetLemmatizer()


def clean_log(text):
    text = re.sub(r"^(INFO|WARNING|ERROR):\s*", "", text)
    text = text.lower()
    text = re.sub(r"[a-z_]*\d+[a-z0-9_]*", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)


# Service status
@app.get("/health")
def health():
    return {"status": "ok", "minutes_loaded": len(infra), "logs_loaded": len(logs)}


# Data range and incident minutes for the dashboard time slider
@app.get("/range")
def data_range():
    incident_times = infra.loc[infra["incident"] == 1, "timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist()
    return {
        "start": str(infra["timestamp"].min()),
        "end": str(infra["timestamp"].max()),
        "incident_times": incident_times
    }


# Current KPIs, forecasts, and model severity for one minute
@app.get("/kpis")
def kpis(timestamp: str):
    row = find_minute(timestamp)
    metrics = row.drop("timestamp").to_dict()
    scored = agent.analyze_metrics({"metrics": metrics})
    record = to_records(pd.DataFrame([row[["timestamp"] + kpi_cols + forecast_cols]]))[0]
    record["timestamp"] = str(row["timestamp"])
    record["severity"] = scored["severity"]
    record["findings"] = scored["findings"]
    return record


# Actual vs forecast history ending at a minute
@app.get("/window")
def window(end: str, minutes: int = 180):
    row = find_minute(end)
    start = row["timestamp"] - pd.Timedelta(minutes=minutes)
    part = infra[(infra["timestamp"] > start) & (infra["timestamp"] <= row["timestamp"])]
    part = part[["timestamp"] + kpi_cols + forecast_cols].copy()
    part["timestamp"] = part["timestamp"].astype(str)
    return to_records(part)


# Classified logs in the window ending at a minute
@app.get("/logs")
def log_window(end: str, minutes: int = 60):
    row = find_minute(end)
    start = row["timestamp"] - pd.Timedelta(minutes=minutes)
    part = logs[(logs["timestamp"] > start) & (logs["timestamp"] <= row["timestamp"])]
    part = part[["timestamp", "log_level", "log_message", "cluster_name", "issue_category"]].copy()
    part["timestamp"] = part["timestamp"].astype(str)
    counts = part["issue_category"].value_counts().to_dict()
    return {"counts": counts, "logs": to_records(part.sort_values("timestamp", ascending=False))}


# Score any metrics snapshot with the incident classifier and anomaly detector
@app.post("/score")
def score(request: ScoreRequest):
    result = agent.analyze_metrics({"metrics": request.metrics})
    return {"severity": result["severity"], "findings": result["findings"]}


# Classify a new log message into a cluster and issue category
@app.post("/classify_log")
def classify_log(request: LogRequest):
    cleaned = clean_log(request.message)
    if cleaned == "":
        raise HTTPException(status_code=400, detail="Log message has no usable words after cleaning.")
    cluster = int(log_kmeans.predict(log_vectorizer.transform([cleaned]))[0])
    category = log_map["cluster_category"][cluster]
    if request.message.strip().upper().startswith("INFO"):
        category = "Normal Operations"
    return {"cluster": cluster, "cluster_name": log_map["cluster_name"][cluster], "issue_category": category, "cleaned": cleaned}


# Run the LangGraph assistant on the metrics at a minute
@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    row = find_minute(request.timestamp)
    metrics = row.drop("timestamp").to_dict()
    result = agent.run_assistant(request.question, metrics)
    return {
        "timestamp": str(row["timestamp"]),
        "severity": result.get("severity", "Unknown"),
        "findings": result.get("findings", ""),
        "sources": result.get("sources", []),
        "answer": result.get("answer", "")
    }
