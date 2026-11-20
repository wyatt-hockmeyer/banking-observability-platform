# dashboard.py: Streamlit dashboard for the Banking Operations Observability Platform
# Start locally with: streamlit run dashboard.py  (the API must be running first)

import os

import datetime

import requests

import pandas as pd

import streamlit as st

# API address: localhost on the Mac, the api service name inside Docker Compose
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Banking Operations Observability", layout="wide")


# GET request with a readable error instead of a crash
def api_get(path, params=None):
    try:
        response = requests.get(API_URL + path, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request to {path} failed: {e}")
        return None


# POST request with a longer timeout for the AI assistant
def api_post(path, body, timeout=120):
    try:
        response = requests.post(API_URL + path, json=body, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request to {path} failed: {e}")
        return None


st.title("Banking Operations Observability and Predictive Analytics")
st.caption("Replay of 7 days of minute-level digital banking data (NEFT, IMPS, RTGS). Pick a minute to see live KPIs, forecasts, classified logs, and AI root cause analysis.")

# Data range and incident list from the API
data_range = api_get("/range")
if data_range is None:
    st.stop()

start = pd.to_datetime(data_range["start"]).to_pydatetime()

end = pd.to_datetime(data_range["end"]).to_pydatetime()

# ---------- Sidebar: choose the "current" minute ----------
st.sidebar.header("Replay Controls")

mode = st.sidebar.radio("Choose a minute", ["Time slider", "Jump to an incident"])

if mode == "Time slider":
    # Start one hour in, so forecasts have a full hour of history
    selected = st.sidebar.slider(
        "Current minute",
        min_value=start + datetime.timedelta(hours=1),
        max_value=end,
        value=datetime.datetime(2025, 10, 20, 11, 0),
        step=datetime.timedelta(minutes=1),
        format="MM/DD HH:mm"
    )
else:
    choice = st.sidebar.selectbox(f"Incident minute ({len(data_range['incident_times'])} total)", data_range["incident_times"])
    selected = pd.to_datetime(choice).to_pydatetime()

window = st.sidebar.selectbox("Chart and log window", [60, 180, 360, 720], index=1, format_func=lambda m: f"{m // 60} hour(s)")

ts = selected.strftime("%Y-%m-%d %H:%M")

st.sidebar.markdown(f"**Selected:** {ts}")

# ---------- Current KPIs ----------
kpi = api_get("/kpis", {"timestamp": ts})
if kpi is None:
    st.stop()

severity = kpi["severity"]
if severity == "Critical":
    st.error(f"Severity: {severity}. {kpi['findings']}")
elif severity == "High":
    st.warning(f"Severity: {severity}. {kpi['findings']}")
elif severity.startswith("Elevated"):
    st.info(f"Severity: {severity}. {kpi['findings']}")
else:
    st.success(f"Severity: {severity}. {kpi['findings']}")


# Show a KPI with its difference from the model forecast, when one exists
def kpi_card(column, label, field, unit, inverse=False):
    value = kpi[field]
    forecast = kpi.get(field + "_forecast")
    if forecast is None:
        column.metric(label, f"{value:.2f}{unit}")
    else:
        column.metric(label, f"{value:.2f}{unit}", f"{value - forecast:+.2f} vs forecast", delta_color="inverse" if inverse else "normal")


st.subheader(f"Current KPIs at {kpi['timestamp']}")

row1 = st.columns(4)
kpi_card(row1[0], "Auth success rate", "auth_success_rate", "%")
kpi_card(row1[1], "Application errors per min", "application_error_rate", "", inverse=True)
kpi_card(row1[2], "Avg processing time", "avg_transaction_processing_time_ms", " ms", inverse=True)
kpi_card(row1[3], "CPU utilization", "cpu_utilization", "%", inverse=True)

row2 = st.columns(4)
kpi_card(row2[0], "NEFT success rate", "neft_success_rate", "%")
kpi_card(row2[1], "IMPS success rate", "imps_success_rate", "%")
kpi_card(row2[2], "RTGS success rate", "rtgs_success_rate", "%")
kpi_card(row2[3], "Memory usage", "memory_usage", "%", inverse=True)

# ---------- KPI trends: actual vs forecast ----------
st.subheader("KPI Trends: Actual vs Forecast")

window_data = api_get("/window", {"end": ts, "minutes": window})
if window_data:
    trend = pd.DataFrame(window_data)
    trend["timestamp"] = pd.to_datetime(trend["timestamp"])
    trend = trend.set_index("timestamp")

    tab1, tab2, tab3, tab4 = st.tabs(["Processing time", "CPU and memory", "Success rates", "Incidents and anomalies"])

    with tab1:
        st.line_chart(trend[["avg_transaction_processing_time_ms", "avg_transaction_processing_time_ms_forecast"]].rename(columns={"avg_transaction_processing_time_ms": "Actual (ms)", "avg_transaction_processing_time_ms_forecast": "Forecast (ms)"}))

    with tab2:
        st.line_chart(trend[["cpu_utilization", "cpu_utilization_forecast", "memory_usage", "memory_usage_forecast"]].rename(columns={"cpu_utilization": "CPU actual", "cpu_utilization_forecast": "CPU forecast", "memory_usage": "Memory actual", "memory_usage_forecast": "Memory forecast"}))

    with tab3:
        st.line_chart(trend[["neft_success_rate", "imps_success_rate", "rtgs_success_rate", "auth_success_rate"]].rename(columns={"neft_success_rate": "NEFT", "imps_success_rate": "IMPS", "rtgs_success_rate": "RTGS", "auth_success_rate": "Auth"}))
        st.caption("Channel success rates vary at random minute to minute, so their forecasts track the average. Auth drops mark incidents.")

    with tab4:
        flags = trend[["incident", "anomaly"]].astype(float).rename(columns={"incident": "Incident (classifier)", "anomaly": "Anomaly (IsolationForest)"})
        st.bar_chart(flags)
        st.write(f"Incident minutes in window: **{int(trend['incident'].sum())}** | Anomaly minutes in window: **{int(trend['anomaly'].sum())}**")

# ---------- Log classification ----------
st.subheader("Log Classification")

log_data = api_get("/logs", {"end": ts, "minutes": window})
if log_data:
    counts = log_data["counts"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Infrastructure issues", counts.get("Infrastructure", 0))
    c2.metric("Business KPI issues", counts.get("Business KPI", 0))
    c3.metric("Normal operations", counts.get("Normal Operations", 0))

    log_table = pd.DataFrame(log_data["logs"])
    if log_table.empty:
        st.info("No logs in this window.")
    else:
        issue_logs = log_table[log_table["issue_category"] != "Normal Operations"]
        # Short labels so every cluster name fits beside its bar
        short_names = {
            "Payment channel failures (NEFT, IMPS, RTGS)": "Payment failures",
            "CPU and memory pressure": "CPU / memory",
            "Container OOM restarts": "OOM restarts",
            "Customer authentication": "Auth failures",
            "Database connection pool": "DB pool",
            "Payment gateway SLA breach": "Gateway SLA",
            "Duplicate transactions": "Duplicates",
            "Network timeouts": "Network timeouts",
            "Disk I/O bottleneck": "Disk I/O",
            "API rate limiting": "Rate limits"
        }
        chart_counts = issue_logs["cluster_name"].map(short_names).fillna(issue_logs["cluster_name"]).value_counts()
        st.bar_chart(chart_counts, horizontal=True)
        st.dataframe(log_table.rename(columns={"timestamp": "Time", "log_level": "Level", "log_message": "Message", "cluster_name": "Cluster", "issue_category": "Category"}), use_container_width=True, hide_index=True)

# Classify a new log message on demand
with st.expander("Classify a new log message"):
    new_log = st.text_input("Log message", "ERROR: Database connection pool exhausted. Connections: 100/100")
    if st.button("Classify"):
        result = api_post("/classify_log", {"message": new_log}, timeout=30)
        if result:
            st.write(f"**Cluster:** {result['cluster_name']} | **Category:** {result['issue_category']}")

# ---------- AI incident analysis ----------
st.subheader("AI Incident Analysis (LangGraph + RAG)")

question = st.text_input("Ask the assistant", f"What is happening at {ts} and what should operations do?")

if st.button("Run AI analysis"):
    with st.spinner("Scoring metrics, retrieving runbooks and past incidents, and writing the analysis..."):
        analysis = api_post("/analyze", {"question": question, "timestamp": ts})
    if analysis:
        st.session_state["analysis"] = analysis
        st.session_state["analysis_ts"] = ts

# Keep the last analysis on screen while it matches the selected minute
if st.session_state.get("analysis_ts") == ts and "analysis" in st.session_state:
    analysis = st.session_state["analysis"]
    st.markdown(f"**Severity:** {analysis['severity']}")
    st.markdown(analysis["answer"])
    st.caption("Retrieved sources: " + ", ".join(analysis["sources"]))
