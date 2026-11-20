# agent.py: LangGraph incident analysis assistant
# Flow: analyze metrics -> retrieve runbooks and incidents -> generate RCA (or fallback)

import os

import joblib

import openai

import pandas as pd

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv

from langchain_chroma import Chroma

from langchain_openai import OpenAIEmbeddings

# Load OPENAI_API_KEY from .env
load_dotenv()

# Paths relative to this file so it works from the notebook, API, dashboard, or Docker
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")

CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# Trained models from Task 2
classifier = joblib.load(os.path.join(MODEL_DIR, "incident_classifier.joblib"))

anomaly_model = joblib.load(os.path.join(MODEL_DIR, "anomaly_detector.joblib"))

config = joblib.load(os.path.join(MODEL_DIR, "model_config.joblib"))

# OpenAI client and retrievers are created on first use, so a missing key fails gracefully
client = None
runbook_retriever = None
incident_retriever = None


def get_client():
    global client
    if client is None and os.getenv("OPENAI_API_KEY"):
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client


# Separate retrievers so runbooks are never crowded out by incident records
def get_retrievers():
    global runbook_retriever, incident_retriever
    if runbook_retriever is None:
        vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=OpenAIEmbeddings())
        runbook_retriever = vectordb.as_retriever(search_kwargs={"k": 2, "filter": {"doc_type": "runbook"}})
        incident_retriever = vectordb.as_retriever(search_kwargs={"k": 2, "filter": {"doc_type": "incident"}})
    return runbook_retriever, incident_retriever


# Shared state passed between nodes
class State(TypedDict):
    question: str
    metrics: dict
    findings: str
    severity: str
    context: str
    sources: list
    answer: str


# Node 1: score the metrics snapshot with the incident classifier and anomaly detector
def analyze_metrics(state: State):
    metrics = state.get("metrics") or {}
    if not metrics:
        return {"findings": "No metrics snapshot provided.", "severity": "Unknown"}

    try:
        row = pd.DataFrame([metrics])
        incident = int(classifier.predict(row[config["classifier_features"]])[0])
        anomaly = int(anomaly_model.predict(row[config["anomaly_features"]])[0] == -1)
    except KeyError as e:
        return {"findings": f"Metrics snapshot is missing fields: {e}", "severity": "Unknown"}

    # Severity rule: classifier first, then anomaly detector, then the known peak-load pattern
    hour = int(metrics.get("hour", 0))
    latency = float(metrics["avg_transaction_processing_time_ms"])
    if incident == 1:
        severity = "Critical"
    elif anomaly == 1:
        severity = "High"
    elif 9 <= hour <= 17 and latency > 300:
        severity = "Elevated (expected peak load)"
    else:
        severity = "Normal"

    findings = (
        f"Incident classifier: {'INCIDENT' if incident == 1 else 'normal'}. "
        f"IsolationForest: {'ANOMALY' if anomaly == 1 else 'normal'}. "
        f"Hour {hour}. Auth success {metrics['auth_success_rate']}%. "
        f"Application errors {metrics['application_error_rate']}/min. "
        f"CPU {metrics['cpu_utilization']}%. Memory {metrics['memory_usage']}%. "
        f"DB pool {metrics['db_connection_pool_usage']}%. Processing time {latency} ms. "
        f"NEFT {metrics['neft_success_rate']}%, IMPS {metrics['imps_success_rate']}%, RTGS {metrics['rtgs_success_rate']}%. "
        f"Service uptime {metrics['service_uptime']}%."
    )
    return {"findings": findings, "severity": severity}


# Node 2: retrieve 2 runbooks and 2 historical incidents from Chroma
def retrieve(state: State):
    query = state["question"] + "\n" + state.get("findings", "")
    try:
        rb_retriever, inc_retriever = get_retrievers()
        docs = rb_retriever.invoke(query) + inc_retriever.invoke(query)
    except Exception as e:
        return {"context": "", "sources": [], "answer": f"Knowledge base unavailable: {e}"}

    context = "\n\n".join([d.page_content for d in docs])
    sources = [os.path.basename(d.metadata.get("source", "unknown")) for d in docs]
    return {"context": context, "sources": sources}


# Conditional edge: generate only when retrieval returned context
def route_after_retrieve(state: State):
    if state.get("context"):
        return "generate"
    return "fallback"


# Node 3: write the root cause summary and recommendations with gpt-5-mini
def generate(state: State):
    llm = get_client()
    if llm is None:
        return {"answer": "OpenAI API key not found. Add OPENAI_API_KEY to the .env file.\n\nModel findings: " + state.get("findings", "")}

    system_msg = (
        "You are a banking operations reliability assistant for a digital bank (NEFT, IMPS, RTGS). "
        "Use only the model findings and the retrieved runbooks and historical incidents. "
        "Do not add tools, systems, or steps that are not in the retrieved runbooks. "
        "Respond in under 250 words with four sections: "
        "Root Cause Summary (2 to 3 sentences), Evidence (bullets), "
        "Recommendations (numbered, taken from the runbooks), Sources (runbook names and incident IDs). "
        "If severity is Normal or Elevated (expected peak load), say no action is needed beyond monitoring and explain why. "
        "If the context does not support a conclusion, say so. "
        "Do not use em dashes; use commas, colons, or periods instead."
    )
    user_msg = (
        f"Question: {state['question']}\n\n"
        f"Severity from models: {state.get('severity', 'Unknown')}\n"
        f"Model findings: {state.get('findings', '')}\n\n"
        f"Retrieved context:\n{state['context']}"
    )

    try:
        response = llm.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
        )
        return {"answer": response.choices[0].message.content}
    except openai.RateLimitError:
        return {"answer": "OpenAI rate limit reached. Wait a minute and try again.\n\nModel findings: " + state.get("findings", "")}
    except Exception as e:
        return {"answer": f"AI analysis failed: {e}\n\nModel findings: " + state.get("findings", "")}


# Node 4: fallback when nothing was retrieved
def fallback(state: State):
    reason = state.get("answer") or "No matching runbooks or incidents were found."
    return {"answer": reason + "\n\nModel findings: " + state.get("findings", "") + "\nSeverity: " + state.get("severity", "Unknown")}


# Build the graph: START -> analyze -> retrieve -> (generate | fallback) -> END
def build_workflow():
    graph = StateGraph(State)
    graph.add_node("analyze_metrics", analyze_metrics)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("fallback", fallback)
    graph.add_edge(START, "analyze_metrics")
    graph.add_edge("analyze_metrics", "retrieve")
    graph.add_conditional_edges("retrieve", route_after_retrieve, {"generate": "generate", "fallback": "fallback"})
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)
    return graph.compile()


workflow = build_workflow()


# Single entry point used by the notebook, FastAPI, and Streamlit
def run_assistant(question, metrics=None):
    return workflow.invoke({"question": question, "metrics": metrics or {}})
