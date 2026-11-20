# Banking Operations Observability and Predictive Analytics Platform

Unit 9 Final Capstone, Fullstack Academy AI and Machine Learning Program
Author: Dr. Wyatt Hockmeyer

An end-to-end observability platform for a digital bank (NEFT, IMPS, RTGS). It unifies business KPIs, infrastructure metrics, and application logs, forecasts key metrics, flags incidents and anomalies, classifies logs by issue type, and generates AI root cause analysis through a LangGraph assistant grounded in runbooks and historical incidents. Everything is delivered through a Streamlit dashboard backed by a FastAPI service, and both run in Docker.

## Business Problem

Traditional monitoring keeps business metrics, logs, and infrastructure data in separate tools, which slows root cause identification during service disruptions. This platform moves operations from reactive monitoring to predictive intelligence: it forecasts load and latency, detects incidents as they happen, and tells operators what is most likely wrong and what to do next.

## Architecture

```
Datasets (transactions, infrastructure metrics, application logs)
        |
Task 1  Data preparation and EDA (notebook)
        |
Task 2  Forecasting, incident classifier, IsolationForest (notebook, saved with joblib)
        |
Task 3  TF-IDF + KMeans log clustering, knowledge base, Chroma RAG, LangGraph assistant
        |
Task 4  FastAPI (api.py)  <---  Streamlit dashboard (dashboard.py)
        |                              |
        +------- Docker Compose -------+
```

LangGraph assistant flow (agent.py):

```
START -> analyze_metrics -> retrieve -> generate -> END
                                    \-> fallback -> END   (if retrieval fails)
```

1. **analyze_metrics:** the incident classifier and IsolationForest score the selected minute and set a severity (Critical, High, Elevated, Normal).
2. **retrieve:** Chroma returns the 2 most relevant runbooks and the 2 most similar historical incidents.
3. **generate:** gpt-5-mini writes a Root Cause Summary, Evidence, Recommendations taken from the runbooks, and Sources.
4. **fallback:** returns the model findings with a clear message if the knowledge base or API key is unavailable.

## Project Structure

```
capstone-banking-observability/
  Hockmeyer Unit 9 Capstone Task 1 EDA.ipynb              Sections 1 to 7
  Hockmeyer Unit 9 Capstone Task 2 Models.ipynb           Sections 8 to 12
  Hockmeyer Unit 9 Capstone Task 3 Log Intelligence.ipynb Sections 13 to 17
  agent.py                 LangGraph incident analysis assistant
  api.py                   FastAPI backend (8 endpoints)
  dashboard.py             Streamlit dashboard
  Dockerfile.api           API container
  Dockerfile.dashboard     Dashboard container
  docker-compose.yml       Runs both containers together
  requirements.txt         Python packages (Python 3.12)
  models/                  Saved models (joblib)
  kb/runbooks/             12 runbooks, one per issue type
  kb/incidents/            163 historical incident records plus 1 load-anomaly summary
  chroma_db/               Chroma vector database (built in Section 16)
  data/                    Datasets and processed files (not in the repository)
```

## Setup (Mac, Python 3.12)

1. Clone the repository and open Terminal in the project folder.
2. Create and activate a virtual environment:
   ```
   python3.12 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Create a file named `.env` in the project folder containing:
   ```
   OPENAI_API_KEY=your-key-here
   ```
4. Place the three course datasets in `data/`: `transactions_fraud_train.csv`, `infrastructure_metrics.csv`, `application_logs.csv`.
5. Run the three notebooks in order (Task 1, Task 2, Task 3), selecting the venv kernel. They create the processed data, saved models, knowledge base, and Chroma database.

## Run Locally

Terminal tab 1, the API:
```
python -m uvicorn api:app --reload
```
API docs: http://127.0.0.1:8000/docs

Terminal tab 2, the dashboard:
```
streamlit run dashboard.py
```
Dashboard: http://localhost:8501

## Run with Docker

With Docker Desktop running:
```
docker compose up --build
```
Dashboard: http://localhost:8501 | API: http://localhost:8000/docs

Stop with:
```
docker compose down
```

The API container mounts `models/`, `data/`, and `chroma_db/` from the project folder and reads the OpenAI key from `.env`. Neither the key nor the data is copied into an image.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /health | Service status and rows loaded |
| GET | /range | Data start and end, plus all incident minutes |
| GET | /kpis | KPIs, forecasts, and model severity for one minute |
| GET | /window | Actual vs forecast history for a time window |
| GET | /logs | Classified logs and category counts for a time window |
| POST | /score | Score any metrics snapshot (incident and anomaly) |
| POST | /classify_log | Classify a new log message into a cluster and issue category |
| POST | /analyze | Run the LangGraph assistant for a minute |

## Results

**Data (Task 1)**
- 10,080 minutes of infrastructure and KPI data (7 days, October 20 to 26, 2025), 4,991 logs after removing 9 duplicates, and 4,000 transactions. No missing values.
- Peak load runs from 09:00 to 17:59 every day. Weekday CPU near 70 percent and processing time near 400 ms; weekend near 50 percent and 200 ms; off-peak floor of 100 ms.
- Processing time correlates 0.99 with CPU, 0.97 with memory, and 0.89 with transaction volume. Channel success rates show no relationship to load.
- 175 incident minutes in which authentication success fell below 98 percent at the same time application errors reached 6 or more per minute. These occur at all hours, independent of load.

**Forecasting (Task 2, next-minute forecast, time-ordered test set)**

| Target | Best model | RMSE | MAE | Naive RMSE |
|---|---|---|---|---|
| Processing time (ms) | Random Forest | 25.26 | 20.31 | 33.54 |
| CPU utilization (%) | Random Forest | 3.12 | 2.64 | 4.15 |
| Memory usage (%) | Random Forest | 1.94 | 1.60 | 2.66 |
| NEFT success rate (%) | Linear Regression | 1.16 | 1.00 | 1.63 |
| IMPS success rate (%) | Linear Regression | 0.58 | 0.50 | 0.81 |
| RTGS success rate (%) | Linear Regression | 0.29 | 0.25 | 0.40 |

All models beat the naive baseline by about 20 to 30 percent. Channel success rates vary at random minute to minute, so their forecasts track the average.

**Incident classifier:** Decision Tree with SMOTE. Accuracy 1.00, incident F1 1.00 on a stratified test set (35 of 35 incidents, 0 false alarms).

**IsolationForest (unsupervised):** found 121 of 175 incidents (recall 0.69) without labels. The other 55 flags were extreme peak-load minutes (average 72 percent CPU and 421 ms).

**Log clustering:** TF-IDF plus KMeans (k = 12) grouped 17 message templates into clear topics, mapped to Infrastructure (2,016 logs), Business KPI (1,736), and Normal Operations (1,239).

**RAG and assistant:** 176 documents (12 runbooks, 164 incident files) in Chroma with OpenAI embeddings. Answers cite runbooks and incident IDs, and recommendations are limited to runbook steps.

## Course Tier Map and Documented Exceptions

| Component | Course reference |
|---|---|
| Data preparation and EDA | Tier 2: ds_lesson_reference_part1 and part2 |
| Regression, classification, SMOTE, IsolationForest, KMeans | Tier 3: ml_lesson_reference_part1 and part2 |
| TF-IDF and NLTK preprocessing | Tier 6: nlp_lesson_reference_part2_notebooks |
| Chroma RAG with OpenAI embeddings | Tier 7: genai_lesson_reference_part2_langchain_rag (Lesson 56.03) |
| LangGraph assistant and Streamlit | Tier 8: agentic_lesson_reference_part2_demos |
| FastAPI, joblib, Docker | Tier 5: mlops_lesson_reference_part2_demos |

Documented exceptions:
- **Time-series forecasting:** the lessons cover resampling and rolling averages, so forecasting uses lag and rolling-average features with regression models evaluated by RMSE and MAE.
- **Docker Compose:** not in the lessons; built from docs.docker.com.
- **Incident label:** the dataset has no incident label, so one was defined by rule (authentication success below 98 percent with an application error spike). The classifier automates triage of that definition, which explains its perfect scores.
- **Versions:** LangGraph is pinned to 0.3.0 as in the demos. Streamlit uses the current release because 1.30.0 requires numpy below version 2.
- **Runbooks:** the dataset includes no documentation, so the 12 runbooks were written for this capstone.

## Limitations

- The data is synthetic. Transaction-level processing times show no relationship to infrastructure load, so the infrastructure file carries the forecasting and correlation work.
- Forecasts are one minute ahead. Longer horizons would need more history than 7 days.
- The dashboard replays historical data rather than streaming live data.
