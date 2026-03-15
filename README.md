# ⚙️ Quant SA Engine: Serverless MLOps Portfolio Optimizer

An institutional-grade, fully automated quantitative trading pipeline built on Google Cloud Platform (GCP). This project demonstrates an end-to-end MLOps architecture, moving from raw data extraction to machine learning inference, mathematical optimization, and real-world trade execution sizing—all orchestrated via serverless containers.

## 🏗️ Architecture Overview

This pipeline is built using **Vertex AI Pipelines (Kubeflow)**. It is designed to be completely stateless, cost-optimized, and self-executing. Every Monday morning, the Directed Acyclic Graph (DAG) spins up isolated Docker containers to execute four distinct phases, passing data strictly via Google Cloud Storage artifacts.

### The 4-Stage DAG:

1. **The Risk Gatekeeper (BigQuery ML)**
   * Executes a highly optimized, cost-controlled SQL query (`$0.0004` per run) against a central data lake.
   * Leverages an **XGBoost Classifier** pre-trained in BigQuery ML to predict the probability of positive forward returns.
   * Filters out high-risk assets and outputs the top 20 "safe" tickers.

2. **The Market Fetcher (Dynamic Time-Series)**
   * Ingests the 20 safe tickers and dynamically calculates a 5-year rolling lookback window.
   * Downloads historical pricing, standardizes trading calendars, and forward-fills gaps.
   * Applies Winsorization (15% daily return clipping) to sanitize the data against extreme market anomalies before mathematical modeling.

3. **The Quant Brain (Genetic Algorithm Optimizer)**
   * Ingests the clean return distributions and applies a **300-population Genetic Algorithm** over 50 generations to find the optimal capital allocation.
   * **Fitness Function:** Maximizes the Sharpe Ratio while accounting for a dynamic risk-free rate and realistic transaction costs (0.75%).
   * **Kill-Switch:** Imposes a strict maximum portfolio volatility cap (e.g., 40%), heavily penalizing any genome that breaches the risk threshold.

4. **The Execution Engine & Notifier**
   * Pulls the live `USDZAR=X` exchange rate to handle cross-currency allocations.
   * Converts percentage weights into exact fractional share/coin quantities based on absolute latest closing prices and total ZAR capital available.
   * Compiles the execution instructions and emails a formatted shopping list via SMTP.

## 🚀 Key Engineering Principles Demonstrated

* **Stateless Component Design:** Functions do not share memory. Data is passed between nodes exclusively via serialized JSON and CSV Artifacts in Cloud Storage.
* **Execution Caching:** Vertex AI automatically caches successful nodes. If a downstream component fails (e.g., SMTP timeout), the pipeline resumes instantly from the point of failure without re-running expensive ML compute.
* **Query Optimization:** BigQuery extraction uses Common Table Expressions (CTEs) and Inner Joins on `MAX(Date)` rather than window functions, dropping data scan volume from ~2.3TB to ~65MB per run.
* **Infrastructure as Code:** The entire architecture is compiled into a deployable JSON blueprint, decoupling the Python logic from the cloud infrastructure.

## 🛠️ Tech Stack
* **Cloud Platform:** Google Cloud (Vertex AI, BigQuery, Cloud Storage)
* **Orchestration:** Kubeflow Pipelines (`kfp`)
* **Machine Learning:** BigQuery ML (XGBoost)
* **Optimization:** Custom Genetic Algorithm (NumPy, Pandas)
* **Market Data:** `yfinance`

## 🔒 Security & Local Setup
This repository contains no hardcoded credentials. To run this pipeline in your own GCP environment:

1. Clone the repository.
2. Create a local `.env` file (this is strictly ignored by `.gitignore`):
   ```text
   GCP_PROJECT_ID=your-project-id
   GCP_REGION=your-region
   GCP_BUCKET=gs://your-artifact-bucket
   SENDER_EMAIL=your_email@gmail.com
   EMAIL_PASSWORD=your_app_password
   RECIPIENT_EMAIL=your_email@gmail.com
