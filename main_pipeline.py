import os
from dotenv import load_dotenv
from kfp import dsl, compiler
from google.cloud import aiplatform

# ==========================================
# 0. LOAD SECRETS FROM .env
# ==========================================
load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION")
BUCKET_NAME = os.getenv("GCP_BUCKET")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")

# ==========================================
# COMPONENT 1: THE DATA ENGINEER (PyArrow Added)
# ==========================================
@dsl.component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-bigquery", "pandas", "db-dtypes", "pyarrow"]
)
def extract_safe_assets(
    project_id: str,
    risk_scores_output: dsl.Output[dsl.Artifact]
) -> list:
    from google.cloud import bigquery
    import json
    import pandas as pd
    
    bq_client = bigquery.Client(project=project_id)
    
    predict_query = f"""
    WITH LatestDates AS (
      SELECT Ticker, MAX(Date) as max_date
      FROM `{project_id}.quant_sa.raw_features`
      WHERE Ticker NOT LIKE '^%'
      GROUP BY Ticker
    )
    SELECT 
        r.Ticker, 
        r.Date,
        r.Close,
        ROUND(r.predicted_target_y_probs[OFFSET(0)].prob, 4) AS risk_score
    FROM ML.PREDICT(MODEL `{project_id}.quant_sa.xgboost_risk_filter_final`,
      (
        SELECT * FROM `{project_id}.quant_sa.raw_features`
        QUALIFY ROW_NUMBER() OVER(PARTITION BY Ticker ORDER BY Date DESC) = 1
      )
    ) r
    INNER JOIN LatestDates l 
      ON r.Ticker = l.Ticker AND r.Date = l.max_date
    ORDER BY risk_score ASC 
    LIMIT 20;
    """
    
    predictions_df = bq_client.query(predict_query).to_dataframe()
    
    if predictions_df.empty:
        print("Warning: No assets found.")
        with open(risk_scores_output.path, "w") as f:
            json.dump({}, f)
        return []

    predictions_df = predictions_df.drop_duplicates(subset=['Ticker'])
    
    scores_dict = predictions_df.set_index('Ticker')[['risk_score', 'Date', 'Close']].to_dict('index')
    with open(risk_scores_output.path, "w") as f:
        json.dump(scores_dict, f)
        
    return predictions_df['Ticker'].tolist()

# ==========================================
# COMPONENT 2: THE MARKET FETCHER
# ==========================================
@dsl.component(
    base_image="python:3.10",
    packages_to_install=["yfinance", "pandas", "numpy"]
)
def fetch_market_data(
    safe_tickers: list, 
    clean_data_output: dsl.Output[dsl.Dataset]
):
    import yfinance as yf
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    if not safe_tickers:
        pd.DataFrame().to_csv(clean_data_output.path)
        return

    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    raw_data = yf.download(safe_tickers, start=start_date, end=end_date, progress=False)['Close']
    
    raw_data.index = pd.to_datetime(raw_data.index).tz_localize(None)
    full_calendar = pd.date_range(start=raw_data.index.min(), end=raw_data.index.max(), freq='D')
    df = raw_data.reindex(full_calendar).ffill().dropna(how='all')
    
    daily_returns = df.pct_change().fillna(0).clip(lower=-0.15, upper=0.15)
    daily_returns.to_csv(clean_data_output.path)

# ==========================================
# COMPONENT 3: THE GA OPTIMIZER
# ==========================================
@dsl.component(
    base_image="python:3.10",
    packages_to_install=["pandas", "numpy", "yfinance"] 
)
def run_genetic_optimizer(
    clean_data_input: dsl.Input[dsl.Dataset],
    risk_scores_input: dsl.Input[dsl.Artifact],
    capital_zar: float,
    max_volatility: float,
    transaction_cost: float,
    risk_free_rate: float,
    results_output: dsl.Output[dsl.Artifact]
):
    import pandas as pd
    import numpy as np
    import random
    import json
    import yfinance as yf

    daily_returns = pd.read_csv(clean_data_input.path, index_col=0)
    if daily_returns.empty:
        with open(results_output.path, "w") as f:
            json.dump({"shopping_list": []}, f)
        return

    annual_returns = daily_returns.mean() * 252
    cov_matrix = daily_returns.cov() * 252
    num_assets = len(daily_returns.columns)
    safe_tickers = daily_returns.columns.tolist()
    
    with open(risk_scores_input.path, "r") as f:
        scores_data = json.load(f)

    def calculate_fitness(weights):
        net_return = np.sum(annual_returns * weights) - (np.sum(weights) * transaction_cost)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        if port_volatility > max_volatility: return -100 
        return (net_return - risk_free_rate) / port_volatility

    population = [np.random.random(num_assets) / np.sum(np.random.random(num_assets)) for _ in range(200)]
    for _ in range(40):
        population = sorted(population, key=lambda ind: calculate_fitness(ind), reverse=True)
        next_gen = population[:50]
        while len(next_gen) < 200:
            p1, p2 = random.sample(population[:80], 2)
            child = np.concatenate((p1[:num_assets//2], p2[num_assets//2:]))
            child = child / np.sum(child)
            next_gen.append(child)
        population = next_gen

    best_weights = population[0]
    usdzar_data = yf.download("USDZAR=X", period="5d", progress=False)
    exchange_rate = float(usdzar_data['Close'].iloc[-1].squeeze())

    final_results = []
    for i, asset in enumerate(safe_tickers):
        weight = float(best_weights[i])
        if weight < 0.01: continue
        
        zar_allocated = weight * capital_zar
        price = scores_data[asset]['Close']
        
        if "USD" in asset:
            units = (zar_allocated / exchange_rate) / price
            action = f"{units:.6f} coins"
        else:
            units = int(np.floor(zar_allocated / (price/100)))
            action = f"{units} shares"

        final_results.append({
            "ticker": asset, "weight": weight, "zar_amount": zar_allocated,
            "action": action, "risk_score": scores_data[asset]['risk_score'],
            "Close": price, 
            "Date": scores_data[asset]['Date']
        })

    with open(results_output.path, "w") as f:
        json.dump({"exchange_rate": exchange_rate, "shopping_list": final_results}, f)

# ==========================================
# COMPONENT 4: THE DASHBOARD LOGGER (PyArrow Added)
# ==========================================
@dsl.component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-bigquery", "pandas", "pyarrow"]
)
def log_results_to_bq(project_id: str, results_input: dsl.Input[dsl.Artifact]):
    from google.cloud import bigquery
    import pandas as pd
    import json
    
    with open(results_input.path, "r") as f:
        data = json.load(f)
    if not data.get('shopping_list'): return
    
    df = pd.DataFrame(data['shopping_list'])
    log_df = pd.DataFrame({
        'Date': df['Date'].astype(int),
        'Ticker': df['ticker'],
        'Risk_Score': df['risk_score'],
        'Allocation_Weight': df['weight'],
        'Execution_Price': df['Close']
    })
    
    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.quant_sa.predictions_history"
    
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    client.load_table_from_dataframe(log_df, table_id, job_config=job_config).result()

# ==========================================
# COMPONENT 5: THE NOTIFIER
# ==========================================
@dsl.component(base_image="python:3.10")
def send_email_notification(
    results_input: dsl.Input[dsl.Artifact],
    sender_email: str, recipient_email: str, email_password: str
):
    import json
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    with open(results_input.path, "r") as f:
        data = json.load(f)

    if not data.get('shopping_list'):
        body = "No assets met criteria today."
    else:
        body = f"Weekly Quant Allocation - (Logged to Dashboard)\n"
        body += f"Live USD/ZAR: R {data['exchange_rate']:.2f}\n"
        body += "-" * 60 + "\n"
        for item in data['shopping_list']:
            body += f"{item['ticker']:<10} | {item['weight']*100:>5.1f}% | R {item['zar_amount']:<10,.2f} | Buy {item['action']}\n"

    msg = MIMEMultipart()
    msg['Subject'] = "⚙️ Quant SA Engine: Monday Allocation Ready"
    msg.attach(MIMEText(body, 'plain'))
    
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, email_password)
    server.sendmail(sender_email, recipient_email, msg.as_string())
    server.quit()

# ==========================================
# THE PIPELINE DAG
# ==========================================
@dsl.pipeline(name="weekly-quant-inference-v1")
def quant_pipeline(
    project_id: str, sender_email: str, recipient_email: str, email_password: str
):
    extract = extract_safe_assets(project_id=project_id)
    fetch = fetch_market_data(safe_tickers=extract.outputs['Output'])
    
    optimize = run_genetic_optimizer(
        clean_data_input=fetch.outputs['clean_data_output'],
        risk_scores_input=extract.outputs['risk_scores_output'],
        capital_zar=100000.0,
        max_volatility=0.40,
        transaction_cost=0.0075,
        risk_free_rate=0.08
    )
    
    log_results_to_bq(project_id=project_id, results_input=optimize.outputs['results_output'])
    
    send_email_notification(
        results_input=optimize.outputs['results_output'],
        sender_email=sender_email,
        recipient_email=recipient_email,
        email_password=email_password
    )

if __name__ == "__main__":
    compiler.Compiler().compile(quant_pipeline, "quant_pipeline_v1.json")
    
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_NAME)
    aiplatform.PipelineJob(
        display_name="Weekly_Quant_Inference_V1",
        template_path="quant_pipeline_v1.json",
        parameter_values={
            "project_id": PROJECT_ID,
            "sender_email": SENDER_EMAIL,
            "recipient_email": RECIPIENT_EMAIL,
            "email_password": EMAIL_PASSWORD
        }
    ).submit()
    print("Pipeline submitted! Check the Vertex AI Pipelines console.")