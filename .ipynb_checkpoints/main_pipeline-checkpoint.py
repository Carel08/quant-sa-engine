import os
from dotenv import load_dotenv
from kfp import dsl, compiler
from google.cloud import aiplatform

# ==========================================
# 0. LOAD SECRETS FROM .env (Local Safe)
# ==========================================
load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION")
BUCKET_NAME = os.getenv("GCP_BUCKET")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")

# ==========================================
# COMPONENT 1: THE DATA ENGINEER (BigQuery)
# ==========================================
@dsl.component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-bigquery", "pandas", "db-dtypes"]
)
def extract_safe_assets(project_id: str) -> list:
    from google.cloud import bigquery
    
    print(f"Connecting to BigQuery in project: {project_id}...")
    bq_client = bigquery.Client(project=project_id)
    
    # THE COST-SAVING SQL UPGRADE
    # Instead of sorting all history, we only grab the row matching the MAX date per ticker.
    predict_query = f"""
    WITH LatestDates AS (
      SELECT Ticker, MAX(Date) as max_date
      FROM `{project_id}.quant_sa.raw_features`
      WHERE Ticker NOT LIKE '^%'
      GROUP BY Ticker
    )
    SELECT Ticker, ROUND(predicted_target_y_probs[OFFSET(0)].prob, 4) AS safe_probability
    FROM ML.PREDICT(MODEL `{project_id}.quant_sa.xgboost_risk_filter_final`,
      (
        SELECT r.* FROM `{project_id}.quant_sa.raw_features` r
        INNER JOIN LatestDates l 
          ON r.Ticker = l.Ticker AND r.Date = l.max_date
      )
    ) 
    ORDER BY safe_probability DESC 
    LIMIT 20;
    """
    
    # We add job_config to put a hard limit on costs. 
    # If it tries to scan more than 5GB (approx $0.03), it will automatically fail instead of billing you!
    job_config = bigquery.QueryJobConfig(
        maximum_bytes_billed=5000000000  # 5 GB limit
    )
    
    predictions_df = bq_client.query(predict_query, job_config=job_config).to_dataframe()
    safe_tickers = predictions_df['Ticker'].tolist()
    
    print(f"Extracted {len(safe_tickers)} safe assets efficiently.")
    return safe_tickers

# ==========================================
# COMPONENT 2: THE MARKET FETCHER (Dynamic)
# ==========================================
@dsl.component(
    base_image="python:3.10",
    packages_to_install=["yfinance", "pandas", "numpy"]
)
def fetch_market_data(
    safe_tickers: list, 
    clean_data_output: dsl.Output[dsl.Dataset],
    latest_prices_output: dsl.Output[dsl.Artifact]
):
    import yfinance as yf
    import pandas as pd
    import json
    from datetime import datetime, timedelta
    
    # Dynamic Time Window: Last 5 Years leading up to "Today"
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    print(f"Downloading historical data from {start_date} to {end_date}...")
    raw_data = yf.download(safe_tickers, start=start_date, end=end_date, progress=False)['Close']
    
    raw_data.index = pd.to_datetime(raw_data.index).tz_localize(None)
    full_calendar = pd.date_range(start=raw_data.index.min(), end=raw_data.index.max(), freq='D')
    df = raw_data.reindex(full_calendar).ffill().dropna(how='all')
    
    # 1. Save absolute latest prices for real-world execution
    latest_prices = df.iloc[-1].to_dict()
    with open(latest_prices_output.path, "w") as f:
        json.dump(latest_prices, f)

    # 2. Winsorization: Clip daily anomalies to 15% and save clean returns
    daily_returns = df.pct_change().fillna(0).clip(lower=-0.15, upper=0.15)
    daily_returns.to_csv(clean_data_output.path)
    print("Market data cleaned, clipped, and saved to artifact storage.")

# ==========================================
# COMPONENT 3: THE QUANT BRAIN & EXECUTION
# ==========================================
@dsl.component(
    base_image="python:3.10",
    packages_to_install=["pandas", "numpy", "yfinance"] 
)
def run_genetic_optimizer(
    clean_data_input: dsl.Input[dsl.Dataset],
    latest_prices_input: dsl.Input[dsl.Artifact],
    capital_zar: float,
    max_volatility: float,
    transaction_cost: float,
    risk_free_rate: float,
    shopping_list_output: dsl.Output[dsl.Artifact]
):
    import pandas as pd
    import numpy as np
    import random
    import json
    import yfinance as yf

    # --- 1. Risk Math & Constraints ---
    daily_returns = pd.read_csv(clean_data_input.path, index_col=0)
    annual_returns = daily_returns.mean() * 252
    cov_matrix = daily_returns.cov() * 252
    num_assets = len(daily_returns.columns)
    safe_tickers = daily_returns.columns.tolist()
    
    def calculate_fitness(weights):
        # Dynamic Rates applied here
        net_return = np.sum(annual_returns * weights) - (np.sum(weights) * transaction_cost)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        if port_volatility == 0 or port_volatility > max_volatility: return -100 
        return (net_return - risk_free_rate) / port_volatility

    # --- 2. Genetic Evolution Engine ---
    print("Igniting Genetic Evolution Engine...")
    population = [np.random.random(num_assets) / np.sum(np.random.random(num_assets)) for _ in range(300)]
    for _ in range(50):
        population = sorted(population, key=lambda ind: calculate_fitness(ind), reverse=True)
        next_gen = population[:60]
        while len(next_gen) < 300:
            p1, p2 = random.sample(population[:100], 2)
            cross_pt = random.randint(1, num_assets - 1)
            child = np.concatenate((p1[:cross_pt], p2[cross_pt:]))
            if random.random() < 0.15: # Mutation
                idx1, idx2 = random.sample(range(num_assets), 2)
                swap = child[idx1] * random.random()
                child[idx1] -= swap
                child[idx2] += swap
            child = child / np.sum(child)
            next_gen.append(child)
        population = next_gen

    best_weights = population[0]
    expected_profit = float(np.sum(annual_returns * best_weights) * capital_zar)

    # --- 3. Live Execution & Shares Calculation ---
    # Pull live USD/ZAR for crypto sizing
    usdzar_data = yf.download("USDZAR=X", period="5d", progress=False)
    exchange_rate = float(usdzar_data['Close'].iloc[-1].squeeze())

    with open(latest_prices_input.path, "r") as f:
        latest_prices = json.load(f)

    shopping_list = []
    
    for i in range(num_assets):
        asset = safe_tickers[i]
        weight = float(best_weights[i])
        zar_allocated = weight * capital_zar 
        
        # Filter out dust (allocations under R50)
        if zar_allocated < 50:
            continue
            
        raw_price = latest_prices[asset]

        if "USD" in asset:
            usd_budget = zar_allocated / exchange_rate
            units = usd_budget / raw_price
            shares_str = f"{units:.6f} coins"
        else:
            price_in_rand = raw_price / 100
            units = int(np.floor(zar_allocated / price_in_rand))
            shares_str = f"{units} shares"

        shopping_list.append({
            "asset": asset,
            "weight": weight,
            "zar_amount": zar_allocated,
            "to_buy": shares_str
        })

    shopping_list = sorted(shopping_list, key=lambda x: x['weight'], reverse=True)

    # Package the final output for the Notifier
    final_output = {
        "expected_profit": expected_profit,
        "exchange_rate": exchange_rate,
        "shopping_list": shopping_list
    }
    
    with open(shopping_list_output.path, "w") as f:
        json.dump(final_output, f)
    
    print("Optimization and execution sizing complete.")

# ==========================================
# COMPONENT 4: THE EMAIL NOTIFIER
# ==========================================
@dsl.component(
    base_image="python:3.10"
)
def send_email_notification(
    shopping_list_input: dsl.Input[dsl.Artifact],
    sender_email: str,
    recipient_email: str,
    email_password: str
):
    import json
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    with open(shopping_list_input.path, "r") as f:
        data = json.load(f)

    # Format the beautiful real-world shopping list
    body = f"Weekly All-Weather Portfolio Optimized \n"
    body += f"Live USD/ZAR Exchange Rate: R {data['exchange_rate']:.2f}\n"
    body += f"Expected Annual ZAR Profit: R {data['expected_profit']:,.2f}\n"
    body += "-" * 55 + "\n"
    body += f"{'Asset':<10} | {'Weight':<7} | {'ZAR Amount':<12} | {'Action':<15}\n"
    body += "-" * 55 + "\n"
    
    for item in data['shopping_list']:
        body += f"{item['asset']:<10} | {item['weight']*100:>5.1f}% | R {item['zar_amount']:<10,.2f} | Buy {item['to_buy']}\n"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = "⚙️ Quant SA Engine: Monday Allocation Ready"
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, email_password)
        server.send_message(msg)
        server.quit()
        print("Success: Detailed shopping list emailed to user.")
    except Exception as e:
        print(f"Failed to send email: {e}")

# ==========================================
# THE PIPELINE DAG & EXECUTION
# ==========================================
@dsl.pipeline(
    name="weekly-quant-inference-pipeline",
    description="Extracts safe assets, dynamically fetches data, runs GA execution, and emails results."
)
def quant_inference_pipeline(
    project_id: str, 
    sender_email: str,
    recipient_email: str,
    email_password: str,
    capital_zar: float = 100000.0, 
    max_volatility: float = 0.40,
    transaction_cost: float = 0.0075,
    risk_free_rate: float = 0.08
):
    extract_task = extract_safe_assets(project_id=project_id)
    
    fetch_task = fetch_market_data(safe_tickers=extract_task.output)
    
    optimize_task = run_genetic_optimizer(
        clean_data_input=fetch_task.outputs['clean_data_output'],
        latest_prices_input=fetch_task.outputs['latest_prices_output'], 
        capital_zar=capital_zar,
        max_volatility=max_volatility,
        transaction_cost=transaction_cost,
        risk_free_rate=risk_free_rate
    )

    send_email_notification(
        shopping_list_input=optimize_task.outputs['shopping_list_output'], 
        sender_email=sender_email,
        recipient_email=recipient_email,
        email_password=email_password
    )

if __name__ == "__main__":
    if not PROJECT_ID or not BUCKET_NAME:
        raise ValueError("Missing environment variables. Check your .env file.")

    print(f"Compiling pipeline for {PROJECT_ID}...")
    compiler.Compiler().compile(
        pipeline_func=quant_inference_pipeline,
        package_path="quant_pipeline_v1.json"
    )

    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_NAME)

    print("Deploying pipeline to Vertex AI...")
    job = aiplatform.PipelineJob(
        display_name="Weekly_Quant_Run",
        template_path="quant_pipeline_v1.json",
        parameter_values={
            "project_id": PROJECT_ID,
            "sender_email": SENDER_EMAIL,
            "recipient_email": RECIPIENT_EMAIL,
            "email_password": EMAIL_PASSWORD,
            "capital_zar": 100000.0,
            "max_volatility": 0.40,
            "transaction_cost": 0.0075, # Dynamic Parameters Injected Here
            "risk_free_rate": 0.08      # Dynamic Parameters Injected Here
        },
        enable_caching=True
    )
    job.submit()
    print("Pipeline submitted! Check the Vertex AI Pipelines console.")