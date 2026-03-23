import os
from dotenv import load_dotenv
from kfp import dsl
from kfp import compiler

# ==========================================
# 1. THE SERVERLESS COMPONENT
# ==========================================
@dsl.component(
    base_image="python:3.10",
    packages_to_install=["pandas", "yfinance", "google-cloud-bigquery", "db-dtypes", "numpy"]
)
def run_weekly_etl(project_id: str):
    import pandas as pd
    import yfinance as yf
    import numpy as np
    from google.cloud import bigquery
    from datetime import datetime, timedelta

    print("Booting up Weekly ETL Engine...")
    client = bigquery.Client(project=project_id)
    
    table_id = f"{project_id}.quant_sa.raw_features"
    universe_table = f"{project_id}.quant_sa.asset_universe"
    staging_table = f"{project_id}.quant_sa.etl_staging"

    # 1. Read Master Universe
    print("🔍 Reading the Master Asset Universe...")
    ticker_query = f"SELECT Ticker FROM `{universe_table}` WHERE Is_Active = TRUE"
    ticker_list = client.query(ticker_query).to_dataframe()['Ticker'].tolist()
    
    if not ticker_list:
        print("No active tickers found. Exiting.")
        return

    # 2. Extract & Transform
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    print(f"Downloading data for {len(ticker_list)} assets...")
    raw = yf.download(ticker_list, start=start_date, interval="1d", group_by='ticker', progress=False, threads=False)
    
    all_processed_rows = []
    for ticker in ticker_list:
        try:
            df = raw.copy() if len(ticker_list) == 1 else raw[ticker].copy()
            df = df.dropna(subset=['Close'])
            if len(df) < 200: continue 
            
            # Math
            sma50 = df['Close'].rolling(window=50).mean()
            sma200 = df['Close'].rolling(window=200).mean()
            df['dist_SMA_50'] = (df['Close'] - sma50) / sma50
            df['dist_SMA_200'] = (df['Close'] - sma200) / sma200
            df['returns'] = df['Close'].pct_change()
            df['vol_30d'] = df['returns'].rolling(window=30).std() * np.sqrt(252)
            
            ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = macd_line - signal_line
            df['log_volume'] = np.log1p(df['Volume'])

            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
            df['ATR_Normalized'] = atr / df['Close']

            sma20 = df['Close'].rolling(window=20).mean()
            std20 = df['Close'].rolling(window=20).std()
            upper_band = sma20 + (std20 * 2)
            lower_band = sma20 - (std20 * 2)
            df['BB_Width'] = (upper_band - lower_band) / sma20

            # Formatting
            df['Ticker'] = ticker
            df.index.name = None 
            df['Date'] = df.index.strftime('%Y%m%d').astype(int)
            
            cols = ['Ticker', 'Date', 'vol_30d', 'dist_SMA_50', 'dist_SMA_200', 'MACD_Hist', 'log_volume', 'ATR_Normalized', 'BB_Width']
            all_processed_rows.append(df[cols].tail(10).reset_index(drop=True))
        except Exception:
            pass

    processed_df = pd.concat(all_processed_rows) if all_processed_rows else pd.DataFrame()

    # 3. Load (Merge)
    if processed_df.empty: return
    
    print("Uploading to staging...")
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    client.load_table_from_dataframe(processed_df, staging_table, job_config=job_config).result()

    print("🔄 Merging into main feature table...")
    merge_sql = f"""
    INSERT INTO `{table_id}` 
        (Ticker, Date, vol_30d, dist_SMA_50, dist_SMA_200, MACD_Hist, log_volume, ATR_Normalized, BB_Width)
    SELECT s.Ticker, s.Date, s.vol_30d, s.dist_SMA_50, s.dist_SMA_200, s.MACD_Hist, s.log_volume, s.ATR_Normalized, s.BB_Width 
    FROM `{staging_table}` s
    WHERE NOT EXISTS (
        SELECT 1 FROM `{table_id}` t
        WHERE t.Ticker = s.Ticker AND t.Date = s.Date
    )
    """
    client.query(merge_sql).result()
    print(f"Weekly ETL Complete!")

# ==========================================
# 2. THE PIPELINE DEFINITION
# ==========================================
@dsl.pipeline(name="weekly-etl-pipeline", description="Fetches weekly data and appends to BigQuery")
def etl_pipeline(project_id: str):
    run_weekly_etl(project_id=project_id)

# ==========================================
# 3. COMPILER
# ==========================================
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=etl_pipeline,
        package_path="etl_pipeline_v1.json"
    )
    print("Pipeline compiled to etl_pipeline_v1.json")