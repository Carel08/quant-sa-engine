import os
from dotenv import load_dotenv
from kfp import dsl, compiler
from google.cloud import aiplatform

# ==========================================
# 0. LOAD ENVIRONMENT VARIABLES
# ==========================================
load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION")
BUCKET_NAME = os.getenv("GCP_BUCKET")

# ==========================================
# 1. THE ETL COMPONENT
# ==========================================
@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "pandas", 
        "yfinance>=0.2.40", 
        "google-cloud-bigquery", 
        "db-dtypes", 
        "numpy", 
        "pyarrow"
    ]
)
def run_weekly_etl(project_id: str):
    import pandas as pd
    import yfinance as yf
    import numpy as np
    from google.cloud import bigquery
    from datetime import datetime, timedelta

    print("Initializing Comprehensive ETL Engine...")
    client = bigquery.Client(project=project_id)
    
    table_id = f"{project_id}.quant_sa.raw_features"
    universe_table = f"{project_id}.quant_sa.asset_universe"
    staging_table = f"{project_id}.quant_sa.etl_staging"

    # 1. Retrieve Active Ticker Universe
    ticker_query = f"SELECT Ticker FROM `{universe_table}` WHERE Is_Active = TRUE"
    ticker_list = client.query(ticker_query).to_dataframe()['Ticker'].tolist()
    
    if not ticker_list:
        return

    # 2. Extract Data
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    raw = yf.download(ticker_list, start=start_date, group_by='ticker', progress=False, threads=False)
    
    all_processed_rows = []
    
    for ticker in ticker_list:
        try:
            df = raw[ticker].copy() if len(ticker_list) > 1 else raw.copy()
            df = df.dropna(subset=['Close'])
            if len(df) < 200:
                continue 

            # --- Basic Metrics ---
            df['daily_return'] = df['Close'].pct_change()
            df['vol_change'] = df['Volume'].pct_change()
            df['log_volume'] = np.log1p(df['Volume'])
            
            # --- Volatility & Trend ---
            df['vol_30d'] = df['daily_return'].rolling(window=30).std() * np.sqrt(252)
            df['dist_SMA_50'] = (df['Close'] - df['Close'].rolling(50).mean()) / df['Close'].rolling(50).mean()
            df['dist_SMA_200'] = (df['Close'] - df['Close'].rolling(200).mean()) / df['Close'].rolling(200).mean()
            
            # --- MACD ---
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = macd - signal

            # --- RSI (Relative Strength Index) ---
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI_14'] = 100 - (100 / (1 + rs))

            # --- MFI (Money Flow Index) ---
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            money_flow = typical_price * df['Volume']
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
            mfr = positive_flow / negative_flow
            df['MFI_14'] = 100 - (100 / (1 + mfr))

            # --- ATR & Bollinger ---
            tr = pd.concat([(df['High'] - df['Low']), abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift())], axis=1).max(axis=1)
            df['ATR_Normalized'] = tr.rolling(window=14).mean() / df['Close']
            sma20 = df['Close'].rolling(window=20).mean()
            std20 = df['Close'].rolling(window=20).std()
            df['BB_Width'] = ((sma20 + std20*2) - (sma20 - std20*2)) / sma20

            # --- Schema Alignment (Future/Metadata Columns) ---
            df['future_price'] = np.nan
            df['target_y'] = 0
            df['Ticker'] = ticker
            df['Date'] = df.index.view('int64')
            df['__index_level_0__'] = 0 # Dummy index level to match schema

            # --- Selection ---
            cols = [
                'Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'log_volume', 
                'vol_change', 'MFI_14', 'BB_Width', 'daily_return', 'vol_30d', 
                'dist_SMA_50', 'dist_SMA_200', 'ATR_Normalized', 'MACD_Hist', 
                'RSI_14', 'future_price', 'target_y', 'Ticker', '__index_level_0__'
            ]
            
            all_processed_rows.append(df[cols].tail(60).reset_index(drop=True))
            
        except Exception as e:
            print(f"Error on {ticker}: {str(e)}")

    if not all_processed_rows: return
    final_df = pd.concat(all_processed_rows)

    # 3. Load and Merge
    print(f"Syncing {len(final_df)} rows across 21 columns...")
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    client.load_table_from_dataframe(final_df, staging_table, job_config=job_config).result()

    merge_sql = f"""
    INSERT INTO `{table_id}` 
    (Date, Close, High, Low, Open, Volume, log_volume, vol_change, MFI_14, BB_Width, daily_return, vol_30d, dist_SMA_50, dist_SMA_200, ATR_Normalized, MACD_Hist, RSI_14, future_price, target_y, Ticker, __index_level_0__)
    SELECT * FROM `{staging_table}` s
    WHERE NOT EXISTS (
        SELECT 1 FROM `{table_id}` t
        WHERE t.Ticker = s.Ticker AND t.Date = s.Date
    )
    """
    client.query(merge_sql).result()
    print("Full schema merge successful.")

# ==========================================
# 2. PIPELINE DEFINITION
# ==========================================
@dsl.pipeline(name="weekly-etl-full-schema")
def etl_pipeline(project_id: str):
    run_weekly_etl(project_id=project_id)

# ==========================================
# 3. EXECUTION
# ==========================================
if __name__ == "__main__":
    compiler.Compiler().compile(etl_pipeline, "etl_pipeline_v1.json")
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_NAME)
    job = aiplatform.PipelineJob(
        display_name="Weekly_ETL_Full_Schema_Backfill",
        template_path="etl_pipeline_v1.json",
        parameter_values={"project_id": PROJECT_ID},
        enable_caching=False
    )
    job.submit()