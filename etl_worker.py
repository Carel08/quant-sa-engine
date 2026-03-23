import os
import pandas as pd
import yfinance as yf
import numpy as np
from google.cloud import bigquery
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ==========================================
# 0. CONFIGURATION & SECRETS
# ==========================================
load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
TABLE_ID = f"{PROJECT_ID}.quant_sa.raw_features"

# ==========================================
# 1. EXTRACT & TRANSFORM (Pure Pandas)
# ==========================================
def fetch_and_engineer(tickers):
    """Downloads 250 days of data and calculates XGBoost features using PURE PANDAS."""
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f"Downloading market data for {len(tickers)} assets...")
    # threads=False prevents SQLite database locks during download
    raw = yf.download(tickers, start=start_date, interval="1d", group_by='ticker', progress=False, threads=False)
    
    all_processed_rows = []

    for ticker in tickers:
        try:
            if len(tickers) == 1:
                df = raw.copy()
            else:
                df = raw[ticker].copy()
                
            df = df.dropna(subset=['Close'])
            if len(df) < 200: 
                continue 
            
            # --- PURE PANDAS FEATURE ENGINEERING ---
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

            # --- METADATA & CLEANUP ---
            df['Ticker'] = ticker
            
            # Formats the datetime index into an INTEGER (e.g., 20260323) to match your BigQuery schema
            df.index.name = None 
            df['Date'] = df.index.strftime('%Y%m%d').astype(int)

            cols = ['Ticker', 'Date', 'vol_30d', 'dist_SMA_50', 'dist_SMA_200', 
                    'MACD_Hist', 'log_volume', 'ATR_Normalized', 'BB_Width']
            
            # reset_index cleanly detaches the index to avoid PyArrow duplicate column errors
            clean_df = df[cols].tail(10).reset_index(drop=True)
            all_processed_rows.append(clean_df)
            
        except Exception as e:
            # Silently skip errors like delisted stocks to keep the pipeline moving
            pass

    return pd.concat(all_processed_rows) if all_processed_rows else pd.DataFrame()

# ==========================================
# 2. LOAD (BigQuery Merge)
# ==========================================
def upload_to_bigquery(df):
    """Appends new records to BigQuery, mapping strictly to the 9 required columns."""
    if df.empty:
        print("No new data to upload.")
        return

    client = bigquery.Client(project=PROJECT_ID)
    
    print("Uploading to staging table...")
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    staging_table = f"{PROJECT_ID}.quant_sa.etl_staging"
    client.load_table_from_dataframe(df, staging_table, job_config=job_config).result()

    print("Merging new rows into main feature table...")
    
    # Explicitly mapping the 9 columns so BigQuery ignores the 12 legacy columns
    merge_sql = f"""
    INSERT INTO `{TABLE_ID}` 
        (Ticker, Date, vol_30d, dist_SMA_50, dist_SMA_200, MACD_Hist, log_volume, ATR_Normalized, BB_Width)
    SELECT 
        s.Ticker, 
        s.Date, 
        s.vol_30d, 
        s.dist_SMA_50, 
        s.dist_SMA_200, 
        s.MACD_Hist, 
        s.log_volume, 
        s.ATR_Normalized, 
        s.BB_Width 
    FROM `{staging_table}` s
    WHERE NOT EXISTS (
        SELECT 1 FROM `{TABLE_ID}` t
        WHERE t.Ticker = s.Ticker AND t.Date = s.Date
    )
    """
    client.query(merge_sql).result()
    print(f"ETL Complete: New features successfully merged into {TABLE_ID}")

# ==========================================
# 3. EXECUTION
# ==========================================
if __name__ == "__main__":
    if not PROJECT_ID:
        raise ValueError("Missing GCP_PROJECT_ID. Check your .env file.")

    bq_client = bigquery.Client(project=PROJECT_ID)
    
    print("🔍 Reading the Master Asset Universe control panel...")
    ticker_query = f"""
        SELECT Ticker 
        FROM `{PROJECT_ID}.quant_sa.asset_universe` 
        WHERE Is_Active = TRUE
    """
    
    try:
        ticker_list = bq_client.query(ticker_query).to_dataframe()['Ticker'].tolist()
        print(f"Found {len(ticker_list)} ACTIVE tickers in the universe.")
        
        if ticker_list:
            processed_df = fetch_and_engineer(ticker_list)
            upload_to_bigquery(processed_df)
        else:
            print("No active tickers found. Flip the 'Is_Active' switch in BigQuery!")
            
    except Exception as e:
        print(f"Failed to run ETL: {e}")