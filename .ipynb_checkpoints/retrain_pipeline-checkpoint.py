import os
from kfp import dsl
from kfp import compiler

@dsl.component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-bigquery", "pandas", "db-dtypes"]
)
def retrain_xgboost_model(project_id: str):
    from google.cloud import bigquery
    client = bigquery.Client(project=project_id)
    
    table_id = f"{project_id}.quant_sa.raw_features"
    model_id = f"{project_id}.quant_sa.xgboost_risk_filter_final"

    print("Step 1: Labeling matured data (Backfilling target_y)...")
    # We look back 28 days to find the Future Price for rows that are now mature
    label_query = f"""
    UPDATE `{table_id}` t1
    SET t1.target_y = (CASE WHEN (t2.Close - t1.Close) / t1.Close < -0.05 THEN 1 ELSE 0 END)
    FROM `{table_id}` t2
    WHERE t1.Ticker = t2.Ticker 
      AND t1.target_y IS NULL
      AND t2.Date = CAST(FORMAT_DATE('%Y%m%d', DATE_ADD(PARSE_DATE('%Y%m%d', CAST(t1.Date AS STRING)), INTERVAL 28 DAY)) AS INT64)
    """
    client.query(label_query).result()
    print("Labels updated successfully.")

    print("Step 2: Executing Monthly Brain Transplant (Retraining)...")
    retrain_query = f"""
    CREATE OR REPLACE MODEL `{model_id}`
    OPTIONS (
      MODEL_TYPE = 'BOOSTED_TREE_CLASSIFIER',
      INPUT_LABEL_COLS = ['target_y'],
      DATA_SPLIT_METHOD = 'SEQUENTIAL',
      DATA_SPLIT_EVAL_FRACTION = 0.1,
      BOOSTER_TYPE = 'GBTREE',
      MAX_ITERATIONS = 20,
      EARLY_STOP = TRUE,
      HPARAM_TUNING_OBJECTIVES = ['ROC_AUC'],
      LEARN_RATE = HPARAM_RANGE(0.01, 0.2),
      MAX_TREE_DEPTH = HPARAM_CANDIDATES([4, 6, 8]),
      SUBSAMPLE = HPARAM_RANGE(0.6, 0.9)
    ) AS
    SELECT 
      Ticker, vol_30d, dist_SMA_50, dist_SMA_200, 
      MACD_Hist, log_volume, ATR_Normalized, BB_Width, target_y
    FROM `{table_id}`
    WHERE target_y IS NOT NULL
    """
    client.query(retrain_query).result()
    print("Model retraining complete. v2 is now LIVE.")

@dsl.pipeline(name="monthly-retraining-pipeline")
def training_pipeline(project_id: str):
    retrain_xgboost_model(project_id=project_id)

if __name__ == "__main__":
    compiler.Compiler().compile(training_pipeline, "retrain_pipeline_v1.json")
    print("Retraining blueprint compiled to retrain_pipeline_v1.json")