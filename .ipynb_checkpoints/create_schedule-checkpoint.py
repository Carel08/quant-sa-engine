import os
from dotenv import load_dotenv
from google.cloud import aiplatform

# 1. Load your hidden keys
load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION")
BUCKET_NAME = os.getenv("GCP_BUCKET")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")

# 2. Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_NAME)

print("Creating Monday Morning Schedule...")

# 3. FIRST: Define the PipelineJob (just like we do for a manual run)
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
        "transaction_cost": 0.0075,
        "risk_free_rate": 0.08
    }
)

# 4. SECOND: Hand that job object to the Scheduler
schedule = aiplatform.PipelineJobSchedule(
    pipeline_job=job, # THE FIX: Pass the job object here!
    display_name="Monday_Quant_Execution"
)

# 5. Set the Cron Timer
schedule.create(
    cron="TZ=Africa/Johannesburg 0 8 * * 1"
)

print("Schedule activated! The engine will run automatically every Monday at 08:00 SAST.")