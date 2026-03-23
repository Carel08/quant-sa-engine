import os
from dotenv import load_dotenv
from google.cloud import aiplatform

# 1. Load your hidden keys
load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION")
BUCKET_NAME = os.getenv("GCP_BUCKET")

# 2. Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_NAME)

print("Creating Sunday Night ETL Schedule...")

# 3. Define the Pipeline Job
job = aiplatform.PipelineJob(
    display_name="Weekly_Data_Update",
    template_path="etl_pipeline_v1.json", 
    parameter_values={"project_id": PROJECT_ID}
)

# 4. Hand to the Scheduler
schedule = aiplatform.PipelineJobSchedule(
    pipeline_job=job,
    display_name="Sunday_ETL_Execution"
)

# 5. Set the Cron Timer (0 20 * * 0 = Sunday at 8:00 PM SAST)
# This gives it plenty of time to finish before the Monday 8:00 AM Inference engine wakes up.
schedule.create(
    cron="TZ=Africa/Johannesburg 0 20 * * 0"
)

print("ETL Engine is on Autopilot! It will run every Sunday at 20:00 SAST.")