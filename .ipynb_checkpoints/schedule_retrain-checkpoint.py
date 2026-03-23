import os
from dotenv import load_dotenv
from google.cloud import aiplatform

load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION")
BUCKET = os.getenv("GCP_BUCKET")

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET)

job = aiplatform.PipelineJob(
    display_name="Monthly_Model_Retrain",
    template_path="retrain_pipeline_v1.json",
    parameter_values={"project_id": PROJECT_ID}
)

schedule = aiplatform.PipelineJobSchedule(
    pipeline_job=job,
    display_name="Monthly_Retraining_Schedule"
)

# Cron for 01:00 AM on the 1st day of every month
schedule.create(
    cron="TZ=Africa/Johannesburg 0 1 1 * *"
)

print("Monthly Retraining Autopilot is ARMED.")