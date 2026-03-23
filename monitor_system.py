import os
import smtplib
from email.message import EmailMessage
from google.cloud import aiplatform
from dotenv import load_dotenv

# 1. Load Environment
load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION")
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASS = os.getenv("GMAIL_PASS")

aiplatform.init(project=PROJECT_ID, location=REGION)

def send_alert(pipeline_name, error_msg):
    """Sends a priority alert to your Gmail."""
    msg = EmailMessage()
    msg.set_content(f"CRITICAL: The {pipeline_name} has failed.\n\nError Context: {error_msg}\n\nPlease log into GCP Console to debug.")
    msg['Subject'] = f"PIPELINE FAILURE: {pipeline_name}"
    msg['From'] = GMAIL_USER
    msg['To'] = GMAIL_USER

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(GMAIL_USER, GMAIL_PASS)
        smtp.send_message(msg)
    print(f"Email alert sent for {pipeline_name}")

def audit_pipelines():
    # The display names of the schedules we created
    monitored_schedules = [
        "Sunday_ETL_Execution",
        "Monday_Quant_Execution",
        "Monthly_Retraining_Schedule"
    ]

    print("Checking pipeline health...")
    
    # Get all pipeline jobs in the last 7 days
    jobs = aiplatform.PipelineJob.list(filter='state="PIPELINE_STATE_FAILED"', order_by="create_time desc")
    
    # Check if any of the failures belong to our 'Big Three'
    for job in jobs:
        # Check if the job name contains our schedule display names (or related keywords)
        # Vertex AI typically prefixes job names with the schedule name
        for schedule in monitored_schedules:
            if schedule.lower() in job.display_name.lower():
                # Check if this failure happened in the last 24 hours to avoid repeat alerts
                # (Simple check: we'll just print it for now)
                print(f"Failure detected: {job.display_name} at {job.create_time}")
                send_alert(schedule, f"Job {job.display_name} failed with state {job.state}")

if __name__ == "__main__":
    audit_pipelines()