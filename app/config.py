from dotenv import load_dotenv
import os

load_dotenv()
config = {
    "gcpCredentials": os.getenv("GCP_CREDENTIALS", "../service-account-gcs.json"),
}
