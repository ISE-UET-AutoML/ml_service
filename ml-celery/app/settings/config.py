import configparser
import datetime
import pytz
import sys

# settings.py
import os
from os.path import dirname, join, abspath
from dotenv import load_dotenv

# Get the directory one level above the current file
env_path = join(dirname(dirname(abspath(__file__))), ".env")

# Load the .env file from the above directory
load_dotenv(env_path)

BUCKET_NAME = os.environ.get("BUCKET_NAME")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
BUCKET_REGION = os.environ.get("BUCKET_REGION")
CLOUD_INSTANCE_SERVICE_URL = os.environ.get("CLOUD_INSTANCE_SERVICE_URL")
VAST_AI_API_KEY=os.environ.get("VAST_AI_API_KEY")


cfg = configparser.ConfigParser()
cfg.read("./environment.ini")
print(cfg)

if sys.platform.startswith("win32"):
    TEMP_DIR = "D:/tmp"
elif sys.platform.startswith("linux"):
    TEMP_DIR = "./tmp"
# =========================================================================
#                           TIMING CONFIG
# =========================================================================
u = datetime.datetime.utcnow()
u = u.replace(tzinfo=pytz.timezone("Asia/Ho_Chi_Minh"))

# =========================================================================
#                          PROJECT INFORMATION
# =========================================================================
PROJECT = cfg["project"]

# =========================================================================
#                          REDIS INFORMATION
# =========================================================================
REDIS = cfg["redis"]
REDIS_BACKEND = "redis://:{password}@{hostname}:{port}/{db}".format(
    hostname=REDIS["host"], password=REDIS["pass"], port=REDIS["port"], db=REDIS["db"]
)

# =========================================================================
#                          BROKER INFORMATION
# =========================================================================
RABBITMQ = cfg["rabbitmq"]
BROKER = "amqp://{user}:{pw}@{hostname}:{port}/{vhost}".format(
    user=RABBITMQ["user"],
    pw=RABBITMQ["pass"],
    hostname=RABBITMQ["host"],
    port=RABBITMQ["post"],
    vhost=RABBITMQ["vhost"],
)

# =========================================================================
#                          GOOGLE CLOUD INFORMATION
# =========================================================================

GCLOUD = cfg["gcloud"]
GCP_CREDENTIALS = GCLOUD["GCP_CREDENTIALS"]

# =========================================================================
#                           BACKEND INFORMATION
# =========================================================================
BACKEND = cfg["backend"]
BACKEND_HOST = BACKEND["host"]
ACCESS_TOKEN = BACKEND["ACCESS_TOKEN_SECRET"]
REFRESH_TOKEN = BACKEND["REFRESH_TOKEN_SECRET"]

# =========================================================================
#                           DATASERVICE INFORMATION
# =========================================================================
DATASERVICE = cfg["data_service"]
DATASERVICE_HOST = DATASERVICE["host"]


