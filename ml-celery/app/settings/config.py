import configparser
import datetime
import pytz

import sys

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
