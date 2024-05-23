import os

from dotenv import load_dotenv

load_dotenv()


class Config(object):

    def __init__(self):
        self.backend_host = os.environ.get('BACKEND_URL', 'localhost:3456/api/v1')
        self.frontend_host = os.environ.get('FRONTEND_URL', 'localhost:3001')
        self.redis_host = os.environ.get('REDIS_HOST', 'localhost:6379')
        self.gcp_storage = os.environ.get('GCP_CREDENTIALS', '')


config = Config()
