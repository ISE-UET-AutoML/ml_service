
docker-compose up

# concurrency: number of workers
celery -A ml-worker.app.celery worker --loglevel=info --concurrency=1 -P threads

python ml-api/app/main.py