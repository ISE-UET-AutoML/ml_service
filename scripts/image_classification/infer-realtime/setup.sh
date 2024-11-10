
pip install -r requirements.txt

wget -O trained_model.zip $1

unzip trained_model.zip

export INFERENCE_SERVICE_PORT=$2

bentoml serve service:ImageClassifyService 