# $1 -> task_id (because we may use one instance for multiple trainings or even deployments)
pip install -r requirements.txt

mkdir $1

wget -O ./$1/trained_model.zip $2

unzip ./$1/trained_model.zip -d ./$1

# TODO: handle port conflicts when there are multiple deployments on a single instance 

bentoml serve service:ImageClassifyService --port $3