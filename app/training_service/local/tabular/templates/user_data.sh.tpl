#!/bin/bash
set +e 


export DATA_CENTER="http://112.137.129.161:8888"

export KAGGLE_USERNAME="xunanl"

export KAGGLE_KEY="8334f63590fe60a445fc309643522e9c"

export bucket_name=lexuanan18102004
export project_name=otto-group
export BUCKET_NAME=lexuanan18102004
export PROJECT_NAME=otto-group


aws s3 cp s3://${bucket_name}/${project_name}/ /home/ec2-user/${bucket_name}/${project_name} --recursive

aws s3 cp s3://automl-train-script/ /home/ec2-user/${bucket_name}/${project_name}/ --recursive

# TODO: Slow start on VM when import first time

# shellcheck disable=SC2164
cd /home/ec2-user/${bucket_name}/${project_name}
source /home/ec2-user/venv/bin/activate

pip install split-folders

pip install  kaggle

sudo chmod -R 777 model/

export HOME=/home/ec2-user
parallel "python3 /home/ec2-user/{1}/{2}/{3}.py --data-path='/home/ec2-user/{1}/{2}/{2}.csv' --target='target'" ::: ${bucket_name} ::: ${project_name} ::: autogluon_trainer xg-boost

sudo shutdown now
