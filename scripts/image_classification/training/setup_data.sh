mkdir $1

mkdir ./$1/dataset

wget -O ./$1/dataset/training_data.zip $2 # download dataset through presigned url


unzip ./$1/dataset/training_data.zip -d ./$1/dataset