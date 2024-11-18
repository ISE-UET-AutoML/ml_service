#!/bin/bash


rm -r ./$1/dataset

rm -r ./$1/model

rm -r ./$1/*.zip

rm -r ./$1/training_logs.txt

rm -r ./$1/*.json

rmdir $1