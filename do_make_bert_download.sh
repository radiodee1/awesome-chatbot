#!/usr/bin/env bash


cd raw/

wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

mkdir -p ../data/bert_data/
mkdir -p ../saved/bert_saved/

mv uncased_L-12_H-768_A-12.zip ../data/bert_data/.

cd ../data/bert_data/

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
#wget https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py

unzip uncased_L-12_H-768_A-12.zip