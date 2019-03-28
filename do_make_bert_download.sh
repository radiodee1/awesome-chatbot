#!/usr/bin/env bash


cd raw/

wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

mkdir -p ../data/bert_data/

mv uncased_L-12_H-768_A-12.zip ../data/bert_data/.

cd ../data/bert_data/

unzip uncased_L-12_H-768_A-12.zip