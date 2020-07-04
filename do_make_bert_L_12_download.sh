#!/usr/bin/env bash

export NAME_BERT=uncased_L-12_H-768_A-12

cd data
mkdir -p ${NAME_BERT}
cd ${NAME_BERT}
touch file
echo ${NAME_BERT} not working
wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip

unzip ${NAME_BERT}.zip