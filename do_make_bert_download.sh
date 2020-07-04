#!/usr/bin/env bash

export NAME_BERT=uncased_L-6_H-512_A-8

cd data
mkdir -p ${NAME_BERT}
cd ${NAME_BERT}
touch file
echo ${NAME_BERT} not working
wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-6_H-512_A-8.zip

unzip ${NAME_BERT}.zip