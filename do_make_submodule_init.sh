#!/usr/bin/env bash

git submodule init
git submodule update

cd model/torch_gpt2

curl --output gpt2-pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin

cd ..

./tf_gpt2_download_model.py 117M

cd torch_gpt2/GPT2

cp ../gpt2-pytorch_model.bin .

