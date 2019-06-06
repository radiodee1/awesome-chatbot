#!/bin/bash

../seq_2_seq/do_split.py $@

echo "try this:"
echo "./do_split_run.sh --filename ../../rc-movie-raw.txt --to-gpt2 --length=500 --mode=train.big --zip=gpt2_chatbot"