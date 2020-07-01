#!/bin/bash

## typical configurations
echo $@


TEST_4="--mode=long --basename=test_s2s_movie_openvocab_d300_v15000_len15 --load-babi --lr=0.001 --dropout=0.5 --load-recurrent --units=300 --record-loss --multiplier=0.5 --length=15 --no-vocab "
TEST_5="--length 12 --top_k 1 $@"
TEST_6=$@

LAUNCH=launch

if [ ! -f ${LAUNCH} ]; then


echo "quit -- no 'launch' file present."
exit

fi
export CHATBOT_MODE="signal"
export CHATBOT_START="hello. G P T 2 model."


cd bot
python3 game.py ${TEST_6} # 2> /dev/null

## this file must be hard coded for any model you want to run
