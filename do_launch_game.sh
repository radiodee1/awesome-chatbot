#!/bin/bash

## typical configurations
TEST_1="--mode=long --basename=test_s2s_long_b_d300_v15000 --load-babi --lr=0.0001 --dropout=0.5 --recurrent-output --load-recurrent --hide-unk --units=300 --skip-unk"
TEST_2="--mode=long --basename=test_s2s_freeze_d300_v15000 --load-babi --lr=0.0001 --dropout=0.5 --recurrent-output --load-recurrent --hide-unk --units=300 --skip-unk --load-embed-size=300 --freeze-embedding"
TEST_3="--mode=long --basename=test_s2s_size_d300_v15000 --load-babi --lr=0.0001 --dropout=0.5 --recurrent-output --load-recurrent --hide-unk --units=300 --skip-unk"

LAUNCH=launch

if [ ! -f ${LAUNCH} ]; then


echo "quit -- no 'launch' file present."
exit

fi

cd bot
python3.6 game.py ${TEST_2} # 2> /dev/null

## this file must be hard coded for any model you want to run