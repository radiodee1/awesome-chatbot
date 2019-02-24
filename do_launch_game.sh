#!/bin/bash

## typical configurations
TEST_1="--mode=long --basename=test_s2s_long_b_d300_v15000 --load-babi --lr=0.0001 --dropout=0.5 --recurrent-output --load-recurrent --hide-unk --units=300 --skip-unk"
TEST_2="--mode=long --basename=test_s2s_freeze_d300_v15000 --load-babi --lr=0.0001 --dropout=0.5 --recurrent-output --load-recurrent --hide-unk --units=300 --skip-unk --load-embed-size=300 --freeze-embedding"
TEST_3="--mode=long --basename=test_s2s_long_b_d300_v15000 --load-babi --lr=0.0001 --dropout=0.5 --recurrent-output --load-recurrent --hide-unk --units=300 --skip-unk"


cd bot
python3.6 game.py $TEST_1  2> /dev/null
