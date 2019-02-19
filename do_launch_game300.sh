#!/bin/bash

cd bot
python3.6 game.py --mode=long --basename=test_s2s_long_b_d300_v15000 --load-babi --lr=0.0001 --dropout=0.5 --recurrent-output --load-recurrent --hide-unk --units=300 --skip-unk  2> /dev/null
