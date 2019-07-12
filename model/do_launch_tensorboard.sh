#!/usr/bin/env bash

echo "open chrome browser with url localhost:2222"

DIR1="../saved/t2t_train/babi/"

if [[ -f logdir.txt ]] ; then
    DIR1=`cat logdir.txt`
    echo "use logdir.txt"
fi

tensorboard --logdir $DIR1 --host localhost --port 2222