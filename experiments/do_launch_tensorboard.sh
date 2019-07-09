#!/usr/bin/env bash

echo "open chrome browser with url localhost:2222"

tensorboard --logdir ../saved/t2t_train/babi/ --host localhost --port 2222