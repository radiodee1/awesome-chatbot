#!/bin/bash

## typical configurations
echo $@


TEST_5="--name chat_movie_30 $@"

LAUNCH=../launch
#export STAT_LIMIT=2000

if [ ! -f ${LAUNCH} ]; then


echo "quit -- no 'launch' file present."
exit

fi
export CHATBOT_MODE="transformer"
export CHATBOT_START="hello."



python3 tf_t2t_train_run.py ${TEST_5}   #2> /dev/null

## this file must be hard coded for any model you want to run
