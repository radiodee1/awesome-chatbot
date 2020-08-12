#!/bin/bash

## typical configurations
echo $@


TEST_4="$@ --apps True --source_file ../data/tf_gpt2_data/1558M/converted/pytorch_model.bin"
TEST_5="$@ --apps True"

LAUNCH=launch
LOG=log
FILENAME=${HOME}/workspace/log.txt
PYTORCH_MODEL='data/tf_gpt2_data/1558M/converted/pytorch_model.bin'

if [ -f ${PYTORCH_MODEL} ]; then
  TEST_5=${TEST_4}
  echo $TEST_5
fi

if [ ! -f ${LAUNCH} ]; then


echo "quit -- no 'launch' file present."
exit

fi
export CHATBOT_MODE="apps"
export CHATBOT_START="start. G P T 2 apps model."

cd bot
python3 game.py ${TEST_5} # 2> /dev/null

## this file must be hard coded for any model you want to run
