#!/bin/bash

## typical configurations
echo $@

if [[ -t 0 ]]; then
    echo "Must input text file like: ${0} < test.txt"
    echo "or                         ${0} < data/train.big.from "
    exit
fi

TEST_5="--name chat_movie_30 $@"

LAUNCH=launch
export STAT_LIMIT=2000

if [ ! -f ${LAUNCH} ]; then


echo "quit -- no 'launch' file present."
exit

fi
export CHATBOT_MODE="memory"
export CHATBOT_START="hello."

cd experiments
python3 chat_model_calc.py ${TEST_5}   #2> /dev/null

## this file must be hard coded for any model you want to run
