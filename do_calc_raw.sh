#!/bin/bash

## typical configurations
echo $@

if [[ -t 0 ]]; then
    echo "Must input text file like: ${0} < test.txt"
    echo "or                         ${0} < data/train.big.from "
    exit
fi

TEST_5=" --no-recent=True $@"

LAUNCH=launch
#export STAT_LIMIT=2000
if [[ -z "${STAT_LIMIT}" ]]; then

  export STAT_LIMIT=2000
fi


if [ ! -f ${LAUNCH} ]; then


echo "quit -- no 'launch' file present."
exit

fi
export CHATBOT_MODE="memory" #"wiki" #"memory"
export CHATBOT_START="hello."


if [[ -z "${STAT_ENUM}" ]]; then
    export STAT_ENUM=0
fi

if [[ -z "${STAT_TAB}" ]]; then
    echo "set STAT_TAB for yourself"
    export STAT_TAB=${STAT_LIMIT}
fi

cd experiments
python3 chat_model_calc.py ${TEST_5}   #2> /dev/null

## this file must be hard coded for any model you want to run
