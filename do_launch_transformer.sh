#!/bin/bash

## typical configurations
echo $@


TEST_4="--mode=long --basename=test_s2s_movie_openvocab_d300_v15000_len15 --load-babi --lr=0.001 --dropout=0.5 --load-recurrent --units=300 --record-loss --multiplier=0.5 --length=15 --no-vocab "
TEST_5=$@

LAUNCH=launch
LOG=log
FILENAME=${HOME}/workspace/log.txt

if [ ! -f ${LAUNCH} ]; then


echo "quit -- no 'launch' file present."
exit

fi
export CHATBOT_MODE="transformer"
export CHATBOT_START="start. transformer model."

if [ -f ${LOG} ]; then
  echo "logging"


  echo ${TEST_5}
  echo "----" >> ${FILENAME}
  date >> ${FILENAME}
  echo "----" >> ${FILENAME}
  cd bot
  python3 -u game.py ${TEST_5} >> ${FILENAME} 2>&1
  exit
else
  cd bot
  python3 game.py ${TEST_5}
  exit
fi

cd bot
python3 game.py ${TEST_5} # 2> /dev/null

## this file must be hard coded for any model you want to run
