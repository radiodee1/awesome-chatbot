#!/usr/bin/env bash

if [[ -z "${INFILE}" ]]; then
    export INFILE=two
fi

#INFILE=gpt
OUTFILE=xxx

TAB_FILE=output.${INFILE}.tab.txt
ENU_FILE=output.${INFILE}.enu.txt

TAB_FILE_OUT=output.${OUTFILE}.tab.txt
ENU_FILE_OUT=output.${OUTFILE}.enu.txt

if [ ! -f "../data/${TAB_FILE_OUT}" ]; then
    cp ../saved/${TAB_FILE} ../data/${TAB_FILE_OUT}
    cp ../saved/${ENU_FILE} ../data/${ENU_FILE_OUT}
fi


export BERT_BASE_DIR=../data/uncased_L-6_H-512_A-8 #uncased_L-12_H-768_A-12
export CHAT_DIR=../data/

python3 run_pretraining_two.py \
  --input_file=../saved/bert_output/input_file.txt \
  --output_dir=../saved/bert_output \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --do_eval \
  --eval_batch_size=1 \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --use_tpu=False

exit 0

