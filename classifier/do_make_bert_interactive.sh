#!/usr/bin/env bash

TAB_FILE=output.t2t.tab.txt
ENU_FILE=output.t2t.enu.txt

if [ ! -f "../data/${TAB_FILE}" ]; then
    cp ../saved/${TAB_FILE} ../data/${TAB_FILE}
    cp ../saved/${ENU_FILE} ../data/${ENU_FILE}
fi


export BERT_BASE_DIR=../data/uncased_L-6_H-512_A-8
export CHAT_DIR=../data/

python3 run_classifier.py \
  --task_name=CHAT \
  --do_train=true \
  --do_predict=true \
  --data_dir=$CHAT_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=0.1 \
  --output_dir=../saved/bert_output/ 2> /dev/null