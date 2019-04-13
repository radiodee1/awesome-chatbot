#!/bin/bash

echo "use this to make link to movie or reddit corpus"
echo
echo "enter path to corpus as first parameter."

SIMPLE_PATH=$PWD/data/t2t_data/
DATA=raw.txt
START=$PWD

GLUE_PATH=$PWD/data/glue_data/chat/
DATA_TSV=train.tsv

WORD_PATH=$PWD/data/glue_data/word/

echo
echo $SIMPLE_PATH
echo $START
echo $1
echo

mkdir -p $SIMPLE_PATH
rm $SIMPLE_PATH/$DATA
ln -s $START/$1 $SIMPLE_PATH/$DATA

ls -hal $SIMPLE_PATH

mkdir -p $GLUE_PATH
rm $GLUE_PATH/$DATA_TSV
ln -s $START/$1 $GLUE_PATH/$DATA_TSV

ls -hal $GLUE_PATH

mkdir -p $WORD_PATH
rm $WORD_PATH/$DATA_TSV
ln -s $START/$1 $WORD_PATH/$DATA_TSV

ls -hal $WORD_PATH