#!/bin/bash


cd data

BIGTRAINFROM=train.big.from
BIGTRAINTO=train.big.to
BIGTRAINQUES=train.big.ques

BIGVALFROM=valid.big.from
BIGVALTO=valid.big.to
BIGVALQUES=valid.big.ques

BIGTESTFROM=test.big.from
BIGTESTTO=test.big.to
BIGTESTQUES=test.big.ques

TRAIN=train
VALID=valid
TEST=test

echo "usage: $0 <namespace word>"
echo "example: $0 babi"
echo "    this will point symbolic links to files 'train.babi.from', 'train.babi.to' , etc."


if [ $# -eq 1 ] ; then
    echo "pointing to " $1

    #exit()
BIGTRAINFROM=train.$1.from
BIGTRAINTO=train.$1.to
BIGTRAINQUES=train.$1.ques

BIGVALFROM=valid.$1.from
BIGVALTO=valid.$1.to
BIGVALQUES=valid.$1.ques

BIGTESTFROM=test.$1.from
BIGTESTTO=test.$1.to
BIGTESTQUES=test.$1.ques

fi
#else
    echo ""
    echo "changing link to large initial file! "

    if [ -f $BIGTRAINFROM ] && [ -f $BIGTRAINTO ] && [ -f $BIGTRAINQUES ] ; then
        rm $TRAIN.from $TRAIN.to $TRAIN.ques
        ln -s $BIGTRAINFROM $TRAIN.from
        ln -s $BIGTRAINTO $TRAIN.to
        ln -s $BIGTRAINQUES $TRAIN.ques
        rm $VALID.from $VALID.to $VALID.ques
        ln -s $BIGVALFROM $VALID.from
        ln -s $BIGVALTO $VALID.to
        ln -s $BIGVALQUES $VALID.ques
        rm $TEST.from $TEST.to $TEST.ques
        ln -s $BIGTESTFROM $TEST.from
        ln -s $BIGTESTTO $TEST.to
        ln -s $BIGTESTQUES $TEST.ques


    else
        echo "doing nothing"
        #mv $TRAIN.from $BIGTRAINFROM
        #mv $TRAIN.to $BIGTRAINTO
        echo "try:"
        echo "ls -hal"
    fi
    #ls -hal

#fi
echo ""
ls -hal $TRAIN.from $TRAIN.to $TRAIN.ques $TEST.from $TEST.to $TEST.ques $VALID.from $VALID.to $VALID.ques
