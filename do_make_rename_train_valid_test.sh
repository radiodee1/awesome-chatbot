#!/bin/bash


cd data

BIGTRAINFROM=train.big.from
BIGTRAINTO=train.big.to
BIGTRAINQUES=train.big.ques

BIGTRAINHIST=train.big.hist

BIGVALFROM=valid.big.from
BIGVALTO=valid.big.to
BIGVALQUES=valid.big.ques

BIGVALHIST=valid.big.hist

BIGTESTFROM=test.big.from
BIGTESTTO=test.big.to
BIGTESTQUES=test.big.ques

BIGTESTHIST=test.big.hist

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

BIGTRAINHIST=train.$1.hist

BIGVALFROM=valid.$1.from
BIGVALTO=valid.$1.to
BIGVALQUES=valid.$1.ques

BIGVALHIST=valid.$1.hist

BIGTESTFROM=test.$1.from
BIGTESTTO=test.$1.to
BIGTESTQUES=test.$1.ques

BIGTESTHIST=test.$1.hist

fi
#else
    echo ""
    echo "changing link to large initial file! "

    if [ -f $BIGTRAINFROM ] && [ -f $BIGTRAINTO ] && [ -f $BIGTRAINQUES ] ; then
        rm $TRAIN.from $TRAIN.to $TRAIN.ques $TRAIN.hist
        ln -s $BIGTRAINFROM $TRAIN.from
        ln -s $BIGTRAINTO $TRAIN.to
        ln -s $BIGTRAINQUES $TRAIN.ques
        ln -s $BIGTRAINHIST $TRAIN.hist
        rm $VALID.from $VALID.to $VALID.ques $VALID.hist
        ln -s $BIGVALFROM $VALID.from
        ln -s $BIGVALTO $VALID.to
        ln -s $BIGVALQUES $VALID.ques
        ln -s $BIGVALHIST $VALID.hist
        rm $TEST.from $TEST.to $TEST.ques $TEST.hist
        ln -s $BIGTESTFROM $TEST.from
        ln -s $BIGTESTTO $TEST.to
        ln -s $BIGTESTQUES $TEST.ques
        ln -s $BIGTESTHIST $TEST.hist


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
ls -hal $TRAIN.from $TRAIN.to $TRAIN.ques $TRAIN.hist $TEST.from $TEST.to $TEST.ques $TEST.hist $VALID.from $VALID.to $VALID.ques $VALID.hist
