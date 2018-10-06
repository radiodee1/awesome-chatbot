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


if [ $# -eq 1 ] ; then
    echo "pointing to " $1

    #exit()

    if [ -f $BIGTRAINFROM ] && [ -f $BIGTRAINTO ] ; then
        rm $TRAIN.from $TRAIN.to
        ln -s $TRAIN.$1.from $TRAIN.from
        ln -s $TRAIN.$1.to $TRAIN.to
    else
        mv $TRAIN.from $BIGTRAINFROM
        mv $TRAIN.to $BIGTRAINTO
        #rm $TRAIN.from $TRAIN.to
        ln -s $TRAIN.$1.from $TRAIN.from
        ln -s $TRAIN.$1.to $TRAIN.to
    fi

else
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

fi

ls -hal $TRAIN.from $TRAIN.to $TRAIN.ques
