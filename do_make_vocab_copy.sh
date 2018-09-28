#!/bin/bash


cp data/vocab.big.txt saved/saved_vocab.big.txt
cp data/vocab.babi.txt saved/saved_vocab.babi.txt
#cp data/vocab.to saved/saved_vocab.to
cp model/settings.py saved/saved_settings.py.txt
cp data/embed.txt saved/saved_embed.txt

cd saved
zip vocab.zip saved*.txt
rm saved*.txt

mv vocab.zip ../
