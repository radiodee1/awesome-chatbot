#!/usr/bin/env bash

cd raw
wget https://download.pytorch.org/tutorial/data.zip

unzip data.zip

cd data
cp eng-fra.txt ..