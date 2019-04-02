#!/bin/bash

echo "enter the glue folder name as the first parameter"
echo $#
echo $@
echo $1


zip -r bert_glue_$1.zip saved/glue_saved/$1/*

#echo "use this script with any savable files as the parameter."