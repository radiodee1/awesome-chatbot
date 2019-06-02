#!/usr/bin/python3

import sys, os

print (sys.argv)

#exit

GPT2_DIR=sys.argv[1]
PYTORCH_DUMP_OUTPUT=sys.argv[2]

l = GPT2_DIR.split('/')
print(l)
m = l[-1].split('.')[0]

GPT2_DIR = '/'.join(l[0:-1]) + '/' + m
print(GPT2_DIR)

os.system("pytorch_pretrained_bert convert_gpt2_checkpoint " + GPT2_DIR + " " +PYTORCH_DUMP_OUTPUT)

print (sys.argv[0] + " input_model_name output_model_directory")