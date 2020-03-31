#!/usr/bin/python3

import sys, os

print (sys.argv)

print (sys.argv[0] + " <input_model_name> <output_model_directory>")
#exit

GPT2_DIR=sys.argv[1]
PYTORCH_DUMP_OUTPUT=sys.argv[2]

l = GPT2_DIR.split('/')
print(l)
m = l[-1].split('.')[0]

GPT2_DIR = '/'.join(l[0:-1]) + '/' + m
print(GPT2_DIR)

GPT2_DIR_X = GPT2_DIR.split('/')
GPT2_DIR_X = GPT2_DIR_X[0:-1]
GPT2_DIR_X = '/'.join(GPT2_DIR_X) + '/'
print(GPT2_DIR_X)

os.system("pytorch_pretrained_bert convert_gpt2_checkpoint " + GPT2_DIR + " " +PYTORCH_DUMP_OUTPUT)

if os.path.isfile(GPT2_DIR_X + '/' + 'encoder.json'):
    os.system("cp " + GPT2_DIR_X + '/' + 'encoder.json ' + PYTORCH_DUMP_OUTPUT + '/.')
    os.system('cp ' + GPT2_DIR_X + "/" + 'vocab.bpe ' + PYTORCH_DUMP_OUTPUT + '/.')