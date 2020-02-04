#!/usr/bin/python3

import dill as pickle

content = None
with open('../data/babi_transformer.bin', 'rb') as f:
    content = pickle.load(f)

print(content['settings'])
print(content['vocab']['txt'].vocab.stoi)
print(content['train'][0].src, content['train'][0].trg)
print(content['test'][0].src, content['test'][0].trg)