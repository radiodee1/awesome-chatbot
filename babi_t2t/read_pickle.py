#!/usr/bin/python3

import dill as pickle

content = None
with open('../data/babi_transformer.bin', 'rb') as f:
    content = pickle.load(f)

print(content)