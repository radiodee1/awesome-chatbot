#!/usr/bin/python3

from collections import Counter
import tokenize_weak
from settings import hparams
import sys

vocab_length = hparams['num_vocab_total']
train_file = ''
v = []

def make_vocab():
    wordlist = []
    for filename in train_file:
        with open(filename, 'r') as x:
            xx = x.read()
            for line in xx.split('\n'):
                line = tokenize_weak.format(line)
                y = line.lower().split()
                for word in y:
                    wordlist.append(word)
            pass
    print('values read')
    wordset = set(wordlist)
    c = Counter(wordset)
    l = len(wordset)
    print(l,'length of raw vocab data')
    if l > vocab_length: l = vocab_length
    cc = c.most_common(l)
    print(len(cc), 'length of result list')
    #v = []
    for z in sorted(cc):
        if z[0].lower() not in v: v.append(z[0].lower())
    #vv = list(set(v))
    v.sort()
    #v = vv
    #print(v)

def save_vocab():
    sol = hparams['sol']
    eol = hparams['eol']
    unk = hparams['unk']
    name = train_file.replace('train', 'vocab')
    if name == train_file:
        name += '.voc.txt'
    with open(name, 'w') as x:
        x.write(unk+'\n'+ sol+'\n'+eol+'\n')
        for z in v:
            x.write(z + "\n")
        print('values written')
    pass

if __name__ == '__main__':
    train_file = ['../data/train.big.from', '../data/train.big.to']

    if len(sys.argv) > 1:
        train_file = str(sys.argv[1])

    print(train_file)
    #global v
    v = []
    #global v
    make_vocab()
    save_vocab()
