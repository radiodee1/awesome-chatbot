#!/usr/bin/python3

from collections import Counter
import tokenize_weak
from settings import hparams
import sys
from operator import itemgetter

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

    #wordset = set(wordlist)
    c = Counter(wordlist)
    l = len(wordlist)
    print(l,'length of raw vocab data')
    if l > vocab_length: l = vocab_length
    cc = c.most_common()

    #cc = wordlist
    print(len(cc), 'length of result list')
    #v = []
    num = 0
    ss = sorted(cc, key=itemgetter(1))
    print(ss[0:10])
    ss.reverse()
    print(ss[0:10])
    for z in ss: # sorted(cc, key= lambda word: word[1]):
        if z[0].lower() not in v and num < vocab_length: v.append(z[0].lower())
        num +=1
    #vv = list(set(v))
    v.sort()
    #v = vv
    #print(v)

def save_vocab():
    ''' remember to leave zero spot blank '''
    sol = hparams['sol']
    eol = hparams['eol']
    unk = hparams['unk']
    name = train_file[0].replace('train', 'vocab')
    if name == train_file[0]:
        name += '.voc.txt'
    with open(name, 'w') as x:
        x.write('\n'+unk+'\n'+ sol+'\n'+eol+'\n')
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
