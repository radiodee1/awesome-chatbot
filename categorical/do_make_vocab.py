#!/usr/bin/python3

from collections import Counter
import tokenize_weak
from settings import hparams
import sys, os
from operator import itemgetter
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
import numpy as np

vocab_length = hparams['num_vocab_total']
FROM = '../raw/glove.6B.100d.txt'
TO = '../data/embed.txt'
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
    #print(ss[0:10])
    ss.reverse()
    #print(ss[0:10])
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


def load_vocab(filename=None):
    if filename is None:
        filename = train_file[0].replace('train','vocab')
    t = []
    with open(filename, 'r') as r:
        for xx in r:
            t.append(xx.strip())
    # r.close()
    return t
    pass


def prep_glove(vocab_list):
    uniform_low = -0.25
    uniform_high = 0.25
    glove2word2vec(glove_input_file=FROM, word2vec_output_file=TO+'-temp')
    glove_model = KeyedVectors.load_word2vec_format(TO+'-temp', binary=False)
    num = 0
    with open(TO,'w') as f:
        f.write(str(len(vocab_list)) + ' ' + str(hparams['embed_size']) + '\n')
        for i in range(len(vocab_list)):
            word = vocab_list[i]
            if word in glove_model.wv.vocab:
                vec = glove_model.wv[word]
            else:
                vec = np.random.uniform(low=uniform_low, high=uniform_high, size=(int(hparams['embed_size']),))
                num += 1
                print(num ,'blanks')
            vec_out = []
            for j in vec:
                vec_out.append(str(round(j, 5)))
            line = str(word) + ' ' + ' '.join(list(vec_out))
            f.write(line + '\n')
    pass

if __name__ == '__main__':
    train_file = ['../data/train.big.from', '../data/train.big.to']

    if len(sys.argv) > 1:
        train_file = str(sys.argv[1])

    print(train_file)
    #global v
    v = []
    #global v
    if True:
        make_vocab()
        save_vocab()
    if len(v) == 0:
        v = load_vocab()
    prep_glove(v)

    if os.path.isfile(TO+'-temp'):
        os.system('rm ' + TO + '-temp')
        pass
