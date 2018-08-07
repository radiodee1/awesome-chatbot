#!/usr/bin/python3

from collections import Counter
import tokenize_weak
from settings import hparams
import sys, os
from operator import itemgetter
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import argparse

embed_size = hparams['embed_size']
vocab_length = hparams['num_vocab_total']
FROM = '../raw/glove.6B.' + str(embed_size) +'d.txt' # 50, 100, 200, 300
TO = '../data/embed.txt'
train_file = ''

hard_coded_list = [
    "i'm",
    "you're",
    "we're",
    "they're",
    "he's",
    "she's",
    "it's",
    "that's",
    "don't",
    "can't",
    "shouldn't",
    "didn't",
    "shan't",
    "wouldn't",
    "we've",
    "i've",
    "they've",
    "you've",
    "here's",
    "there's",
    "i'd",
    "you'd",
    "they'd",
    "we'd",
    "he'd",
    "she'd"
    #"won't"
]

directions = [
    'nn',
    'ss',
    'ee',
    'ww',
    'ne',
    'nw',
    'se',
    'sw',
    'en',
    'es',
    'wn',
    'ws',
    'n',
    's',
    'w',
    'e'
]

v = []

def make_vocab(train_file, order=False, read_glove=False, contractions=False):
    global v
    wordlist = []

    if contractions:
        wordlist.extend(hard_coded_list)
        wordlist.extend(directions)

    for filename in train_file:
        with open(filename, 'r') as x:
            xx = x.read()
            for line in xx.split('\n'):
                line = tokenize_weak.format(line)
                y = line.lower().split()
                for word in y:
                    wordlist.append(word)
            pass
    print('values read from text file.')

    if read_glove:
        with open(FROM, 'r') as x:
            xx = x.read()
            for line in xx.split('\n'):
                l = line.split(' ')
                #print( len(l))
                if len(l) > 2:
                    wordlist.append(l[0].strip())
        pass

        print('values read from glove file.')

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
    if order:
        ss = sorted(cc, key=itemgetter(1))
        #print(ss[0:10])
        ss.reverse()
    else:
        ss = cc

    #print(ss[0:10])
    for z in ss: # sorted(cc, key= lambda word: word[1]):
        if z[0].lower() not in v and num < vocab_length: v.append(z[0].lower())
        num +=1
    #vv = list(set(v))
    v.sort()

    vv = [hparams['unk'], hparams['sol'], hparams['eol']]
    for z in v:
        if len(vv) < vocab_length and z not in vv: vv.append(z)
    v = vv
    #print(v)
    return v

def save_vocab(v, babi=False):
    ''' remember to leave 3 spots blank '''
    sol = hparams['sol']
    eol = hparams['eol']
    unk = hparams['unk']
    #name = train_file[0].replace('train', 'vocab')
    name = hparams['data_dir'] + hparams['vocab_name']

    if name == train_file[0]:
        name += '.voc.txt'

    if babi:
        name = name.replace('big', hparams['babi_name'])

    with open(name, 'w') as x:
        #x.write(unk+'\n'+ sol+'\n'+eol+'\n')
        for z in range(len(v)):
            if z < int(hparams['num_vocab_total']) - 3: ## magic num for hparams tokens
                x.write(v[z] + "\n")
        print('values written')
    pass


def load_vocab(filename=None):
    if filename is None:
        #filename = train_file[0].replace('train','vocab')
        filename = hparams['data_dir'] + hparams['vocab_name']

    t = []
    with open(filename, 'r') as r:
        for xx in r:
            t.append(xx.strip())
    # r.close()
    return t
    pass


def prep_glove(vocab_list, w2v=False):
    uniform_low = -1.0
    uniform_high = 1.0
    if not w2v:
        glove2word2vec(glove_input_file=FROM, word2vec_output_file=TO+'-temp')
        glove_model = KeyedVectors.load_word2vec_format(TO+'-temp', binary=False)
    else:
        glove_model = KeyedVectors.load_word2vec_format('../raw/GoogleNews-vectors-negative300.bin', binary=True)
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
                print(num ,'blanks',word)
            vec_out = []
            for j in vec:
                vec_out.append(str(round(j, 5)))
            line = str(word) + ' ' + ' '.join(list(vec_out))
            f.write(line + '\n')
    pass

if __name__ == '__main__':
    print('In order to get a vocab file with all babi vocab words at once, you must temporarily build a question set\n'
          'with ALL questions at once. Later you can rebuild the question set for just one task after your vocab\n'
          'file is complete.')

    parser = argparse.ArgumentParser(description='Make vocab file.')
    parser.add_argument('--basefile',metavar='FILE', type=str, help='Base file from training for vocab output')
    parser.add_argument('--babi', help='Flag for babi input.', action='store_true')
    parser.add_argument('--all-glove', help='Load all words from the glove set.', action='store_true')
    parser.add_argument('--w2v', help='replace all glove data with data obtained from w2v downloads.', action='store_true')
    parser.add_argument('--load-embed-size', help='Override settings embed-size hparam.')
    parser.add_argument('--order', help='put in alpha order.', action='store_true')
    parser.add_argument('--contractions', help='add some contractions to the vocab that are not present in glove.',
                        action='store_true')
    args = parser.parse_args()
    args = vars(args)
    print(args)
    #exit()
    train_file = [] #'../data/train.big.from'] # , '../data/train.big.to']
    order = False
    read_glove = False
    use_w2v = False
    use_contractions = False

    if args['babi']:
        train_file = []
        #args['babi'] = True
    elif args['basefile'] is not None :
        train_file = [args['basefile']]

    if args['load_embed_size'] is not None:
        hparams['embed_size'] = int(args['load_embed_size'])

    if args['order'] is True:
        order = True
    else:
        order = False

    if args['all_glove'] is True:
        read_glove = True
    else:
        read_glove = False

    if args['w2v'] is True:
        use_w2v = True
        hparams['embed_size'] = 300
    else:
        use_w2v = False

    if args['contractions'] is True: use_contractions = True

    babi_file = hparams['data_dir'] + hparams['train_name'] + '.' + hparams['babi_name'] + '.' + hparams['src_ending']
    babi_file2 = hparams['data_dir'] + hparams['train_name'] + '.' + hparams['babi_name'] + '.' + hparams['tgt_ending']
    babi_file3 = hparams['data_dir'] + hparams['train_name'] + '.' + hparams['babi_name'] + '.' + hparams['question_ending']

    if args['babi']: # or args['babi_only']:
        train_file.append(babi_file)
        train_file.append(babi_file2)
        train_file.append(babi_file3)
    print(train_file)

    embed_size = hparams['embed_size']
    vocab_length = hparams['num_vocab_total']
    FROM = '../raw/glove.6B.' + str(embed_size) + 'd.txt'  # 50, 100, 200, 300
    TO = '../data/embed.txt'

    v = []

    if True:
        v = make_vocab(train_file, order=order, read_glove=read_glove, contractions=use_contractions)
        save_vocab(v, args['babi'])
    if len(v) == 0:
        filename = hparams['data_dir'] + hparams['vocab_name']
        if args['babi'] == True: filename = filename.replace('big', hparams['babi_name'])
        v = load_vocab(filename)

    if hparams['embed_size'] is not None and hparams['embed_size'] != 0:
        prep_glove(v, w2v=use_w2v)

        if os.path.isfile(TO+'-temp'):
            os.system('rm ' + TO + '-temp')
            pass
    else:
        print('glove vectors disabled in settings.py')
        print('set embed_size to usable value: 50, 100, 200, 300, None for none.')
        print('note: glove vectors dont have contractions')