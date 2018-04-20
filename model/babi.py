#!/usr/bin/python3.6
from __future__ import unicode_literals, print_function, division

import sys
#sys.path.append('..')
from io import open
import unicodedata
import string
import re
import random
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import time
import math
import argparse
from settings import hparams
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


'''
Some code was originally written by Austin Jacobson. This refers specifically 
to the Encoder and Decoder classes and came from:

https://github.com/A-Jacobson/minimal-nmt

MIT License

Copyright (c) 2018 Austin Jacobson

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
THE SOFTWARE.

Some code is originally written by Sean Robertson. This code includes 
some of the text processing code and early versions of the Decoder and Encoder 
classes. This can be found at the following site:

http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#sphx-glr-intermediate-seq2seq-translation-tutorial-py

The MIT License (MIT)

Copyright (c) 2017 Sean Robertson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

'''

use_cuda = torch.cuda.is_available()
SOS_token = 0
EOS_token = 1
MAX_LENGTH = hparams['tokens_per_sentence']


teacher_forcing_ratio = hparams['teacher_forcing_ratio'] #0.5

class EpisodicAttn(nn.Module):

    def __init__(self,  hidden_size, a_list_size=7):
        super(EpisodicAttn, self).__init__()

        self.hidden_size = hidden_size

        self.a_list_size = a_list_size

        self.W_b = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
        self.W_1 = nn.Parameter(torch.FloatTensor(hidden_size, self.a_list_size * hidden_size))
        self.W_2 = nn.Parameter(torch.FloatTensor(1, hidden_size))
        self.b_1 = nn.Parameter(torch.FloatTensor(hidden_size,))
        self.b_2 = nn.Parameter(torch.FloatTensor(1,))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self,concat_list):

        assert len(concat_list) == self.a_list_size
        ''' attention list '''
        self.c_list_z = torch.cat(concat_list)
        print(self.c_list_z.size(), 'list')
        self.l_1 = torch.mm(self.W_1, self.c_list_z.view(-1, self.hidden_size)) + self.b_1
        self.l_1 = torch.tanh(self.l_1)
        self.l_2 = torch.mm(self.W_2, self.l_1) + self.b_2
        self.G = torch.sigmoid(self.l_2)[0]

        return  self.G


class LuongAttention(nn.Module):
    """
    LuongAttention from Effective Approaches to Attention-based Neural Machine Translation
    https://arxiv.org/pdf/1508.04025.pdf
    """

    def __init__(self, dim):
        super(LuongAttention, self).__init__()
        self.W = nn.Linear(dim, dim, bias=False)

    def score(self, decoder_hidden, encoder_out):
        # linear transform encoder out (seq, batch, dim)
        encoder_out = self.W(encoder_out)
        # (batch, seq, dim) | (2, 15, 50)
        encoder_out = encoder_out.permute(1, 0, 2)
        # (2, 15, 50) @ (2, 50, 1)
        return encoder_out @ decoder_hidden.permute(1, 2, 0)

    def forward(self, decoder_hidden, encoder_out):
        energies = self.score(decoder_hidden, encoder_out)
        mask = F.softmax(energies, dim=1)  # batch, seq, 1
        context = encoder_out.permute(
            1, 2, 0) @ mask  # (2, 50, 15) @ (2, 15, 1)
        context = context.permute(2, 0, 1)  # (seq, batch, dim)
        mask = mask.permute(2, 0, 1)  # (seq2, batch, seq1)
        return context, mask

class MemRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MemRNN, self).__init__()
        self.hidden_size = hidden_size
        #self.embedding = nn.Embedding(input_size, hidden_size)
        self.bi_gru = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=False,bidirectional=False)


    def forward(self, input, hidden=None):
        #embedded = self.embedding(input).view(1, 1, -1)
        #output = embedded
        output, hidden = self.gru(input,hidden)
        #bi_output = (bi_output[:, :, :self.hidden_size] +
        #               bi_output[:, :, self.hidden_size:])
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class Encoder(nn.Module):
    def __init__(self, source_vocab_size, embed_dim, hidden_dim,
                 n_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embed = nn.Embedding(source_vocab_size, embed_dim, padding_idx=1)
        self.gru = nn.GRU(embed_dim, hidden_dim, n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, source, hidden=None):
        embedded = self.embed(source)  # (batch_size, seq_len, embed_dim)
        encoder_out, encoder_hidden = self.gru(
            embedded, hidden)  # (seq_len, batch, hidden_dim*2)
        # sum bidirectional outputs, the other option is to retain concat features
        encoder_out = (encoder_out[:, :, :self.hidden_dim] +
                       encoder_out[:, :, self.hidden_dim:])
        return encoder_out, encoder_hidden

class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.embed = nn.Embedding(target_vocab_size, embed_dim, padding_idx=1)
        self.attention = LuongAttention(hidden_dim)
        self.gru = nn.GRU(embed_dim + hidden_dim, hidden_dim, n_layers,
                          dropout=dropout)
        self.out = nn.Linear(hidden_dim * 2, target_vocab_size)

    def forward(self, output, encoder_out, decoder_hidden):
        """
        decodes one output frame
        """
        embedded = self.embed(output)  # (1, batch, embed_dim)
        if self.n_layers == 1:
            context, mask = self.attention(decoder_hidden, encoder_out)  # 1, 1, 50 (seq, batch, hidden_dim)
        else:
           context, mask = self.attention(decoder_hidden[:-1], encoder_out)  # 1, 1, 50 (seq, batch, hidden_dim)

        rnn_output, decoder_hidden = self.gru(torch.cat([embedded, context], dim=2),
                                              decoder_hidden)
        output = self.out(torch.cat([rnn_output, context], 2))
        return output, decoder_hidden, mask

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: hparams['sol'], 1: hparams['eol']}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


################################
class NMT:
    def __init__(self):
        self.model_1_enc = None
        self.model_2_dec = None
        self.model_3_mem = None
        self.model_4_att = None
        self.opt_1 = None
        self.opt_2 = None
        self.best_loss = None
        self.long_term_loss = None
        self.tag = ''

        self.input_lang = None
        self.output_lang = None
        self.question_lang = None
        self.vocab_lang = None

        self.print_every = hparams['steps_to_stats']
        self.epochs = hparams['epochs']
        self.hidden_size = hparams['units']
        self.memory_hops = 5
        self.start = 0

        self.train_fr = None
        self.train_to = None
        self.train_ques = None
        self.pairs = []

        self.do_train = False
        self.do_infer = False
        self.do_review = False
        self.do_train_long = False
        self.do_interactive = False
        self.do_convert = False
        self.do_plot = False
        self.do_load_babi = False

        self.printable = ''

        self.input_var = None     # for input
        self.q_var = None         # for question
        self.answer_var = None    # for answer
        self.q_q = None           # extra question
        self.inp_c = None         # extra input
        self.last_mem = None      # output of mem unit
        self.prediction = None    # final single word prediction

        # part of output
        self.W_a = nn.Parameter(torch.FloatTensor(hparams['num_vocab_total'], self.hidden_size))


        parser = argparse.ArgumentParser(description='Train some NMT values.')
        parser.add_argument('--mode', help='mode of operation. (train, infer, review, long, interactive, plot)')
        parser.add_argument('--printable', help='a string to print during training for identification.')
        parser.add_argument('--basename', help='base filename to use if it is different from settings file.')
        parser.add_argument('--autoencode', help='enable auto encode from the command line with a ratio.')
        parser.add_argument('--train-all', help='(broken) enable training of the embeddings layer from the command line',
                            action='store_true')
        parser.add_argument('--convert-weights',help='convert weights', action='store_true')
        parser.add_argument('--load-babi', help='Load three babi input files instead of chatbot data',
                            action='store_true')
        self.args = parser.parse_args()
        self.args = vars(self.args)
        # print(self.args)

        if self.args['printable'] is not None:
            self.printable = str(self.args['printable'])
        if self.args['mode'] == 'train': self.do_train = True
        if self.args['mode'] == 'infer': self.do_infer = True
        if self.args['mode'] == 'review': self.do_review = True
        if self.args['mode'] == 'long': self.do_train_long = True
        if self.args['mode'] == 'interactive': self.do_interactive = True
        if self.args['mode'] == 'plot':
            self.do_review = True
            self.do_plot = True
        if self.args['basename'] is not None:
            hparams['base_filename'] = self.args['basename']
            print(hparams['base_filename'], 'set name')
        if self.args['autoencode'] is not  None and float(self.args['autoencode']) > 0.0:
            hparams['autoencode'] = float(self.args['autoencode'])
        else: hparams['autoencode'] = 0.0
        if self.args['train_all'] == True:
            # hparams['embed_train'] = True
            self.trainable = True
        else:
            self.trainable = False
        if self.args['convert_weights'] == True: self.do_convert = True
        if self.args['load_babi'] == True: self.do_load_babi = True


    def task_normal_train(self):
        self.train_fr = hparams['data_dir'] + hparams['train_name'] + '.' + hparams['src_ending']
        self.train_to = hparams['data_dir'] + hparams['train_name'] + '.' + hparams['tgt_ending']
        pass

    def task_review_set(self):
        self.train_fr = hparams['data_dir'] + hparams['test_name'] + '.' + hparams['src_ending']
        self.train_to = hparams['data_dir'] + hparams['test_name'] + '.' + hparams['tgt_ending']
        pass

    def task_babi_files(self):
        self.train_fr = hparams['data_dir'] + hparams['train_name'] + '.' + hparams['babi_name'] + '.' + hparams['src_ending']
        self.train_to = hparams['data_dir'] + hparams['train_name'] + '.' + hparams['babi_name'] + '.' + hparams['tgt_ending']
        self.train_ques = hparams['data_dir'] + hparams['train_name'] + '.' + hparams['babi_name'] + '.' + hparams['question_ending']
        pass

    def task_review_weights(self, pairs, stop_at_fail=False):
        plot_losses = []
        num = 0 # hparams['base_file_num']
        for i in range(100):
            local_filename = hparams['save_dir'] + hparams['base_filename'] + '.'+ str(num) + '.pth.tar'
            if os.path.isfile(local_filename):
                ''' load weights '''

                print('==============================')
                print(str(i)+'.')
                print('here:',local_filename)
                self.load_checkpoint(local_filename)
                print('loss', self.best_loss)
                plot_losses.append(self.best_loss)
                print('tag', self.tag)
                choice = random.choice(pairs)
                print(choice[0])
                out, _ =self.evaluate(None,None,choice[0])
                print(out)
            else:
                plot_losses.append(self.best_loss)
                if stop_at_fail: break
            num = 10 * self.print_every * i
        pass
        if self.do_plot:
            plt.figure()
            fig, ax = plt.subplots()
            loc = ticker.MultipleLocator(base=2)
            ax.yaxis.set_major_locator(loc)
            plt.plot(plot_losses)
            plt.show()


    def task_train_epochs(self,num=0):
        lr = hparams['learning_rate']
        if num == 0:
            num = hparams['epochs']
        for i in range(num):
            self.printable = ' epoch #' + str(i+1)
            self.trainIters(None, None, len(self.pairs), print_every=self.print_every, learning_rate=lr)
            self.start = 0
            self.save_checkpoint(num=len(self.pairs))
        self.input_lang, self.output_lang, self.pairs = self.prepareData(self.train_fr, self.train_to, reverse=False, omit_unk=True)

        pass

    def task_interactive(self):

        print('-------------------')
        while True:
            line = input("> ")
            line = tokenize_weak.format(line)
            print(line)
            out , _ =self.evaluate(None, None, line)
            print(out)

    def task_convert(self):
        hparams['base_filename'] += '.small'
        self.save_checkpoint(is_best=False,converted=True)

    ################################################


    def open_sentences(self, filename):
        t_yyy = []
        with open(filename, 'r') as r:
            for xx in r:
                t_yyy.append(xx)
        return t_yyy

    def asMinutes(self,s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)


    def timeSince(self,since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.asMinutes(s), self.asMinutes(rs))

    def unicodeToAscii(self,s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )


    def normalizeString(self,s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def readLangs(self,lang1, lang2,lang3=None, reverse=False, load_vocab_file=None, babi_ending=False):
        print("Reading lines...")
        self.pairs = []
        if not self.do_interactive:

            l_in = self.open_sentences(hparams['data_dir'] + lang1)
            l_out = self.open_sentences(hparams['data_dir'] + lang2)
            if lang3 is not None:
                l_ques = self.open_sentences(hparams['data_dir'] + lang3)

            #pairs = []
            for i in range(len(l_in)):
                #print(i)
                if i < len(l_out):
                    if not babi_ending:
                        line = [ l_in[i].strip('\n'), l_out[i].strip('\n') ]
                    else:
                        lin = l_in[i].strip('\n')

                        if lang3 is not None:
                            lques = l_ques[i].strip('\n')
                        else:
                            lques = l_in[i].strip('\n')
                            lques = lques.split(' ')
                            if len(lques) > MAX_LENGTH:
                                lques = lques[: - MAX_LENGTH]
                            lques = ' '.join(lques)

                        lans = l_out[i].strip('\n')
                        line = [ lin, lques , lans]
                    self.pairs.append(line)

        # Reverse pairs, make Lang instances
        if load_vocab_file is not None:
            self.vocab_lang = Lang(load_vocab_file)
            pass

        if reverse:
            self.pairs = [list(reversed(p)) for p in self.pairs]
            self.input_lang = Lang(lang2)
            self.output_lang = Lang(lang1)
        else:
            self.input_lang = Lang(lang1)
            self.output_lang = Lang(lang2)

        if hparams['autoencode'] == 1.0:
            self.pairs = [ [p[0], p[0], p[0]] for p in self.pairs]
            self.output_lang = self.input_lang

        return self.input_lang, self.output_lang, self.pairs

    def filterPair(self,p):
        return (len(p[0].split(' ')) < MAX_LENGTH and \
            len(p[1].split(' ')) < MAX_LENGTH)  or True #\



    def filterPairs(self,pairs):
        return [pair for pair in pairs if self.filterPair(pair)]



    def prepareData(self,lang1, lang2, reverse=False, omit_unk=False):
        if hparams['vocab_name'] is not None:
            v_name = hparams['data_dir'] + hparams['vocab_name']
            v_name = v_name.replace('big', hparams['babi_name'])
        else:
            v_name = None

        if not self.do_load_babi:
            self.input_lang, self.output_lang, self.pairs = self.readLangs(lang1, lang2, lang3=None,
                                                                           reverse=reverse,
                                                                           load_vocab_file=v_name)
            lang3 = None
        else:
            self.input_lang, self.output_lang, self.pairs = self.readLangs(lang1, lang2, self.train_ques,
                                                                           reverse=False,
                                                                           babi_ending=True,
                                                                           load_vocab_file=v_name)
            lang3 = self.train_ques
        print("Read %s sentence pairs" % len(self.pairs))
        self.pairs = self.filterPairs(self.pairs)
        print("Trimmed to %s sentence pairs" % len(self.pairs))
        print("Counting words...")
        if v_name is not None:
            v = self.open_sentences(self.vocab_lang.name)
            for word in v:
                self.vocab_lang.addSentence(word.strip())
                #print(word)
            self.input_lang = self.vocab_lang
            self.output_lang = self.vocab_lang

            new_pairs = []
            for p in range(len(self.pairs)):
                #print(self.pairs[p])
                a = []
                b = []
                c = []
                for word in self.pairs[p][0].split(' '):
                    if word in self.vocab_lang.word2index:
                        a.append(word)
                    elif not omit_unk:
                        a.append(hparams['unk'])
                for word in self.pairs[p][1].split(' '):
                    if word in self.vocab_lang.word2index:
                        b.append(word)
                    elif not omit_unk:
                        b.append(hparams['unk'])
                pairs = [' '.join(a), ' '.join(b)]
                if lang3 is not None:
                    for word in self.pairs[p][2].split(' '):
                        if word in self.vocab_lang.word2index:
                            c.append(word)
                        elif not omit_unk:
                            c.append(hparams['unk'])
                    pairs.append( ' '.join(c) )
                new_pairs.append(pairs)
            self.pairs = new_pairs

        else:
            for pair in self.pairs:
                self.input_lang.addSentence(pair[0])
                self.output_lang.addSentence(pair[1])

        print("Counted words:")
        print(self.input_lang.name, self.input_lang.n_words)
        print(self.output_lang.name, self.output_lang.n_words)
        return self.input_lang, self.output_lang, self.pairs


    def indexesFromSentence(self,lang, sentence):
        s = sentence.split(' ')
        sent = []
        for word in s:
            if word == hparams['eol']: word = EOS_token
            elif word == hparams['sol']: word = SOS_token
            else: word = lang.word2index[word]
            sent.append(word)
        if len(sent) >= MAX_LENGTH:
            sent = sent[:MAX_LENGTH]
            sent[-1] = EOS_token
            #print(sent,'<<<<')
        return sent

        #return [lang.word2index[word] for word in sentence.split(' ')]


    def variableFromSentence(self,lang, sentence, add_eol=False):
        indexes = self.indexesFromSentence(lang, sentence)
        if add_eol: indexes.append(EOS_token)
        result = Variable(torch.LongTensor(indexes).view(-1, 1))
        if use_cuda:
            return result.cuda()
        else:
            return result


    def variablesFromPair(self,pair):
        input_variable = self.variableFromSentence(self.input_lang, pair[0])
        question_variable = self.variableFromSentence(self.output_lang, pair[1])
        if len(pair) > 2:
            target_variable = self.variableFromSentence(self.output_lang, pair[2])
        else:
            return (input_variable, question_variable)
        return (input_variable,question_variable, target_variable)


    def make_state(self, converted=False):
        if not converted:
            z = [
                {
                    'epoch':0,
                    'start': self.start,
                    'arch': None,
                    'state_dict': self.model_1_enc.state_dict(),
                    'best_prec1': None,
                    'optimizer': self.opt_1.state_dict(),
                    'best_loss': self.best_loss,
                    'long_term_loss' : self.long_term_loss,
                    'tag': self.tag
                },
                {
                    'epoch':0,
                    'start': self.start,
                    'arch':None,
                    'state_dict':self.model_2_dec.state_dict(),
                    'best_prec1':None,
                    'optimizer': self.opt_2.state_dict(),
                    'best_loss': self.best_loss,
                    'long_term_loss': self.long_term_loss,
                    'tag': self.tag
                }
            ]
        else:
            z = [
                {
                    'epoch': 0,
                    'start': self.start,
                    'arch': None,
                    'state_dict': self.model_1_enc.state_dict(),
                    'best_prec1': None,
                    'optimizer': None , # self.opt_1.state_dict(),
                    'best_loss': self.best_loss,
                    'long_term_loss': self.long_term_loss,
                    'tag': self.tag
                },
                {
                    'epoch': 0,
                    'start': self.start,
                    'arch': None,
                    'state_dict': self.model_2_dec.state_dict(),
                    'best_prec1': None,
                    'optimizer': None, # self.opt_2.state_dict(),
                    'best_loss': self.best_loss,
                    'long_term_loss': self.long_term_loss,
                    'tag': self.tag
                }
            ]
        #print(z)
        return z
        pass

    def save_checkpoint(self, state=None, is_best=True, num=0, converted=False, extra=''):
        if state is None:
            state = self.make_state(converted=converted)
            if converted: print(converted, 'is converted.')
        basename = hparams['save_dir'] + hparams['base_filename']
        torch.save(state, basename + extra + '.' + str(num)+ '.pth.tar')
        if is_best:
            os.system('cp '+ basename + extra +  '.' + str(num) + '.pth.tar' + ' '  +
                      basename + '.best.pth.tar')

    def load_checkpoint(self, filename=None):
        if True:
            basename = hparams['save_dir'] + hparams['base_filename'] + '.best.pth.tar'
            if filename is not None: basename = filename
            if os.path.isfile(basename):
                print("loading checkpoint '{}'".format(basename))
                checkpoint = torch.load(basename)
                #print(checkpoint)
                try:
                    bl = checkpoint[0]['best_loss']
                    if self.best_loss is None or self.best_loss == 0 or bl < self.best_loss or self.do_review: self.best_loss = bl
                except:
                    print('no best loss saved with checkpoint')
                    pass
                try:
                    l = checkpoint[0]['long_term_loss']
                    self.long_term_loss = l
                except:
                    print('no long term loss saved with checkpoint')
                try:
                    self.start = checkpoint[0]['start']
                    if self.start >= len(self.pairs): self.start = 0
                except:
                    print('no start saved with checkpoint')
                    pass
                try:
                    self.tag = checkpoint[0]['tag']
                except:
                    print('no tag saved with checkpoint')
                    self.tag = ''
                    pass
                if hparams['zero_start'] is True:
                    self.start = 0
                self.model_1_enc.load_state_dict(checkpoint[0]['state_dict'])
                if self.opt_1 is not None:
                    self.opt_1.load_state_dict(checkpoint[0]['optimizer'])

                self.model_2_dec.load_state_dict(checkpoint[1]['state_dict'])
                if self.opt_2 is not None:
                    self.opt_2.load_state_dict(checkpoint[1]['optimizer'])

                print("loaded checkpoint '"+ basename + "' ")
            else:
                print("no checkpoint found at '"+ basename + "'")

    def _mod_hidden(self, encoder_hidden):
        z = torch.cat((encoder_hidden,), 2)[0].view(1, 1, self.hidden_size )
        #print(z.size())
        return z

    def _shorten(self, sentence):
        # assume input is list already
        # get longest mutter possible!!
        # trim start!!!
        j = 0
        k = 0
        while k < len(sentence):
            if sentence[j] == hparams['sol'] :
                j += 1
            else:
                break
            k += 1
            pass
        sentence = sentence[j:]
        #print('shorten:', sentence)

        # trim end!!!
        last = ''
        saved = [hparams['eol']]
        punct = ['.','!','?']
        out = []
        for i in sentence:
            if i in saved and last != i: break
            if i != hparams['sol'] and last != i:
                out.append(i)
            if i in punct: break
            if last != i :
                saved.append(i)
            last = i
        return ' '.join(out)


    def new_input_module(self, input_variable, question_variable):

        out1, hidden1 = self.model_1_enc(input_variable)
        self.inp_c = out1
        out2, hidden2 = self.model_1_enc(question_variable)
        self.q_q = out2
        return out1, hidden1
        pass

    def new_episodic_module(self):
        if self.q_q is not None:
            print(self.q_q.size(),'qq')
            memory = [self.q_q]
            for iter in range(1, self.memory_hops+1):
                current_episode = self.new_episode_big_step(memory[iter - 1])
                out,  _ = self.model_3_mem(memory[iter - 1], current_episode)
                memory.append(out)
            self.last_mem = memory[-1]
        pass

    def new_episode_big_step(self, mem):
        g_record = []
        sequences = self.inp_c
        print(sequences.size(),'seq')
        for i in range(len(sequences)):
            g = self.new_attention_step(sequences[i],None,mem,self.q_q)
            g_record.append(g)
            ## do something with g!!
            pass
        sequences = self.inp_c
        for i in range(len(sequences)):
            e = self.new_episode_small_step(sequences[i], g_record[i], None) ## something goes here!!
            pass
        return e
        pass

    def new_episode_small_step(self, ct, g, prev_h):
        gru, _ = self.model_3_mem(ct, prev_h)
        h = g * gru + (1 - g) * prev_h
        return h
        pass

    def new_attention_step(self, ct, prev_g, mem, q_q):
        ct = ct.view(1,1,-1)
        print(ct.size(), mem.size(), q_q.size(),'all')
        concat_list = [ct, mem, q_q, ct * q_q, ct * mem, torch.abs(ct - q_q), torch.abs(ct - mem)]
        return self.model_4_att(concat_list)

    def new_answer_feed_forward(self):
        # do something with last_mem
        y = torch.mm(self.W_a, self.last_mem)
        y = nn.Softmax(y)
        return y
        pass

    def new_answer_module(self, target_variable, encoder_hidden, criterion, max_length=MAX_LENGTH):
        targets = target_variable  # input_variable
        outputs = []
        masks = []
        outputs_index = []
        decoder_hidden = encoder_hidden[- self.model_2_dec.n_layers:]  # take what we need from encoder
        output = targets[0].unsqueeze(0)  # start token
        print(output)

        self.prediction = self.new_answer_feed_forward()

        is_teacher = random.random() < teacher_forcing_ratio
        raw = 0
        loss = 0

        for t in range(1, max_length - 1):
            # print(t,'t', decoder_hidden.size())

            if len(targets) > 1 and t == 2: output = self.prediction

            output, decoder_hidden, mask = self.model_2_dec(output, self.inp_c, decoder_hidden)
            outputs.append(output)
            masks.append(mask.data)

            if t < len(target_variable):
                loss += criterion(output.view(1, -1), target_variable[t])
            else:
                loss += criterion(output.view(1, -1), Variable(torch.LongTensor([0])))

            # raw = output[:]
            output = Variable(output.data.max(dim=2)[1])
            # raw.append(output)
            if int(output.data[0].int()) == EOS_token:
                # print('eos token',t)
                break

            # teacher forcing
            if is_teacher and t < targets.size()[0]:
                # print(output,'out')
                output = targets[t].unsqueeze(0)
                # print(self.output_lang.index2word[int(output)])

        return output, masks, loss
        pass

    def train(self,input_variable, target_variable,question_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_output , encoder_hidden = self.new_input_module(input_variable, question_variable)

        self.new_episodic_module()

        outputs, masks, loss = self.new_answer_module(target_variable, encoder_hidden, criterion)


        '''
        encoder_output, encoder_hidden = encoder(input_variable)
        
        

        targets = target_variable #input_variable
        outputs = []
        masks = []
        outputs_index = []
        decoder_hidden = encoder_hidden[-decoder.n_layers:]  # take what we need from encoder
        output = targets[0].unsqueeze(0)  # start token
        #output = decoder_input
        is_teacher = random.random() < teacher_forcing_ratio
        raw = 0
        loss = 0

        for t in range(1, max_length - 1):
            #print(t,'t', decoder_hidden.size())

            output, decoder_hidden, mask = decoder(output, encoder_output, decoder_hidden)
            outputs.append(output)
            masks.append(mask.data)

            if t < len(target_variable):
                loss += criterion(output.view(1,-1), target_variable[t])
            else:
                loss += criterion(output.view(1,-1), Variable(torch.LongTensor([0])))

            #raw = output[:]
            output = Variable(output.data.max(dim=2)[1])
            #raw.append(output)
            if int(output.data[0].int()) == EOS_token:
                #print('eos token',t)
                break

            # teacher forcing
            if is_teacher and t < targets.size()[0]:
                #print(output,'out')
                output = targets[t].unsqueeze(0)
                #print(self.output_lang.index2word[int(output)])
        '''

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return torch.cat(outputs), torch.cat(masks).permute(1, 2, 0) , loss # batch, src, trg



    def trainIters(self, encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
        if (encoder is not None and decoder is not None and
                self.model_1_enc is None and self.model_2_dec is None):
            self.model_1_enc = encoder
            self.model_2_dec = decoder
        else:
            encoder = self.model_1_enc
            decoder = self.model_2_dec

        save_thresh = 2
        saved_files = 0

        save_num = 0
        print_loss_total = 0  # Reset every print_every


        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
        #adam_optimizer = optim.Adam(filter(lambda p: p.requires_grad, [encoder.parameters(), decoder.parameters()]),
        #                            lr=learning_rate)
        #self.opt_1 = adam_optimizer

        #decoder_optimizer = None

        training_pairs = [self.variablesFromPair(self.pairs[i]) for i in range(n_iters)]
        #training_pairs = self.pairs

        #criterion = nn.NLLLoss()
        criterion = nn.CrossEntropyLoss()


        if (self.opt_1 is None and self.opt_2 is None) or True:
            self.opt_1 = encoder_optimizer
            self.opt_2 = decoder_optimizer
            #self.opt_1 = adam_optimizer

        self.load_checkpoint()

        start = 1
        if self.start != 0 and self.start is not None:
            start = self.start + 1

        for iter in range(start, n_iters + 1):
            training_pair = training_pairs[iter - 1]

            #print(training_pair)
            #exit()

            input_variable = training_pair[0]
            question_variable = training_pair[1]
            if len(training_pair) > 2:
                target_variable = training_pair[2]
            else:
                target_variable = training_pair[1]

            is_auto = random.random() < hparams['autoencode']
            if is_auto:
                target_variable = training_pair[0]
                #print('is auto')



            outputs, masks , l = self.train(input_variable, target_variable,question_variable, encoder,
                                            decoder, encoder_optimizer, decoder_optimizer, criterion)

            print_loss_total += float(l)


            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('iter = '+str(iter)+ ', num of iters = '+str(n_iters) +", countdown = "+ str(save_thresh - save_num)
                      + ' ' + self.printable + ', saved files = ' + str(saved_files) + ', low = ' + str(self.long_term_loss))
                if iter % (print_every * 20) == 0:
                    save_num +=1
                    if (self.long_term_loss is None or print_loss_avg <= self.long_term_loss or save_num > save_thresh):

                        self.tag = 'timeout'
                        if self.long_term_loss is None or print_loss_avg <= self.long_term_loss:
                            self.tag = 'performance'

                        if self.long_term_loss is None or print_loss_avg <= self.long_term_loss:
                            self.long_term_loss = print_loss_avg

                        self.start = iter
                        save_num = 0
                        extra = ''
                        #if hparams['autoencode'] == True: extra = '.autoencode'
                        self.best_loss = print_loss_avg
                        self.save_checkpoint(num=iter,extra=extra)
                        saved_files += 1
                        print('======= save file '+ extra+' ========')
                    else:
                        print('skip save!')
                print('%s (%d %d%%) %.4f' % (self.timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))
                choice = random.choice(self.pairs)
                print('src:',choice[0])
                print('ques:', choice[1])
                print('ref:',choice[2])
                words, _ = self.evaluate(None, None, choice[0])
                #print(choice)
                print('ans:',words)
                print('try:',self._shorten(words))
                print("-----")




    def evaluate(self, encoder, decoder, sentence, max_length=MAX_LENGTH):
        if (encoder is not None and decoder is not None and
                self.model_1_enc is None and self.model_2_dec is None):
            self.model_1_enc = encoder
            self.model_2_dec = decoder
        else:
            encoder = self.model_1_enc
            decoder = self.model_2_dec

        input_variable = self.variableFromSentence(self.input_lang, sentence)
        input_length = input_variable.size()[0]

        decoder_hidden = torch.zeros((self.hidden_size  * encoder.n_layers))
        decoder_hidden = decoder_hidden.view(1,1,self.hidden_size * encoder.n_layers)

        #if input_length >= max_length : input_length = max_length

        #encoder_outputs = Variable(torch.zeros(max_length, self.hidden_size  ))
        #encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        encoder_output, encoder_hidden = encoder(input_variable)

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        output = decoder_input


        decoded_words = []
        decoder_attentions = torch.zeros(encoder_output.size()[0] , encoder_output.size()[0] )
        decoder_hidden = encoder_hidden[-decoder.n_layers:]  # take what we need from encoder

        seq, batch, _ = encoder_output.size()

        output = Variable(torch.zeros(1, batch).long() + SOS_token)  # start token


        outputs = []
        masks = []
        for di in range(max_length):

            output, decoder_hidden, mask = decoder(output, encoder_output, decoder_hidden)
            outputs.append(output)
            masks.append(mask.data)
            output = Variable(output.data.max(dim=2)[1]) #1

            if di -1 < len(decoder_attentions) : decoder_attentions[di-1] = mask.data
            #topv, topi = output.data.topk(1)
            ni = output # = next_words[0][0]
            #print(ni,'ni')
            if int(ni) == int(EOS_token):
                xxx = hparams['eol']
                decoded_words.append(xxx)
                print('eol found.')
                break
            else:
                decoded_words.append(self.output_lang.index2word[int(ni)])

        return decoded_words, decoder_attentions[:di + 1]

    def get_sentence(self, s_in):
        wordlist, _ = self.evaluate(None,None,s_in)

        ## filter words ##
        wordlist = self._shorten(wordlist)

        return wordlist

    def setup_for_interactive(self):
        self.do_interactive = True
        self.input_lang, self.output_lang, self.pairs = self.prepareData(self.train_fr, self.train_to, reverse=False, omit_unk=True)
        layers = hparams['layers']
        dropout = hparams['dropout']
        pytorch_embed_size = hparams['pytorch_embed_size']

        self.model_1_enc = Encoder(self.input_lang.n_words, pytorch_embed_size, self.hidden_size, layers, dropout=dropout)

        self.model_2_dec = Decoder(self.output_lang.n_words, pytorch_embed_size, self.hidden_size, layers, dropout=dropout)
        self.load_checkpoint()


if __name__ == '__main__':

    n = NMT()

    if False:
        att = EpisodicAttn(5, 7)
        print(att([' ',' ']))
        exit()

    if not n.do_review and not n.do_load_babi:
        n.task_normal_train()
    elif not n.do_load_babi:
        n.task_review_set()
    elif n.do_load_babi:
        n.task_babi_files()

    n.input_lang, n.output_lang, n.pairs = n.prepareData(n.train_fr, n.train_to, reverse=False, omit_unk=False)

    layers = hparams['layers']
    dropout = hparams['dropout']
    pytorch_embed_size = hparams['pytorch_embed_size']

    n.model_1_enc = Encoder(n.input_lang.n_words, pytorch_embed_size, n.hidden_size,layers, dropout=dropout)

    n.model_2_dec = Decoder(n.output_lang.n_words, pytorch_embed_size ,n.hidden_size, layers ,dropout=dropout)

    n.model_3_mem = MemRNN(n.hidden_size,n.hidden_size)

    n.model_4_att = EpisodicAttn(n.hidden_size, a_list_size=7)

    if use_cuda:
        n.model_1_enc = n.model_1_enc.cuda()
        n.model_2_dec = n.model_2_dec.cuda()

    if n.do_train:
        lr = hparams['learning_rate']
        n.trainIters(None, None, len(n.pairs), print_every=n.print_every, learning_rate=lr)
    if n.do_train_long:
        n.task_train_epochs()

    if n.do_interactive:
        n.load_checkpoint()
        n.task_interactive()

    if n.do_review:
        n.task_review_weights(n.pairs,stop_at_fail=False)

    if n.do_convert:
        n.load_checkpoint()
        n.task_convert()

    if n.do_infer:
        n.load_checkpoint()
        choice = random.choice(n.pairs)[0]
        print(choice)
        words, _ = n.evaluate(None,None,choice)
        print(words)

