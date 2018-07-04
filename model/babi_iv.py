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
import torch.nn.init as init
import time
import math
import argparse
from settings import hparams
import tokenize_weak
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
import numpy as np


'''
Some code was originally written by Yerevann Research Lab. This theano code
implements the DMN Network Model.

https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano

The MIT License (MIT)

Copyright (c) 2016 YerevaNN

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
------------------------

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
---------------------------

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
UNK_token = 0
SOS_token = 1
EOS_token = 2
MAX_LENGTH = hparams['tokens_per_sentence']

#hparams['teacher_forcing_ratio'] = 0.0
teacher_forcing_ratio = hparams['teacher_forcing_ratio'] #0.5
hparams['layers'] = 1
hparams['pytorch_embed_size'] = hparams['units']
#hparams['dropout'] = 0.3

word_lst = ['.', ',', '!', '?', "'", hparams['unk']]


################# pytorch modules ###############



class EpisodicAttn(nn.Module):

    def __init__(self,  hidden_size, a_list_size=7, dropout=0.3):
        super(EpisodicAttn, self).__init__()

        self.hidden_size = hidden_size
        self.a_list_size = a_list_size
        self.batch_size = hparams['batch_size']
        self.c_list_z = None

        #self.W_c1 = nn.Parameter(torch.zeros(hidden_size, 1 * hidden_size * a_list_size))
        #self.W_c2 = nn.Parameter(torch.zeros(1,hidden_size))

        self.out_a = nn.Linear( a_list_size * hidden_size,hidden_size)
        init.xavier_normal_(self.out_a.state_dict()['weight'])

        self.out_b = nn.Linear( hidden_size, 1)
        init.xavier_normal_(self.out_b.state_dict()['weight'])

        #self.b_c1 = nn.Parameter(torch.zeros(hidden_size,))
        #self.b_c2 = nn.Parameter(torch.zeros(1,))

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        #print('reset')
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            #print('here...')
            weight.data.uniform_(-stdv, stdv)
            if len(weight.size()) > 1:
                init.xavier_normal_(weight)

    def forward(self,concat_list):

        ''' attention list '''
        self.c_list_z = concat_list# torch.cat(concat_list,dim=0)
        #self.c_list_z = self.c_list_z.permute(1,0)
        #print(self.c_list_z.size(),'cz')

        self.c_list_z = self.dropout(self.c_list_z)
        l_1 = self.out_a(self.c_list_z)

        #l_1 = torch.mm(self.W_c1, self.c_list_z)
        #print(l_1.size(),'l1')
        l_1 = F.tanh(l_1) ## <---- this line?

        l_2 = self.out_b( l_1)
        #print(l_2, 'l2')
        #l_2 = F.tanh(l_2)
        #print(self.c_list_z.size(),'cz', l_1.size(), l_2)

        self.G = l_2 #* F.softmax(l_2, dim=1)

        return self.G


class MemRNN(nn.Module):
    def __init__(self, hidden_size, dropout=0.3):
        super(MemRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout1 = nn.Dropout(dropout) # this is just for if 'nn.GRU' is used!!
        self.gru = nn.GRU(hidden_size, hidden_size,dropout=0, num_layers=1, batch_first=False,bidirectional=False)
        self.reset_parameters()

    def reset_parameters(self):
        #print('reset')
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            #print('here...')
            weight.data.uniform_(-stdv, stdv)
            if len(weight.size()) > 1:
                init.xavier_normal_(weight)

    def prune_tensor(self, input, size):
        if len(input.size()) < size:
            input = input.unsqueeze(0)
        if len(input.size()) > size:
            input = input.squeeze(0)
        return input

    def forward(self, input, hidden=None):

        input = self.dropout1(input) # weak dropout
        num = 3
        input = self.prune_tensor(input,num)

        output, hidden_out = self.gru(input, hidden)

        #output = None

        return output, hidden_out

class Encoder(nn.Module):
    def __init__(self, source_vocab_size, embed_dim, hidden_dim,
                 n_layers, dropout=0.3, bidirectional=False, embedding=None):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.embed = nn.Embedding(source_vocab_size, embed_dim, padding_idx=1)
        self.gru = nn.GRU(embed_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

        if embedding is not None:
            self.embed.weight.data.copy_(torch.from_numpy(embedding))
            print('embedding encoder')
        #self.gru = MGRU(self.hidden_dim)

    def reset_parameters(self):
        #print('reset')
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            #print('here...')
            weight.data.uniform_(-stdv, stdv)
            if len(weight.size()) > 1:
                init.xavier_normal_(weight)

    def forward(self, source, hidden=None):

        embedded = self.embed(source)
        embedded = self.dropout(embedded)
        encoder_out = None

        encoder_out, encoder_hidden = self.gru( embedded, hidden)

        return encoder_out, encoder_hidden

class AnswerModule(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout=0.3):
        super(AnswerModule, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.batch_size = hparams['batch_size']

        self.out_a = nn.Linear(hidden_size, vocab_size)
        init.xavier_normal_(self.out_a.state_dict()['weight'])

        self.dropout = nn.Dropout(dropout)

        self.log_soft = nn.LogSoftmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self):
        #print('reset')
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            #print('here...')
            weight.data.uniform_(-stdv, stdv)
            if len(weight.size()) > 1:
                init.xavier_normal_(weight)

    def forward(self, mem, question_h):
        mem = self.dropout(mem)
        mem = mem.squeeze(0)#.permute(1,0)#.squeeze(0)

        out = self.out_a(mem)

        #out = self.log_soft(out)
        return out.permute(1,0)

#################### Wrapper ####################

class WrapMemRNN(nn.Module):
    def __init__(self,vocab_size, embed_dim,  hidden_size, n_layers, dropout=0.3, do_babi=True, bad_token_lst=[], freeze_embedding=False, embedding=None, print_to_screen=False):
        super(WrapMemRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.do_babi = do_babi
        self.print_to_screen = print_to_screen
        self.bad_token_lst = bad_token_lst
        self.embedding = embedding
        self.freeze_embedding = freeze_embedding
        self.teacher_forcing_ratio = hparams['teacher_forcing_ratio']

        gru_dropout = dropout * 0

        self.model_1_enc = Encoder(vocab_size, embed_dim, hidden_size, n_layers, dropout=dropout,embedding=embedding, bidirectional=False)
        self.model_2_enc = Encoder(vocab_size, embed_dim, hidden_size, n_layers, dropout=gru_dropout, embedding=embedding, bidirectional=False)
        self.model_3_mem_a = MemRNN(hidden_size, dropout=gru_dropout)
        self.model_3_mem_b = MemRNN(hidden_size, dropout=gru_dropout)
        self.model_4_att = EpisodicAttn(hidden_size, dropout=gru_dropout)
        self.model_5_ans = AnswerModule(vocab_size, hidden_size,dropout=dropout)

        self.input_var = None  # for input
        self.q_var = None  # for question
        self.answer_var = None  # for answer
        self.q_q = None  # extra question
        self.inp_c = None  # extra input
        self.inp_c_seq = None
        self.all_mem = None
        self.last_mem = None  # output of mem unit
        self.prediction = None  # final single word prediction
        self.memory_hops = hparams['babi_memory_hops']
        self.inv_idx = torch.arange(100 - 1, -1, -1).long() ## inverse index for 100 values

        self.reset_parameters()

        if self.freeze_embedding or self.embedding is not None:
            self.new_freeze_embedding()
        #self.criterion = nn.CrossEntropyLoss()

        pass

    def reset_parameters(self):
        #print('reset')
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            #print('here...')
            weight.data.uniform_(-stdv, stdv)
            if len(weight.size()) > 1:
                init.xavier_normal_(weight)

    def forward(self, input_variable, question_variable, target_variable, criterion=None):

        self.new_input_module(input_variable, question_variable)
        self.new_episodic_module()
        outputs,  ans = self.new_answer_module_simple()

        return outputs, None, ans, None

    def new_freeze_embedding(self):
        self.model_1_enc.embed.weight.requires_grad = False
        self.model_2_enc.embed.weight.requires_grad = False
        print('freeze embedding')
        pass

    def new_input_module(self, input_variable, question_variable):

        prev_h1 = []

        for ii in input_variable:

            ii = self.prune_tensor(ii, 2)

            out1, hidden1 = self.model_1_enc(ii, None)

            prev_h1.append(hidden1)

        self.inp_c_seq = prev_h1
        self.inp_c = prev_h1[-1]

        prev_h2 = []

        for ii in question_variable:
            ii = self.prune_tensor(ii, 2)

            out2, hidden2 = self.model_2_enc(ii, None)
            prev_h2.append(hidden2)

        self.q_q = hidden2[:,-1,:]

        return


    def new_episodic_module(self):
        if True:

            mem_list = []

            sequences = self.inp_c_seq

            for i in range(len(sequences)):

                m_list = [self.q_q.clone()]

                #print(sequences[i].size(),'seq')

                for iter in range(self.memory_hops):

                    x = self.new_attention_step(sequences[i], None, m_list[iter], self.q_q)

                    if self.print_to_screen and not self.training:
                        print(x,'x -- after', len(x), sequences[i].size())

                    e, _ = self.new_episode_small_step_2(sequences[i], x.permute(1,0), None)

                    assert len(sequences[i].size()) == 3
                    #print(e.size(),'e')
                    ee = e[:, 0, -1]#.permute(2,1,0)
                    #print(ee.size(),'ee')
                    _, out = self.model_3_mem_a(ee.unsqueeze(0), self.prune_tensor(m_list[iter], 3))

                    m_list.append(out)

                mem_list.append(m_list[self.memory_hops])

            mm_list = torch.cat(mem_list, dim=1)

            self.last_mem = mm_list

            #print(self.last_mem.size(),'lm')

        return None

    def new_episode_small_step(self, ct, g, prev_h):

        assert len(ct.size()) == 3
        bat, sen, emb = ct.size()

        last = [prev_h]

        ep = []
        for iii in range(sen):

            c = ct[0,iii,:].unsqueeze(0)

            if prev_h is not None:
                prev_h = self.prune_tensor(prev_h, 3)

            out, gru = self.model_3_mem_b(c, last[iii] )

            last.append(out)

            g = g.squeeze(0)
            gru = gru.squeeze(0).permute(1,0)

            #ggg = g[:, iii]
            ggg = g[iii]
            h = torch.mul(ggg , gru)#  + torch.mul((1 - g[iii]) , prev_h.permute(1,0))

            index = 0 #-1 # -2
            if last[iii + index] is not None:
                if False: h = h + torch.mul((1 - ggg), last[iii + index])

            if iii == sen - 1 : ep.append(h.unsqueeze(1))

        h = torch.cat(ep, dim=1)

        return h, gru

    def new_episode_small_step_2(self, ct, g, prev_h):

        assert len(ct.size()) == 3
        bat, sen, emb = ct.size()

        last = [None]

        ep = []
        for iii in range(sen):

            c = ct[0,iii,:].unsqueeze(0)

            if prev_h is not None:
                prev_h = self.prune_tensor(prev_h, 3)

            out, gru = self.model_3_mem_b(c, last[-1])

            last.append(out)

            g = g.squeeze(0)
            gru = gru.squeeze(0).permute(1,0)

            ggg = g[iii]
            h = torch.mul(ggg , gru)#  + torch.mul((1 - g[iii]) , prev_h.permute(1,0))

            #h = h.flip(0)
            #print(h)
            if last[-1] is not None:
                if True:
                    z = torch.mul((1 - ggg), last[-1 ])
                    h = h.permute(1,0).unsqueeze(0)
                    #print(h,'h')
                    h_new = h.index_select(2, self.inv_idx)
                    #print(h_new,'later')
                    h = h_new + z

            if iii == sen - 1 : ep.append(h.unsqueeze(1))

            last.append(h)

        h = torch.cat(ep, dim=1)

        return h, gru

    def new_attention_step(self, ct, prev_g, mem, q_q):

        q_q = self.prune_tensor(q_q,3)
        mem = self.prune_tensor(mem,3)

        assert len(ct.size()) == 3
        bat, sen, emb = ct.size()

        #print(sen,'len sen')

        att = []
        for iii in range(sen):
            c = ct[0,iii,:]

            concat_list = [
                c.unsqueeze(0),
                mem.squeeze(0),
                q_q.squeeze(0),
                (c * q_q).squeeze(0),
                (c * mem).squeeze(0),
                (torch.abs(c - q_q) ).squeeze(0),
                (torch.abs(c - mem) ).squeeze(0)
            ]
            #for ii in concat_list: print(ii.size())
            #print(sen,'sen')
            #exit()
            #z = F.sigmoid(z)
            concat_list = torch.cat(concat_list, dim=1)
            #print(concat_list.size(),'cl')
            att.append(concat_list)

        att = torch.cat(att, dim=0)
        #z = torch.cat(att, dim=0)
        z = self.model_4_att(att)
        z = F.sigmoid(z)
        #z =  F.softmax(z, dim=1) #F.sigmoid(z)
        #print(z.size(),'z')
        return z

    def prune_tensor(self, input, size):
        if len(input.size()) < size:
            input = input.unsqueeze(0)
        if len(input.size()) > size:
            input = input.squeeze(0)
        return input

    def new_answer_module_simple(self):
        #outputs

        ansx = self.model_5_ans(self.last_mem, None)

        ansx = F.softmax(ansx, dim=0)

        if self.print_to_screen:
            print(ansx, 'ansx printed')
            print(ansx.size(), 'ansx')
            vocab, sen = ansx.size()
            aa = torch.argmax(ansx, dim=0)
            print(aa.size(),'aa')
            for i in range(sen):
                zz = aa[i]
                z = ansx[:, i]
                a = torch.argmax(z, dim=0)
                print(a.item(), zz.item())
            print('----')
        #ans = torch.argmax(ansx,dim=1)#[0]


        return [None], ansx

        pass


######################## end pytorch modules ####################

class Lang:
    def __init__(self, name, limit=None):
        self.name = name
        self.limit = limit
        self.word2index = {hparams['unk']:0, hparams['sol']: 1, hparams['eol']: 2}
        self.word2count = {hparams['unk']:1, hparams['sol']: 1, hparams['eol']: 1}
        self.index2word = {0: hparams['unk'], 1: hparams['sol'], 2: hparams['eol']}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if self.limit is None or self.n_words < self.limit :
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

        self.model_0_wra = None
        self.opt_1 = None
        self.embedding_matrix = None
        self.criterion = None

        self.best_loss = None
        self.long_term_loss = 0
        self.tag = ''

        self.input_lang = None
        self.output_lang = None
        self.question_lang = None
        self.vocab_lang = None

        self.print_every = hparams['steps_to_stats']
        self.epochs = hparams['epochs']
        self.hidden_size = hparams['units']
        self.first_load = True
        self.memory_hops = 5
        self.start = 0
        self.this_epoch = 0
        self.score = 0
        self.saved_files = 0
        self.babi_num = '1'
        self.score_list = []
        self.score_list_training = []
        self.teacher_forcing_ratio = hparams['teacher_forcing_ratio']

        ''' used by auto-stop function '''
        self.epochs_since_adjustment = 0
        self.lr_adjustment_num = 0
        self.lr_low = hparams['learning_rate']
        self.lr_increment = self.lr_low / 4.0
        self.best_accuracy = None
        self.best_accuracy_old = None
        self.record_threshold = 95.00
        self._recipe_switching = 0

        self.uniform_low = -1.0
        self.uniform_high = 1.0

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
        self.do_hide_unk = False
        self.do_conserve_space = False
        self.do_test_not_train = False
        self.do_freeze_embedding = False
        self.do_load_embeddings = False
        self.do_auto_stop = False
        self.do_skip_validation = False
        self.do_print_to_screen = False
        self.do_recipe_dropout = False
        self.do_recipe_lr = False
        self.do_batch_process = True

        self.printable = ''


        parser = argparse.ArgumentParser(description='Train some NMT values.')
        parser.add_argument('--mode', help='mode of operation. (train, infer, review, long, interactive, plot)')
        parser.add_argument('--printable', help='a string to print during training for identification.')
        parser.add_argument('--basename', help='base filename to use if it is different from settings file.')
        parser.add_argument('--autoencode', help='enable auto encode from the command line with a ratio.')
        parser.add_argument('--train-all', help='(broken) enable training of the embeddings layer from the command line',
                            action='store_true')
        #parser.add_argument('--convert-weights',help='convert weights', action='store_true')
        parser.add_argument('--load-babi', help='Load three babi input files instead of chatbot data',
                            action='store_true')
        parser.add_argument('--hide-unk', help='hide all unk tokens', action='store_true')
        parser.add_argument('--use-filename', help='use base filename as basename for saved weights.', action='store_true')
        parser.add_argument('--conserve-space', help='save only one file for all training epochs.',
                            action='store_true')
        parser.add_argument('--babi-num', help='number of which babi test set is being worked on')
        parser.add_argument('--units', help='Override UNITS hyper parameter.')
        parser.add_argument('--test',help='Disable all training. No weights will be changed and no new weights will be saved.',
                            action='store_true')
        parser.add_argument('--lr', help='learning rate')
        parser.add_argument('--freeze-embedding', help='Stop training on embedding elements',action='store_true')
        parser.add_argument('--load-embed-size', help='Load trained embeddings of the following size only: 50, 100, 200, 300')
        parser.add_argument('--auto-stop', help='Auto stop during training.', action='store_true')
        parser.add_argument('--dropout', help='set dropout ratio from the command line. (Float value)')
        parser.add_argument('--no-validation', help='skip validation printout until first lr correction.',action='store_true')
        parser.add_argument('--print-to-screen', help='print some extra values to the screen for debugging', action='store_true')
        parser.add_argument('--cuda', help='enable cuda on device.', action='store_true')
        parser.add_argument('--lr-adjust', help='resume training at particular lr adjust value.')
        parser.add_argument('--save-num', help='threshold for auto-saving files. (0-100)')
        parser.add_argument('--recipe-dropout', help='use dropout recipe', action='store_true')
        parser.add_argument('--recipe-lr', help='use learning rate recipe', action='store_true')
        parser.add_argument('--batch',help='enable batch processing. (default)',action='store_true')

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
        #if self.args['convert_weights'] == True: self.do_convert = True
        if self.args['load_babi'] == True: self.do_load_babi = True
        if self.args['hide_unk'] == True or self.do_load_babi: self.do_hide_unk = True
        if self.args['use_filename'] == True:
            z = sys.argv[0]
            if z.startswith('./'): z = z[2:]
            hparams['base_filename'] = z.split('.')[0]
            print(hparams['base_filename'], 'basename')
        if self.args['conserve_space'] == True: self.do_conserve_space = True
        if self.args['babi_num'] is not None: self.babi_num = self.args['babi_num']
        if self.args['units'] is not None:
            hparams['units'] = int(self.args['units'])
            hparams['pytorch_embed_size'] = hparams['units']
            self.hidden_size = hparams['units']
        if self.args['test'] == True:
            self.do_test_not_train = True
            hparams['teacher_forcing_ratio'] = 0.0
        if self.args['lr'] is not None: hparams['learning_rate'] = float(self.args['lr'])
        if self.args['freeze_embedding'] == True: self.do_freeze_embedding = True
        if self.args['load_embed_size'] is not None:
            hparams['embed_size'] = int(self.args['load_embed_size'])
            self.do_load_embeddings = True
            self.do_freeze_embedding = True
        if self.args['auto_stop'] == True: self.do_auto_stop = True
        if self.args['dropout'] is not None: hparams['dropout'] = float(self.args['dropout'])
        if self.args['no_validation'] is True: self.do_skip_validation = True
        if self.args['print_to_screen'] is True: self.do_print_to_screen = True
        if self.args['cuda'] is True: hparams['cuda'] = True
        if self.args['save_num'] is not None: self.record_threshold = float(self.args['save_num'])
        if self.args['recipe_dropout'] is not False: self.do_recipe_dropout = True
        if self.args['recipe_lr'] is not False: self.do_recipe_lr = True
        if self.args['batch'] is not False:
            self.do_batch_process = True
            print('batch operation now enabled by default.')
        if self.printable == '': self.printable = hparams['base_filename']
        if hparams['cuda']: torch.set_default_tensor_type('torch.cuda.FloatTensor')

        ''' reset lr vars if changed from command line '''
        self.lr_low = hparams['learning_rate'] #/ 100.0
        self.lr_increment = self.lr_low / 4.0
        if self.args['lr_adjust'] is not None:
            self.lr_adjustment_num = int(self.args['lr_adjust'])
            hparams['learning_rate'] = self.lr_low + float(self.lr_adjustment_num) * self.lr_increment

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

    def task_babi_test_files(self):
        self.train_fr = hparams['data_dir'] + hparams['test_name'] + '.' + hparams['babi_name'] + '.' + hparams['src_ending']
        self.train_to = hparams['data_dir'] + hparams['test_name'] + '.' + hparams['babi_name'] + '.' + hparams['tgt_ending']
        self.train_ques = hparams['data_dir'] + hparams['test_name'] + '.' + hparams['babi_name'] + '.' + hparams['question_ending']
        pass

    def task_babi_valid_files(self):
        self.train_fr = hparams['data_dir'] + hparams['valid_name'] + '.' + hparams['babi_name'] + '.' + hparams['src_ending']
        self.train_to = hparams['data_dir'] + hparams['valid_name'] + '.' + hparams['babi_name'] + '.' + hparams['tgt_ending']
        self.train_ques = hparams['data_dir'] + hparams['valid_name'] + '.' + hparams['babi_name'] + '.' + hparams['question_ending']
        pass

    def task_set_embedding_matrix(self):
        print('stage: set_embedding_matrix')
        glove_data = hparams['data_dir'] + hparams['embed_name']
        from gensim.models.keyedvectors import KeyedVectors
        import numpy as np

        embed_size = int(hparams['embed_size'])

        embeddings_index = {}
        if not os.path.isfile(glove_data) :
            self.embedding_matrix = None  # np.zeros((len(self.vocab_list),embed_size))
            #self.trainable = True
        else:
            # load embedding
            glove_model = KeyedVectors.load_word2vec_format(glove_data, binary=False)

            f = open(glove_data)
            for line in range(self.output_lang.n_words): #len(self.vocab_list)):
                #if line == 0: continue
                word = self.output_lang.index2word[line]
                # print(word, line)
                if word in glove_model.wv.vocab:
                    #print('fill with values',line)
                    values = glove_model.wv[word]
                    value = np.asarray(values, dtype='float32')
                    embeddings_index[word] = value
                else:
                    print('fill with random values',line, word)
                    value = np.random.uniform(low=self.uniform_low, high=self.uniform_high, size=(embed_size,))
                    # value = np.zeros((embed_size,))
                    embeddings_index[word] = value
            f.close()

            self.embedding_matrix = np.zeros((self.output_lang.n_words, embed_size))
            for i in range( self.output_lang.n_words ):#len(self.vocab_list)):
                word = self.output_lang.index2word[i]
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all random.
                    self.embedding_matrix[i] = embedding_vector[:embed_size]
                else:
                    print('also fill with random values',i,word)

                    self.embedding_matrix[i] = np.random.uniform(high=self.uniform_high, low=self.uniform_low,
                                                                 size=(embed_size,))
                    # self.embedding_matrix[i] = np.zeros((embed_size,))
        pass

    def task_review_weights(self, pairs, stop_at_fail=False):
        plot_losses = []
        num = 0 # hparams['base_file_num']
        for i in range(100):
            local_filename = hparams['save_dir'] + hparams['base_filename'] + '.'+ str(num) + '.pth'
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
            import matplotlib.pyplot as plt
            import matplotlib.ticker as ticker

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
            self.this_epoch = i
            self.printable = 'epoch #' + str(i+1)
            self.do_test_not_train = False
            #self.score = 0.0

            self.input_lang, self.output_lang, self.pairs = self.prepareData(self.train_fr, self.train_to,
                                                                             lang3=self.train_ques, reverse=False,
                                                                             omit_unk=self.do_hide_unk)

            self.first_load = True
            self.train_iters(None, None, len(self.pairs), print_every=self.print_every, learning_rate=lr)
            self.start = 0

            print('auto save.')
            print('%.2f' % self.score,'score')

            self.save_checkpoint(num=len(self.pairs))
            self.saved_files += 1
            self.validate_iters()
            self.start = 0
            self.task_babi_files()
        self.input_lang, self.output_lang, self.pairs = self.prepareData(self.train_fr, self.train_to,lang3=self.train_ques, reverse=False, omit_unk=self.do_hide_unk)

        pass

    def task_interactive(self):

        print('-------------------')
        while True:
            line = input("> ")
            line = tokenize_weak.format(line)
            print(line)
            line = self.variableFromSentence(self.input_lang, line, add_eol=True)
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
            self.vocab_lang = Lang(load_vocab_file, limit=hparams['num_vocab_total'])
            pass

        if reverse:
            self.pairs = [list(reversed(p)) for p in self.pairs]
            self.input_lang = Lang(lang2, limit=hparams['num_vocab_total'])
            self.output_lang = Lang(lang1, limit=hparams['num_vocab_total'])
        else:
            self.input_lang = Lang(lang1, limit=hparams['num_vocab_total'])
            self.output_lang = Lang(lang2, limit=hparams['num_vocab_total'])

        if hparams['autoencode'] == 1.0:
            self.pairs = [ [p[0], p[0], p[0]] for p in self.pairs]
            self.output_lang = self.input_lang

        return self.input_lang, self.output_lang, self.pairs




    def prepareData(self,lang1, lang2,lang3=None, reverse=False, omit_unk=False):
        if hparams['vocab_name'] is not None:
            v_name = hparams['data_dir'] + hparams['vocab_name']
            v_name = v_name.replace('big', hparams['babi_name'])
        else:
            v_name = None

        if not self.do_load_babi:
            self.input_lang, self.output_lang, self.pairs = self.readLangs(lang1, lang2, lang3=None,# babi_ending=True,
                                                                           reverse=reverse,
                                                                           load_vocab_file=v_name)
            #lang3 = None
        else:
            self.input_lang, self.output_lang, self.pairs = self.readLangs(lang1, lang2, lang3=lang3, #self.train_ques,
                                                                           reverse=False,
                                                                           babi_ending=True,
                                                                           load_vocab_file=v_name)
            #lang3 = self.train_ques
        print("Read %s sentence pairs" % len(self.pairs))
        #self.pairs = self.filterPairs(self.pairs)
        #print("Trimmed to %s sentence pairs" % len(self.pairs))
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

        if self.do_load_embeddings:
            self.task_set_embedding_matrix()

        return self.input_lang, self.output_lang, self.pairs


    def indexesFromSentence(self,lang, sentence):
        s = sentence.split(' ')
        sent = []
        for word in s:
            if word in lang.word2index:
                if word == hparams['eol']: word = EOS_token
                elif word == hparams['sol']: word = SOS_token
                else: word = lang.word2index[word]
                sent.append(word)
            elif not self.do_hide_unk:
                sent.append(lang.word2index[hparams['unk']])
        if len(sent) >= MAX_LENGTH and not self.do_load_babi:
            sent = sent[:MAX_LENGTH]
            sent[-1] = EOS_token
        if self.do_load_babi and False:
            sent.append(EOS_token)
            #print(sent,'<<<<')
        return sent

        #return [lang.word2index[word] for word in sentence.split(' ')]

    def variables_for_batch(self, pairs, size, start):
        if start + size >= len(pairs) and start < len(pairs) - 1:
            size = len(pairs) - start - 1
            print('process size', size,'next')
        if size == 0 or start >= len(pairs):
            print('empty return.')
            return self.variablesFromPair(('','',''))
        g1 = []
        g2 = []
        g3 = []

        group = pairs[start:start + size]
        for i in group:
            g = self.variablesFromPair(i)
            #print(g[0])
            g1.append(g[0].squeeze(1))
            g2.append(g[1].squeeze(1))
            g3.append(g[2].squeeze(1))

        return (g1, g2, g3)

    def variableFromSentence(self,lang, sentence, add_eol=False):
        indexes = self.indexesFromSentence(lang, sentence)
        if add_eol: indexes.append(EOS_token)
        result = Variable(torch.LongTensor(indexes).unsqueeze(1))#.view(-1, 1))
        #print(result.size(),'r')
        if hparams['cuda']:
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
                    'state_dict': self.model_0_wra.state_dict(),
                    'best_prec1': None,
                    'optimizer': self.opt_1.state_dict(),
                    'best_loss': self.best_loss,
                    'long_term_loss' : self.long_term_loss,
                    'tag': self.tag,
                    'score': self.score
                }
            ]
        else:
            z = [
                {
                    'epoch': 0,
                    'start': self.start,
                    'arch': None,
                    'state_dict': self.model_0_wra.state_dict(),
                    'best_prec1': None,
                    'optimizer': None , # self.opt_1.state_dict(),
                    'best_loss': self.best_loss,
                    'long_term_loss': self.long_term_loss,
                    'tag': self.tag,
                    'score': self.score
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
        if self.do_load_babi or self.do_conserve_space:
            num = self.this_epoch * len(self.pairs) + num
            torch.save(state,basename+ '.best.pth')
            #if self.do_test_not_train: self.score_list.append('%.2f' % self.score)
            if ((self.best_accuracy_old is None and self.best_accuracy is not None) or
                    (self.best_accuracy_old is not None and self.best_accuracy > self.best_accuracy_old)):
                torch.save(state, basename + '.' + str(int(math.floor(self.best_accuracy * 100))) + '.best.pth')
                self.best_accuracy_old = self.best_accuracy
            return
        torch.save(state, basename + extra + '.' + str(num)+ '.pth')
        if is_best:
            os.system('cp '+ basename + extra +  '.' + str(num) + '.pth' + ' '  +
                      basename + '.best.pth')

    def load_checkpoint(self, filename=None):
        if self.first_load:
            self.first_load = False
            basename = hparams['save_dir'] + hparams['base_filename'] + '.best.pth'
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
                try:
                    self.score = checkpoint[0]['score']
                except:
                    print('no score saved with checkpoint')
                    self.score = 0

                if hparams['zero_start'] is True:
                    self.start = 0

                self.model_0_wra.load_state_dict(checkpoint[0]['state_dict'])
                if self.do_freeze_embedding:
                    self.model_0_wra.new_freeze_embedding()
                if self.opt_1 is not None:
                    #####
                    try:
                        self.opt_1.load_state_dict(checkpoint[0]['optimizer'])
                        if self.opt_1.param_groups[0]['lr'] != hparams['learning_rate']:
                            raise Exception('new optimizer...')
                    except:
                        #print('new optimizer', hparams['learning_rate'])
                        #parameters = filter(lambda p: p.requires_grad, self.model_0_wra.parameters())
                        #self.opt_1 = optim.Adam(parameters, lr=hparams['learning_rate'])
                        self.opt_1 = self._make_optimizer()
                print("loaded checkpoint '"+ basename + "' ")
                if self.do_recipe_dropout:
                    self.set_dropout(hparams['dropout'])

            else:
                print("no checkpoint found at '"+ basename + "'")

    def _make_optimizer(self):
        print('new optimizer', hparams['learning_rate'])
        parameters = filter(lambda p: p.requires_grad, self.model_0_wra.parameters())
        return optim.Adam(parameters, lr=hparams['learning_rate'])
        #return optim.SGD(parameters, lr=hparams['learning_rate'])


    def _auto_stop(self):
        threshold = 70.00
        use_recipe_switching = self.do_recipe_dropout and self.do_recipe_lr

        use_lr_recipe = self.do_recipe_lr
        use_dropout_recipe = self.do_recipe_dropout

        ''' switch between two recipe types '''
        if use_recipe_switching and self._recipe_switching % 2 == 0:
            use_dropout_recipe = False
            use_lr_recipe = True
        elif use_recipe_switching and self._recipe_switching % 2 == 1:
            use_lr_recipe = False
            use_dropout_recipe = True

        self.epochs_since_adjustment += 1

        if self.epochs_since_adjustment > 0:

            z1 = z2 = z3 = z4 = 0.0

            if len(self.score_list_training) >= 1:
                z1 = float(self.score_list_training[-1])
            if len(self.score_list_training) >= 3:
                z2 = float(self.score_list_training[-2])
                z3 = float(self.score_list_training[-3])

            if len(self.score_list) > 0:
                z4 = float(self.score_list[-1])

            zz1 = z1 == 100.00 and z4 != 100.00 #and z2 == 100.00  ## TWO IN A ROW

            zz2 = z1 == z2 and z1 == z3 and z1 != 0.0 ## TWO IN A ROW

            if ( len(self.score_list) >= 2 and (
                    (float(self.score_list[-2]) == 100 and float(self.score_list[-1]) == 100) or
                    (float(self.score_list[-2]) == float(self.score_list[-1]) and
                     float(self.score_list[-1]) != 0.0))):

                self.do_skip_validation = False

                ''' adjust learning_rate to different value if possible. -- validation '''

                if float(self.score_list[-1]) == 100.00 and float(self.score_list[-2]) == 100.00 and z1 == 100.00:
                    time.ctime()
                    t = time.strftime('%l:%M%p %Z on %b %d, %Y')
                    print(t)
                    print('list:', self.score_list)
                    exit()

                if self.lr_adjustment_num < 1 and use_dropout_recipe:
                    hparams['dropout'] = 0.0
                    self.set_dropout(0.0)

            if len(self.score_list_training) < 1: return

            if z1 >= threshold and self.lr_adjustment_num != 0 and (self.lr_adjustment_num % 8 == 0 or self.epochs_since_adjustment > 15 ):
                if use_lr_recipe:
                    hparams['learning_rate'] = self.lr_low  # self.lr_increment + hparams['learning_rate']
                self.epochs_since_adjustment = 0
                self.do_skip_validation = False
                self._recipe_switching += 1
                if use_dropout_recipe:
                    hparams['dropout'] = 0.00
                    self.set_dropout(0.00)
                print('max changes or max epochs')

            if self.lr_adjustment_num > 25 or self.epochs_since_adjustment > 300:
                print('max adjustments -- quit')
                exit()

            if ((zz2) or (zz1 ) or ( abs(z4 - z1) > 10.0 and self.lr_adjustment_num <= 2) ):

                ''' adjust learning_rate to different value if possible. -- training '''

                if (float(self.score_list_training[-1]) == 100.00 and
                        float(self.score_list[-1]) != 100.00):
                    if use_lr_recipe:
                        hparams['learning_rate'] = self.lr_increment + hparams['learning_rate']
                    if use_dropout_recipe:
                        hparams['dropout'] = hparams['dropout'] + 0.025
                        self.set_dropout(hparams['dropout'])
                    self.do_skip_validation = False
                    self.lr_adjustment_num += 1
                    self.epochs_since_adjustment = 0
                    print('train reached 100 but not validation')

            elif use_lr_recipe and False:
                print('reset learning rate.')
                hparams['learning_rate'] = self.lr_low ## essentially old learning_rate !!



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

    def set_dropout(self, p):
        print('dropout',p)
        if self.model_0_wra is not None:
            self.model_0_wra.model_1_enc.dropout.p = p
            #self.model_0_wra.model_4_att.dropout.p = p
            self.model_0_wra.model_5_ans.dropout.p = p


    #######################################

    def train(self,input_variable, target_variable,question_variable, encoder, decoder, wrapper_optimizer, decoder_optimizer, memory_optimizer, attention_optimizer, criterion, max_length=MAX_LENGTH):

        if criterion is not None:
            wrapper_optimizer.zero_grad()
            self.model_0_wra.train()
            outputs, _, ans, _ = self.model_0_wra(input_variable, question_variable, target_variable,
                                                  criterion)

            if self.do_batch_process:
                target_variable = torch.cat(target_variable,dim=0)
                ans = ans.permute(1,0)
            else:
                target_variable = target_variable[0]
                print(len(ans),ans.size())
                ans = torch.argmax(ans,dim=1)
                #ans = ans[0]

            #ans = ans.permute(1,0)

            #print(ans.size(), target_variable.size(),'criterion')

            loss = criterion(ans, target_variable)
            loss.backward()
            wrapper_optimizer.step()

        else:
            self.model_0_wra.eval()
            with torch.no_grad():
                outputs, _, ans, _ = self.model_0_wra(input_variable, question_variable, target_variable,
                                                      criterion)
                loss = None
                ans = ans.permute(1,0)


        return outputs, ans , loss

    #######################################

    def train_iters(self, encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.001):

        save_thresh = 2
        #self.saved_files = 0
        step = 1
        save_num = 0
        print_loss_total = 0  # Reset every print_every
        num_right = 0
        num_tot = 0
        num_right_small = 0
        num_count = 0
        temp_batch_size = 0

        if self.opt_1 is None or self.first_load:

            wrapper_optimizer = self._make_optimizer()
            self.opt_1 = wrapper_optimizer

        #self.criterion = nn.NLLLoss()
        self.criterion = nn.CrossEntropyLoss(size_average=False)

        training_pairs = [self.variablesFromPair(self.pairs[i]) for i in range(n_iters)]

        if not self.do_test_not_train:
            criterion = self.criterion
        else:
            criterion = None

        self.load_checkpoint()

        start = 1
        if self.do_load_babi:
            self.start = 0

        if self.start != 0 and self.start is not None and not self.do_batch_process:
            start = self.start + 1

        if self.do_load_babi and  self.do_test_not_train:

            #print('list:', ', '.join(self.score_list))
            print('hidden:', hparams['units'])
            for param_group in self.opt_1.param_groups:
                print(param_group['lr'], 'lr')
            print(self.output_lang.n_words, 'num words')

        print(self.train_fr,'loaded file')

        print("-----")

        if self.do_load_babi:
            if self.do_test_not_train:
                self.model_0_wra.eval()

            else:
                self.model_0_wra.train()

        if self.do_batch_process:
            step = hparams['batch_size']
            start = 0

        for iter in range(start, n_iters + 1, step):

            if not self.do_batch_process:
                training_pair = training_pairs[iter - 1]

                input_variable = training_pair[0]
                question_variable = training_pair[1]

                if len(training_pair) > 2:
                    target_variable = training_pair[2]
                else:
                    question_variable = training_pair[0]
                    target_variable = training_pair[1]

                is_auto = random.random() < hparams['autoencode']
                if is_auto:
                    target_variable = training_pair[0]
                    #print('is auto')
            elif self.do_batch_process and (iter ) % hparams['batch_size'] == 0 and iter < len(self.pairs):
                group = self.variables_for_batch(self.pairs, hparams['batch_size'], iter)

                input_variable = group[0]
                question_variable = group[1]
                target_variable = group[2]

                temp_batch_size = len(input_variable)

            elif self.do_batch_process:
                continue
                pass

            #print(iter,'iter')
            outputs, ans, l = self.train(input_variable, target_variable, question_variable, encoder,
                                            decoder, self.opt_1, None,
                                            None, None, criterion)
            num_count += 1

            if self.do_load_babi:

                for i in range(len(target_variable)):
                    o_val = torch.argmax(ans[i], dim=0)[0]
                    t_val = target_variable[i]

                    if int(o_val.item()) == int(t_val.item()):
                        num_right += 1
                        num_right_small += 1

                if False and int(outputs[0].int()) == int(target_variable[0][0].int()):
                    num_right += 1
                    num_right_small += 1

                if self.do_batch_process: num_tot += temp_batch_size # hparams['batch_size']
                else: num_tot += 1

                self.score = float(num_right/num_tot) * 100

            if l is not None:
                print_loss_total += float(l)

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0

                print('iter = '+str(iter)+ ', num of iters = '+str(n_iters) +", countdown = "+ str(save_thresh - save_num)
                      + ', ' + self.printable + ', saved files = ' + str(self.saved_files) + ', low loss = %.6f' % self.long_term_loss)
                if iter % (print_every * 20) == 0 or self.do_load_babi:
                    save_num +=1
                    if (self.long_term_loss is None or print_loss_avg <= self.long_term_loss or save_num > save_thresh):

                        self.tag = 'timeout'
                        if self.long_term_loss is None or print_loss_avg <= self.long_term_loss:
                            self.tag = 'performance'

                        if ((self.long_term_loss is None or self.long_term_loss == 0 or
                             print_loss_avg <= self.long_term_loss) and not self.do_test_not_train):
                            self.long_term_loss = print_loss_avg

                        self.start = iter
                        save_num = 0
                        extra = ''
                        #if hparams['autoencode'] == True: extra = '.autoencode'
                        self.best_loss = print_loss_avg
                        if not self.do_test_not_train and not self.do_load_babi:
                            self.save_checkpoint(num=iter,extra=extra)
                            self.saved_files += 1
                            print('======= save file '+ extra+' ========')
                    elif not self.do_load_babi:
                        print('skip save!')
                print('(%d %d%%) %.6f loss' % (iter, iter / n_iters * 100, print_loss_avg), end=' ')
                if self.do_batch_process: print('- batch-size', temp_batch_size)
                else: print('')

                if not self.do_skip_validation and True:
                    ###########################
                    choice = random.choice(self.pairs)
                    print('src:',choice[0])
                    question = None
                    if self.do_load_babi:
                        print('ques:', choice[1])
                        print('ref:',choice[2])
                    else:
                        print('tgt:',choice[1])
                    nums = self.variablesFromPair(choice)
                    if self.do_load_babi:
                        question = nums[1]
                        target = nums[2]
                    if not self.do_load_babi:
                        question = nums[0]
                        target = None
                    words, _ = self.evaluate(None, None, nums[0], question=question, target_variable=target)
                    #print(choice)
                    if not self.do_load_babi:
                        print('ans:',words)
                        print('try:',self._shorten(words))
                        #self._word_from_prediction()
                    ############################
                if self.do_load_babi and self.do_test_not_train:

                    print('current accuracy: %.4f' % self.score, '- num right '+ str(num_right_small))
                    num_right_small = 0

                if self.do_load_babi and not self.do_test_not_train:

                    print('training accuracy: %.4f' % self.score, '- num right '+ str(num_right_small))
                    num_right_small = 0

                if self.lr_adjustment_num > 0:
                    if self._recipe_switching % 2 == 0 or not self.do_recipe_dropout:
                        print('[ lr adjust:', self.lr_adjustment_num, '-', hparams['learning_rate'],',', self.epochs_since_adjustment ,']')
                    if self._recipe_switching % 2 == 1 or not self.do_recipe_lr:
                        print('[ dropout adjust:', self.lr_adjustment_num,'-', hparams['dropout'],',',self.epochs_since_adjustment,']')

                if self.score_list is not None and len(self.score_list_training) > 0 and len(self.score_list) > 0:
                    print('[ last train:', self.score_list_training[-1],']',end='')
                    if self.do_test_not_train:
                        print('[ older valid:', self.score_list[-1],']')
                    else:
                        print('[ last valid:', self.score_list[-1],']')
                elif len(self.score_list_training) >= 1 and self.do_skip_validation:
                    print('[ last train:', self.score_list_training[-1],'][ no valid ]')

                print("-----")

        if self.do_batch_process:
            self.save_checkpoint(num=len(self.pairs))

        str_score = ' %.2f'
        if self.score >= 100.00: str_score = '%.2f'
        if self.score < 10.00: str_score = '  %.2f'

        if not self.do_test_not_train and self.do_load_babi:
            self.score_list_training.append(str_score % self.score)

        if self.do_test_not_train and self.do_load_babi:
            self.score_list.append(str_score % self.score)
            if self.do_auto_stop:
                self._auto_stop()

        if self.do_load_babi:
            print('train list:', ', '.join(self.score_list_training))
            print('valid list:', ', '.join(self.score_list))
        print('dropout:',hparams['dropout'])
        print('learning rate:', hparams['learning_rate'])
        print(num_count, 'exec count')

    def evaluate(self, encoder, decoder, sentence, question=None, target_variable=None, max_length=MAX_LENGTH):

        input_variable = sentence
        question_variable = Variable(torch.LongTensor([UNK_token])) # [UNK_token]

        if target_variable is None:
            sos_token = Variable(torch.LongTensor([SOS_token]))
        else:
            sos_token = target_variable[0]

        if question is not None:
            question_variable = question
            if not self.do_load_babi: sos_token = question_variable

        if False:
            context_array = [ [] for _ in range(hparams['batch_size'])]
            question_array = [ [] for _ in range(hparams['batch_size'])]
            target_array = [ [] for _ in range(hparams['batch_size'])]

            context_array[0] = input_variable.squeeze(0)
            question_array[0] = question_variable.squeeze(0)
            target_array[0] = sos_token.squeeze(0)

            print(context_array,'ca')

        #print(question_variable.squeeze(0).squeeze(0).permute(1,0).squeeze(0).size(),'iv')

        self.model_0_wra.eval()
        with torch.no_grad():
            outputs, _, ans , _ = self.model_0_wra([input_variable.squeeze(0).squeeze(0).permute(1,0).squeeze(0)],
                                                   [question_variable.squeeze(0).squeeze(0).permute(1,0).squeeze(0)],
                                                   [sos_token.squeeze(0).squeeze(0).squeeze(0)],
                                                   None)
        outputs = [ans]
        #####################

        decoded_words = []
        for di in range(len(outputs)):

            output = outputs[di] #.permute(1,0)

            output = output.permute(1,0)#torch.cat(output, dim=0)

            ni = torch.argmax(output, dim=1)[0] # = next_words[0][0]

            if int(ni) == int(EOS_token):
                xxx = hparams['eol']
                decoded_words.append(xxx)
                print('eol found.')
                if True: break
            else:
                if di < 4:
                    print(int(ni), self.output_lang.index2word[int(ni)])
                if di == 5 and len(outputs) > 5:
                    print('...etc')
                decoded_words.append(self.output_lang.index2word[int(ni)])


        return decoded_words, None #decoder_attentions[:di + 1]



    def validate_iters(self):
        if self.do_skip_validation:
            self.score_list.append('00.00')
            return
        self.task_babi_valid_files()
        self.printable = 'validate'
        self.input_lang, self.output_lang, self.pairs = self.prepareData(self.train_fr, self.train_to,lang3=self.train_ques, reverse=False, omit_unk=self.do_hide_unk)
        self.do_test_not_train = True
        self.first_load = True
        self.load_checkpoint()
        lr = hparams['learning_rate']
        self.start = 0
        self.train_iters(None,None, len(self.pairs), print_every=self.print_every, learning_rate=lr)
        if len(self.score_list) > 0 and float(self.score_list[-1]) >= self.record_threshold: #100.00:
            self.best_accuracy = float(self.score_list[-1])
            self.save_checkpoint(num=len(self.pairs))

        pass

    def setup_for_interactive(self):
        self.do_interactive = True
        self.input_lang, self.output_lang, self.pairs = self.prepareData(self.train_fr, self.train_to,lang3=self.train_ques, reverse=False, omit_unk=self.do_hide_unk)
        layers = hparams['layers']
        dropout = hparams['dropout']
        pytorch_embed_size = hparams['pytorch_embed_size']

        self.model_0_wra = WrapMemRNN(self.input_lang.n_words, pytorch_embed_size, self.hidden_size,layers,
                                      dropout=dropout,do_babi=self.do_load_babi,
                                      freeze_embedding=self.do_freeze_embedding, embedding=self.embedding_matrix,
                                      print_to_screen=self.do_print_to_screen)
        if hparams['cuda']: self.model_0_wra = self.model_0_wra.cuda()

        self.load_checkpoint()

    def setup_for_babi_test(self):
        #hparams['base_filename'] = filename
        self.printable = hparams['base_filename']

        self.do_test_not_train = True
        #self.task_babi_files()
        self.task_babi_test_files()
        self.input_lang, self.output_lang, self.pairs = self.prepareData(self.train_fr, self.train_to,lang3=self.train_ques, reverse=False,
                                                                         omit_unk=self.do_hide_unk)
        hparams['num_vocab_total'] = self.output_lang.n_words

        layers = hparams['layers']
        dropout = hparams['dropout']
        pytorch_embed_size = hparams['pytorch_embed_size']

        self.model_0_wra = WrapMemRNN(self.input_lang.n_words, pytorch_embed_size, self.hidden_size, layers,
                                      dropout=dropout, do_babi=self.do_load_babi,
                                      freeze_embedding=self.do_freeze_embedding, embedding=self.embedding_matrix,
                                      print_to_screen=self.do_print_to_screen)
        if hparams['cuda']: self.model_0_wra = self.model_0_wra.cuda()

        self.first_load = True
        self.load_checkpoint()
        lr = hparams['learning_rate']
        self.train_iters(None, None, len(self.pairs), print_every=self.print_every, learning_rate=lr)


if __name__ == '__main__':

    n = NMT()

    if not n.do_review and not n.do_load_babi:
        n.task_normal_train()
    elif not n.do_load_babi:
        n.task_review_set()
    elif n.do_load_babi and not n.do_test_not_train:
        n.task_babi_files()
    elif n.do_load_babi and n.do_test_not_train:
        n.task_babi_test_files()
        print('load test set -- no training.')
        print(n.train_fr)

    n.input_lang, n.output_lang, n.pairs = n.prepareData(n.train_fr, n.train_to,lang3=n.train_ques, reverse=False,
                                                         omit_unk=n.do_hide_unk)


    if n.do_load_babi:
        hparams['num_vocab_total'] = n.output_lang.n_words

    layers = hparams['layers']
    dropout = hparams['dropout']
    pytorch_embed_size = hparams['pytorch_embed_size']

    token_list = []
    if False:
        for i in word_lst: token_list.append(n.output_lang.word2index[i])

    n.model_0_wra = WrapMemRNN(n.vocab_lang.n_words, pytorch_embed_size, n.hidden_size,layers,
                               dropout=dropout, do_babi=n.do_load_babi, bad_token_lst=token_list,
                               freeze_embedding=n.do_freeze_embedding, embedding=n.embedding_matrix,
                               print_to_screen=n.do_print_to_screen)

    #print(n.model_0_wra)
    #n.set_dropout(0.1334)
    #exit()

    if hparams['cuda'] :n.model_0_wra = n.model_0_wra.cuda()

    if n.do_test_not_train and n.do_load_babi:
        print('test not train')
        n.setup_for_babi_test()
        exit()

    if n.do_train:
        lr = hparams['learning_rate']
        n.train_iters(None, None, len(n.pairs), print_every=n.print_every, learning_rate=lr)


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

