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
import datetime
import math
import argparse
import json
import cpuinfo
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
---------------------------

Some code is originally written by Wonjae Kim. This code includes 
some of the Pytorch models for processing input in the Encoder stage. 
He does not have a license file in his project repository.
The code can be found at the following site:

https://github.com/dandelin/Dynamic-memory-networks-plus-Pytorch

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

def encoding_positional(embedded_sentence, sum=False):

    _, slen, elen = embedded_sentence.size()

    slen2 = slen
    elen2 = elen

    if slen == 1 or elen == 1:
        exit()

    if slen == 1: slen2 += 0.01
    if elen == 1: elen2 += 0.01

    # print(slen, elen, 'slen,elen')

    l = [[(1 - s / (slen2 - 1)) - (e / (elen2 - 1)) * (1 - 2 * s / (slen2 - 1)) for e in range(elen)] for s in
         range(slen)]
    l = torch.FloatTensor(l)
    l = l.unsqueeze(0)  # for #batch
    # print(l.size(),"l", l)
    l = l.expand_as(embedded_sentence)
    if hparams['cuda'] is True: l = l.cuda()
    weighted = embedded_sentence * Variable(l)
    if sum:
        weighted = torch.sum(weighted, dim=1)
    return weighted

def prune_tensor( input, size):
    if isinstance(input, list): return input
    if input is None: return input
    n = 0
    while len(input.size()) < size:
        input = input.unsqueeze(0)
        n += 1
        if n > size + 1:
            break
    n = 0
    z = len(input.size()) + 1
    while len(input.size()) > size and input.size()[0] == 1:
        input = input.squeeze(0)
        n += 1
        if n > z:
            break
    return input

'''
###################################
'''

class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.3):
        super(CustomGRU, self).__init__()
        self.hidden_size = hidden_size
        self.Wr = nn.Linear(input_size, hidden_size)
        init.xavier_normal_(self.Wr.state_dict()['weight'])
        self.Ur = nn.Linear(hidden_size, hidden_size)
        init.xavier_normal_(self.Ur.state_dict()['weight'])
        self.W = nn.Linear(input_size, hidden_size)
        init.xavier_normal_(self.W.state_dict()['weight'])
        self.U = nn.Linear(hidden_size, hidden_size)
        init.xavier_normal_(self.U.state_dict()['weight'])

        self.Wz = nn.Linear(input_size, hidden_size)
        init.xavier_normal_(self.Wz.state_dict()['weight'])
        self.Uz = nn.Linear(hidden_size, hidden_size)
        init.xavier_normal_(self.Uz.state_dict()['weight'])

        self.dropout = nn.Dropout(dropout)

    def forward(self, fact, C, g=None):

        r = torch.sigmoid(self.Wr(fact) + self.Ur(C))
        h_tilda = torch.tanh(self.W(fact) + r * self.U(C))

        g = g.unsqueeze(0).expand_as(h_tilda)
        #print(g)
        #exit()

        zz = g * h_tilda + (1-g) * C

        return zz, zz

class EpisodicAttn(nn.Module):

    def __init__(self,  hidden_size, a_list_size=4, dropout=0.3):
        super(EpisodicAttn, self).__init__()

        self.hidden_size = hidden_size
        self.a_list_size = a_list_size
        self.batch_size = hparams['batch_size']
        self.c_list_z = None

        self.out_a = nn.Linear( a_list_size * hidden_size,hidden_size,bias=True)
        init.xavier_normal_(self.out_a.state_dict()['weight'])

        self.out_b = nn.Linear( hidden_size, 1, bias=True)
        init.xavier_normal_(self.out_b.state_dict()['weight'])

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

        self.c_list_z = concat_list

        self.c_list_z = self.dropout(self.c_list_z)
        l_1 = self.out_a(self.c_list_z)

        l_1 = torch.tanh(l_1) ## <---- this line? used to be tanh !!

        l_2 = self.out_b( l_1)

        self.G = l_2

        return self.G


class MemRNN(nn.Module):
    def __init__(self, hidden_size, dropout=0.3):
        super(MemRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout1 = nn.Dropout(dropout) # this is just for if 'nn.GRU' is used!!
        #self.gru = nn.GRU(hidden_size, hidden_size,dropout=0, num_layers=1, batch_first=False,bidirectional=False)
        self.gru = CustomGRU(hidden_size,hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        #print('reset')
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            #print('here...')
            weight.data.uniform_(-stdv, stdv)
            if len(weight.size()) > 1:
                init.xavier_normal_(weight)

    '''
    def prune_tensor(self, input, size):
        if len(input.size()) < size:
            input = input.unsqueeze(0)
        if len(input.size()) > size:
            input = input.squeeze(0)
        return input
    '''

    def forward(self, input, hidden=None, g=None):

        output, hidden_out = self.gru(input, hidden, g)

        return output, hidden_out

class SimpleInputEncoder(nn.Module):
    def __init__(self, source_vocab_size, embed_dim, hidden_dim,
                 n_layers, dropout=0.3, bidirectional=False, embedding=None, position=False, sum_bidirectional=True, batch_first=False):
        super(SimpleInputEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.position = position
        self.sum_bidirectional = sum_bidirectional
        self.embed = None
        self.gru = nn.GRU(embed_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=bidirectional, batch_first=batch_first)

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

        if embedding is not None:
            self.embed = embedding
            print('embedding encoder')

    def reset_parameters(self):
        #print('reset')
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            #print('here...')
            weight.data.uniform_(-stdv, stdv)
            if len(weight.size()) > 1:
                init.xavier_normal_(weight)

    def load_embedding(self, embedding):
        self.embed = embedding

    def test_embedding(self, num=None):
        if num is None:
            num = 55 # magic number for testing
        e = self.embed(num)
        print(e.size(), 'test embedding')
        print(e[0,0,0:10]) # print first ten values

    def sum_output(self, output):
        if self.bidirectional and self.sum_bidirectional:
            e1 = output[0, :, :self.hidden_dim]
            e2 = output[0, :, self.hidden_dim:]
            output = e1 + e2  #
            output = output.unsqueeze(0)
        return output

    def forward(self, source, hidden=None):

        #for s in source:
        if True:
            s = source

            embedded = self.embed(s)

            embedded = self.dropout(embedded)
            embedded = embedded.permute(1,0,2) ## batch first

            encoder_out, hidden = self.gru( embedded, hidden)

            encoder_out = self.sum_output(encoder_out)

        return encoder_out, hidden

class Encoder(nn.Module):
    def __init__(self, source_vocab_size, embed_dim, hidden_dim,
                 n_layers, dropout=0.3, bidirectional=False, embedding=None, position=False, sum_bidirectional=True, batch_first=False):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.position = position
        self.sum_bidirectional = sum_bidirectional
        self.embed = None
        self.gru = nn.GRU(embed_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=bidirectional, batch_first=batch_first)

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

        if embedding is not None:
            self.embed = embedding
            print('embedding encoder')

    def reset_parameters(self):
        #print('reset')
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            #print('here...')
            weight.data.uniform_(-stdv, stdv)
            if len(weight.size()) > 1:
                init.xavier_normal_(weight)

    def load_embedding(self, embedding):
        self.embed = embedding

    def test_embedding(self, num=None):
        if num is None:
            num = 55 # magic number for testing
        e = self.embed(num)
        print(e.size(), 'test embedding')
        print(e[0,0,0:10]) # print first ten values

    def sum_output(self, output):
        if self.bidirectional and self.sum_bidirectional:
            e1 = output[0, :, :self.hidden_dim]
            e2 = output[0, :, self.hidden_dim:]
            output = e1 + e2  #
            output = output.unsqueeze(0)
        return output

    def list_encoding(self, lst, hidden, permute_sentence=False):

        l = []

        for i in lst:
            #print(i)
            embedded = self.embed(i)
            #print(embedded.size(),'list')
            l.append(embedded.permute(1,0,2))
        embedded = torch.cat(l, dim=0) # dim=0

        #if len(l) == 1: permute_sentence=True
        #print(embedded.size(),'e0')

        embedded = encoding_positional(embedded)

        #embedded = self.position_encoding(embedded, permute_sentence=permute_sentence)
        #print(embedded.size(),'e1')

        embedded = torch.sum(embedded, dim=1)
        #print(embedded.size(),'e2')

        embedded = embedded.unsqueeze(0)
        embedded = self.dropout(embedded)

        #if hidden is not None: print(hidden.size())

        #hidden = None # Variable(torch.zeros(zz, slen, elen))
        encoder_out, encoder_hidden = self.gru(embedded, hidden)

        encoder_out = self.sum_output(encoder_out)
        #print(encoder_out.size(), 'e-out')
        #print(encoder_hidden.size(), 'e-hid')

        #print(encoder_out.size(),'list')
        return encoder_out, encoder_hidden

    def forward(self, source, hidden=None):

        if self.position :

            if isinstance(source, list):
                return self.list_encoding(source, hidden)
            else:
                print(source.size(),'src')
                exit()
                return self.list_encoding([source], hidden, permute_sentence=True)

        if isinstance(source, list):
            source = ' '.join(source)

        #print(source,'<-- src')

        embedded = self.embed(source)

        embedded = self.dropout(embedded)

        encoder_out, encoder_hidden = self.gru( embedded, hidden)

        encoder_out = self.sum_output(encoder_out)

        return encoder_out, encoder_hidden

class AnswerModule(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout=0.3, embed=None, recurrent_output=False, sol_token=0, cancel_attention=False):
        super(AnswerModule, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.batch_size = hparams['batch_size']
        self.recurrent_output = recurrent_output
        self.sol_token = sol_token
        self.decoder_layers = 1 # hparams['decoder_layers']
        self.cancel_attention = cancel_attention
        self.embed = embed
        out_size = vocab_size
        if recurrent_output: out_size = hidden_size

        self.out_a = nn.Linear(hidden_size * 2 , out_size, bias=True)
        init.xavier_normal_(self.out_a.state_dict()['weight'])

        #self.out_b1 = nn.Linear(hidden_size , hidden_size , bias=True)
        #init.xavier_normal_(self.out_b1.state_dict()['weight'])

        #self.out_b2 = nn.Linear(hidden_size , hidden_size , bias=True)
        #init.xavier_normal_(self.out_b2.state_dict()['weight'])

        self.out_c = nn.Linear(hidden_size , vocab_size, bias=True)
        init.xavier_normal_(self.out_c.state_dict()['weight'])

        self.out_d = nn.Linear(hidden_size, hidden_size * 2, bias=True)
        init.xavier_normal_(self.out_d.state_dict()['weight'])

        self.dropout   = nn.Dropout(dropout)
        self.dropout_b = nn.Dropout(dropout)
        self.dropout_c = nn.Dropout(dropout)
        self.dropout_d = nn.Dropout(dropout)
        self.maxtokens = hparams['tokens_per_sentence']

        self.decoder = nn.GRU(input_size=self.hidden_size, hidden_size=hidden_size, num_layers=self.decoder_layers, dropout=dropout, bidirectional=False, batch_first=True)

        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.decoder_layers)

        self.h0 = None
        self.c0 = None

        self.h0, self.c0 = self.init_hidden(self.decoder_layers)

    def init_hidden(self, batch_size):
        hidden = nn.Parameter(next(self.parameters()).data.new(self.decoder_layers, batch_size, self.hidden_size), requires_grad=False)
        cell = nn.Parameter(next(self.parameters()).data.new(self.decoder_layers, batch_size, self.hidden_size), requires_grad=False)
        return (hidden, cell)

    def reset_parameters(self):

        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():

            weight.data.uniform_(-stdv, stdv)
            if len(weight.size()) > 1:
                init.xavier_normal_(weight)

    def load_embedding(self, embed):

        self.embed = embed

    def recurrent(self, out, hid1=None):

        l, hid = out.size()

        all_out = []
        for k in range(l):

            e_out_list = [
                prune_tensor(out[k,:],2)
            ]

            while len(e_out_list) < self.decoder_layers:
                e_out_list.append(prune_tensor(out[k,:],2))

            e_out = torch.cat(e_out_list, dim=0)
            #e_out = F.softmax(e_out, dim=1)
            #e_out = self.dropout_c(e_out)

            outputs = []
            decoder_hidden = prune_tensor(e_out,3) #.permute(1,0,2)

            token = SOS_token #EOS_token

            if self.lstm is not None:
                decoder_hidden = decoder_hidden.permute(1,0,2)
                self.h0, _ = self.init_hidden(self.decoder_layers)

                #self.h0 =  nn.Parameter(decoder_hidden, requires_grad=False)
                self.c0 = nn.Parameter(decoder_hidden, requires_grad=False)
            ##############################################

            for i in range(self.maxtokens):

                output = self.embed(Variable(torch.tensor([token])))
                output = prune_tensor(output, 3)
                output = self.dropout_b(output)

                if self.lstm is not None:
                    output, (hn , cn) = self.lstm(output, (self.h0, self.c0))
                    #hn = self.dropout_d(hn)
                    #cn = self.dropout_c(cn)
                    self.h0 = nn.Parameter(hn, requires_grad=False)
                    self.c0 = nn.Parameter(cn, requires_grad=False)
                    #print(i, hn.size(), cn.size(),'hn,cn')
                    pass
                else:

                    output, decoder_hidden = self.decoder(output, decoder_hidden)

                output_x = self.out_c(output)

                output_x = self.dropout(output_x)

                #output_x = F.log_softmax(output_x, dim=2) ## log_softmax
                #output_x = self.dropout(output_x) ## <---

                outputs.append(output_x)

                token = torch.argmax(output_x, dim=2)

                if token == EOS_token:
                    for _ in range(i + 1, self.maxtokens):
                        out_early = Variable(torch.zeros((1,1,self.vocab_size)), requires_grad=False).detach()
                        #out_early = self.embed(Variable(torch.tensor([UNK_token])))
                        #out_early = prune_tensor(out_early, 3)
                        outputs.append(out_early)
                    #print(len(outputs))
                    break

            some_out = torch.cat(outputs, dim=0)

            some_out = prune_tensor(some_out, 3)

            all_out.append(some_out)

        val_out = torch.cat(all_out, dim=1)
        #val_out = F.softmax(val_out, dim=2)

        return val_out

    def forward(self, mem, question_h):

        question_h = F.relu(question_h)
        mem = F.relu(mem)
        #question_h = F.relu(question_h)

        mem_in = mem.permute(1,0,2)
        question_h = question_h.permute(1,0,2)

        mem_in = torch.cat([mem_in, question_h], dim=2)

        mem_in = self.dropout(mem_in)
        mem_in = mem_in.squeeze(0)

        out = self.out_a(mem_in)

        if self.recurrent_output:

            #out = F.tanh(out) ## <-- not this

            return self.recurrent(out, None)

        return out.permute(1,0)

#################### Wrapper ####################

class WrapMemRNN(nn.Module):
    def __init__(self,vocab_size, embed_dim,  hidden_size, n_layers, dropout=0.3, do_babi=True, bad_token_lst=[],
                 freeze_embedding=False, embedding=None, recurrent_output=False,print_to_screen=False,
                 cancel_attention=False, sol_token=0, simple_input=False):
        super(WrapMemRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.do_babi = do_babi
        self.print_to_screen = print_to_screen
        self.bad_token_lst = bad_token_lst
        self.embedding = embedding
        self.freeze_embedding = freeze_embedding
        self.teacher_forcing_ratio = hparams['teacher_forcing_ratio']
        self.recurrent_output = recurrent_output
        self.sol_token = sol_token
        self.cancel_attention = cancel_attention
        self.simple_input = simple_input
        position = hparams['split_sentences']
        gru_dropout = dropout #* 0.0 #0.5

        self.embed = nn.Embedding(vocab_size,hidden_size,padding_idx=1)

        self.model_1_enc = None

        if simple_input:
            self.model_1_enc = SimpleInputEncoder(vocab_size, embed_dim, hidden_size, 2, dropout=dropout,
                                   embedding=self.embed, bidirectional=True, position=position,
                                   batch_first=True)
        else:
            self.model_1_enc = Encoder(vocab_size, embed_dim, hidden_size, n_layers, dropout=dropout,
                                       embedding=self.embed, bidirectional=True, position=position,
                                       batch_first=True)

        self.model_2_enc = Encoder(vocab_size, embed_dim, hidden_size, n_layers, dropout=gru_dropout,
                                   embedding=self.embed, bidirectional=False, position=False, sum_bidirectional=False,
                                   batch_first=True)

        self.model_3_mem = MemRNN(hidden_size, dropout=dropout)
        self.model_4_att = EpisodicAttn(hidden_size, dropout=gru_dropout)
        self.model_5_ans = AnswerModule(vocab_size, hidden_size,dropout=dropout, embed=self.embed,
                                        recurrent_output=self.recurrent_output, sol_token=self.sol_token,
                                        cancel_attention=self.cancel_attention)

        self.next_mem = nn.Linear(hidden_size * 3, hidden_size)
        #init.xavier_normal_(self.next_mem.state_dict()['weight'])

        self.input_var = None  # for input
        #self.q_var = None  # for question
        self.answer_var = None  # for answer
        #self.q_q = None  # extra question
        self.q_q_last = None # question
        self.inp_c = None  # extra input
        self.inp_c_seq = None
        self.all_mem = None
        self.last_mem = None  # output of mem unit
        self.memory_list = None
        self.prediction = None  # final single word prediction
        self.memory_hops = hparams['babi_memory_hops']
        #self.inv_idx = torch.arange(100 - 1, -1, -1).long() ## inverse index for 100 values

        self.reset_parameters()

        if self.embedding is not None:
            self.load_embedding(self.embedding)
        self.share_embedding()

        if self.freeze_embedding or self.embedding is not None:
            self.wrap_freeze_embedding()
        #self.criterion = nn.CrossEntropyLoss()

        pass

    def load_embedding(self, embedding):
        #embedding = np.transpose(embedding,(1,0))
        e = torch.from_numpy(embedding)
        #e = e.permute(1,0)
        self.embed.weight.data.copy_(e) #torch.from_numpy(embedding))

    def share_embedding(self):
        self.model_1_enc.load_embedding(self.embed)
        self.model_2_enc.load_embedding(self.embed)
        self.model_5_ans.load_embedding(self.embed)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():

            weight.data.uniform_(-stdv, stdv)
            if len(weight.size()) > 1:
                init.xavier_normal_(weight)

        init.uniform_(self.embed.state_dict()['weight'], a=-(3**0.5), b=3**0.5)
        init.xavier_normal_(self.next_mem.state_dict()['weight'])

    def forward(self, input_variable, question_variable, target_variable, criterion=None):

        self.wrap_input_module(input_variable, question_variable)
        self.wrap_episodic_module()
        outputs,  ans = self.wrap_answer_module_simple()

        return outputs, None, ans, None

    def wrap_freeze_embedding(self, do_freeze=True):
        self.embed.weight.requires_grad = not do_freeze
        self.model_1_enc.embed.weight.requires_grad = not do_freeze # False
        self.model_2_enc.embed.weight.requires_grad = not do_freeze # False
        #self.model_5_ans.decoder.embed.weight.requires_grad = not do_freeze
        print('freeze embedding')
        pass

    def test_embedding(self, num=None):

        if num is None:
            num = 55  # magic number for testing = garden
        e = self.embed(num)
        print('encoder 0:')
        print(e.size(), 'test embedding')
        print(e[0, 0, 0:10])  # print first ten values

    def wrap_input_module(self, input_variable, question_variable):

        prev_h1 = []

        hidden1 = None
        for ii in input_variable:

            ii = prune_tensor(ii, 2)
            #print(ii, 'ii')
            out1, hidden1 = self.model_1_enc(ii, hidden1)
            #print(out1.size(),'out1')

            prev_h1.append(out1)


        self.inp_c_seq = prev_h1
        self.inp_c = prev_h1[-1]

        #prev_h2 = [None]
        prev_h3 = []

        for ii in question_variable:
            ii = prune_tensor(ii, 2)

            out2, hidden2 = self.model_2_enc(ii, None) #, prev_h2[-1])

            #prev_h2.append(out2)
            prev_h3.append(hidden2)

        #self.q_q = prev_h2[1:] # hidden2[:,-1,:]
        self.q_q_last = prev_h3
        #for i in self.q_q_last: print(i.size())
        #exit()
        return


    def wrap_episodic_module(self):
        if True:

            mem_list = []

            sequences = self.inp_c_seq

            self.memory_list = None
            positional_list = []


            for i in range(len(sequences)):

                #slot_list = [ self.q_q_last[i] for _ in range(self.memory_hops) ]

                z = self.q_q_last[i].clone()
                m_list = [z]

                #zz = self.prune_tensor(z.clone(), 3)
                zz = Variable(torch.zeros(1,1,self.hidden_size))

                index = -1
                for iter in range(self.memory_hops):

                    if len(m_list) is 1 : mem_last = m_list[index]
                    else: mem_last = F.relu(m_list[index])

                    x = self.wrap_attention_step(sequences[i], None, mem_last, self.q_q_last[i])

                    #print( x.size(), len(self.inp_c_seq),self.inp_c_seq[0].size(),'info')

                    e, _ = self.wrap_episode_small_step(sequences[i], x, zz, mem_last, self.q_q_last[i])

                    out = prune_tensor(e, 3)

                    m_list.append(out)

                mem_list.append(m_list[self.memory_hops])
                tmp = torch.cat(m_list, dim=1)
                #print(tmp.size(),'tmp')

                positional_list.append(tmp)

            mm_list = torch.cat(mem_list, dim=0)

            self.memory_list = torch.cat(positional_list, dim=0)
            #print(self.memory_list.size(),'pl')

            self.last_mem = mm_list

            if self.print_to_screen: print(self.last_mem,'lm')

        return None

    def wrap_episode_small_step(self, ct, g, prev_h, prev_mem=None, question=None):

        #assert len(ct.size()) == 3
        bat, sen, emb = ct.size()

        #print(sen,'sen')

        last = [prev_h]

        #ep = []
        for iii in range(sen):

            #index = 0 #- 1
            c = ct[0,iii,:].unsqueeze(0)

            ggg = g[iii,0,0]

            out, gru = self.model_3_mem(prune_tensor(c, 3), prune_tensor(last[iii], 3), ggg) # <<--- iii-1 or iii-0 ??

            last.append(gru) # gru <<--- this is supposed to be the hidden value

        #q_index = question.size()[1] - 1

        concat = [
            prune_tensor(prev_mem, 1),
            prune_tensor(out, 1),
            prune_tensor(question,1)
        ]
        #for i in concat: print(i.size())
        #exit()

        concat = torch.cat(concat, dim=0)
        h = self.next_mem(concat)
        #h = F.tanh(h)


        if self.recurrent_output and not hparams['split_sentences'] and False:
            #h = out
            pass

        return h, gru # h, gru



    def wrap_attention_step(self, ct, prev_g, mem, q_q):

        q_q = prune_tensor(q_q,3)
        mem = prune_tensor(mem,3)

        bat, sen, emb = ct.size()

        att = []
        for iii in range(sen):
            c = ct[0,iii,:]
            c = prune_tensor(c, 3)

            qq = prune_tensor(q_q, 3)


            concat_list = [
                #c,
                #mem,
                #qq,
                (c * qq),
                (c * mem),
                torch.abs(c - qq) ,
                torch.abs(c - mem)
            ]
            #for ii in concat_list: print(ii.size())
            #for ii in concat_list: print(ii)
            #exit()

            concat_list = torch.cat(concat_list, dim=2)

            att.append(concat_list)

        att = torch.cat(att, dim=0)

        z = self.model_4_att(att)

        z = F.softmax(z, dim=0) # <--- use this!!

        return z

    def wrap_answer_module_simple(self):
        #outputs

        #print(self.last_mem.size())
        q = self.q_q_last

        q_q = torch.cat(q, dim=0)
        q_q = prune_tensor(q_q, 3)

        mem = prune_tensor(self.last_mem, 3)

        ansx = self.model_5_ans(mem, q_q)

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
        self.embedding_matrix_is_loaded = False
        self.criterion = None

        self.best_loss = None
        self.long_term_loss = 0
        self.tag = ''

        self.input_lang = None
        self.output_lang = None
        self.question_lang = None
        self.vocab_lang = None

        self.print_every = hparams['steps_to_stats']
        self.epoch_length = 10000
        self.starting_epoch_length = self.epoch_length #10000
        self.epochs = hparams['epochs']
        self.hidden_size = hparams['units']
        self.start_epoch = 0
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
        self.time_num = time.time()
        self.time_str = ''
        self.time_elapsed_num = 0
        self.time_elapsed_str = ''

        self.print_control_num = 0

        ''' used by auto-stop function '''
        self.epochs_since_adjustment = 0
        self.lr_adjustment_num = 0
        self.lr_low = hparams['learning_rate']
        self.lr_increment = self.lr_low / 4.0
        self.best_accuracy = None
        self.best_accuracy_old = None
        self.best_accuracy_dict = {}
        self.best_accuracy_record_offset = 0
        self.record_threshold = 95.00
        self._recipe_switching = 0
        self._highest_validation_for_quit = 0
        self._count_epochs_for_quit = 0

        self.uniform_low = -1.0
        self.uniform_high = 1.0

        self.train_fr = None
        self.train_to = None
        self.train_ques = None
        self.pairs = []

        self.pairs_train = []
        self.pairs_valid = []

        self.blacklist = ['re', 've', 's', 't', 'll', 'm', 'don', 'd']

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
        self.do_sample_on_screen = True
        self.do_recurrent_output = False
        self.do_load_recurrent = False
        self.do_no_positional = False
        self.do_no_attention = False
        self.do_simple_input = False
        self.do_print_control = False
        self.do_skip_unk = False
        self.do_chatbot_train = False
        self.do_load_once = True

        self.printable = ''


        parser = argparse.ArgumentParser(description='Train some NMT values.')
        parser.add_argument('--mode', help='mode of operation. (train, infer, review, long, interactive, plot, chatbot)')
        parser.add_argument('--printable', help='a string to print during training for identification.')
        parser.add_argument('--basename', help='base filename to use if it is different from settings file.')
        parser.add_argument('--autoencode', help='enable auto encode from the command line with a ratio.')
        parser.add_argument('--train-all', help='(broken) enable training of the embeddings layer from the command line',
                            action='store_true')
        #parser.add_argument('--convert-weights',help='convert weights', action='store_true')
        parser.add_argument('--load-babi', help='Load three babi input files instead of chatbot data',
                            action='store_true')
        parser.add_argument('--load-recurrent',help='load files from "train.big" recurrent filenames', action='store_true')
        parser.add_argument('--hide-unk', help='hide all unk tokens', action='store_true')
        parser.add_argument('--skip-unk', help='do not use sentences with unknown words.', action='store_true')
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
        parser.add_argument('--lr-adjust', help='resume training at particular lr adjust value. (disabled)')
        parser.add_argument('--save-num', help='threshold for auto-saving files. (0-100)')
        parser.add_argument('--recipe-dropout', help='use dropout recipe', action='store_true')
        parser.add_argument('--recipe-lr', help='use learning rate recipe', action='store_true')
        parser.add_argument('--batch',help='enable batch processing. (default)',action='store_true')
        parser.add_argument('--batch-size', help='actual batch size when batch mode is specified.')
        parser.add_argument('--decay', help='weight decay.')
        parser.add_argument('--hops', help='babi memory hops.')
        parser.add_argument('--no-sample', help='Print no sample text on the screen.', action='store_true')
        parser.add_argument('--recurrent-output', help='use recurrent output module', action='store_true')
        parser.add_argument('--no-split-sentences', help='do not do positional encoding on input', action='store_true')
        parser.add_argument('--decoder-layers', help='number of layers in the recurrent output decoder (1 or 2)')
        parser.add_argument('--no-attention', help='disable attention if desired.', action='store_true')
        parser.add_argument('--simple-input', help='use simple input module. No positional encoding.', action='store_true')
        parser.add_argument('--print-control', help='set print control num to space out output.')
        parser.add_argument('--start-epoch', help='Starting epoch number if desired.')
        parser.add_argument('--json-record-offset', help='starting record number for json file')

        self.args = parser.parse_args()
        self.args = vars(self.args)
        # print(self.args)

        if self.args['printable'] is not None:
            self.printable = str(self.args['printable'])
        if self.args['mode'] is None or self.args['mode'] not in ['train', 'infer', 'review', 'long', 'interactive', 'plot']:
            self.args['mode'] = 'long'
        if self.args['mode'] == 'chatbot':
            self.do_chatbot_train = True
            self.do_train_long = True
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
        if self.args['recipe_dropout'] is not False:
            self.do_recipe_dropout = True
            self.do_auto_stop = True
        if self.args['recipe_lr'] is not False:
            self.do_recipe_lr = True
            self.do_auto_stop = True
        if self.args['batch'] is not False:
            self.do_batch_process = True
            print('batch operation now enabled by default.')
        if self.args['batch_size'] is not None: hparams['batch_size'] = int(self.args['batch_size'])
        if self.args['decay'] is not None: hparams['weight_decay'] = float(self.args['decay'])
        if self.args['hops'] is not None: hparams['babi_memory_hops'] = int(self.args['hops'])
        if self.args['no_sample'] is True: self.do_sample_on_screen = False
        if self.args['recurrent_output'] is True: self.do_recurrent_output = True
        if self.args['load_recurrent'] is True: self.do_load_recurrent = True
        if self.args['no_attention'] is not False: self.do_no_attention = True
        if self.args['no_split_sentences'] is True:
            self.do_no_positional = True
            hparams['split_sentences'] = False
        if self.args['decoder_layers'] is not None: hparams['decoder_layers'] = int(self.args['decoder_layers'])
        if self.args['start_epoch'] is not None: self.start_epoch = int(self.args['start_epoch'])
        if self.args['simple_input'] is True: self.do_simple_input = True
        if self.args['print_control'] is not None:
            self.do_print_control = True
            self.print_control_num = float(self.args['print_control'])
        if self.args['skip_unk'] is True: self.do_skip_unk = True
        if self.args['json_record_offset'] is not None:
            self.best_accuracy_record_offset = int(self.args['json_record_offset'])
        if self.printable == '': self.printable = hparams['base_filename']
        if hparams['cuda']: torch.set_default_tensor_type('torch.cuda.FloatTensor')

        ''' reset lr vars if changed from command line '''
        self.lr_low = hparams['learning_rate'] #/ 100.0
        self.lr_increment = self.lr_low / 4.0
        if self.args['lr_adjust'] is not None:
            self.lr_adjustment_num = int(self.args['lr_adjust'])
            hparams['learning_rate'] = self.lr_low + float(self.lr_adjustment_num) * self.lr_increment
        if not self.do_skip_unk and not self.do_hide_unk:
            #global blacklist
            self.blacklist = []

        self.read_json_file()

    def task_choose_files(self, mode=None):
        if mode is None:
            mode = 'train'
        if self.do_load_babi and not self.do_load_recurrent:
            if mode == 'train':
                self.task_babi_files()
                return
            if mode == 'valid':
                self.task_babi_valid_files()
                return
            if mode == 'test':
                self.task_babi_test_files()
                return
            pass
        if self.do_load_babi and self.do_load_recurrent:
            if mode == 'train':
                self.task_normal_train()
                return
            if mode == 'valid':
                self.task_normal_valid()
                return
            if mode == 'test':
                self.task_normal_test()
                return
            pass
        if True: # not self.do_load_babi and self.do_load_recurrent:
            if mode == 'train':
                self.task_normal_train()
                return
            if mode == 'valid':
                self.task_normal_valid()
                return
            if mode == 'test':
                self.task_normal_test()
                return
            pass

    def task_normal_train(self):
        self.train_fr = hparams['data_dir'] + hparams['train_name'] + '.' + hparams['src_ending']
        self.train_to = hparams['data_dir'] + hparams['train_name'] + '.' + hparams['tgt_ending']
        self.train_ques = hparams['data_dir'] + hparams['train_name'] + '.' + hparams['question_ending']
        pass

    def task_normal_valid(self):
        self.train_fr = hparams['data_dir'] + hparams['valid_name'] + '.' + hparams['src_ending']
        self.train_to = hparams['data_dir'] + hparams['valid_name'] + '.' + hparams['tgt_ending']
        self.train_ques = hparams['data_dir'] + hparams['valid_name'] + '.' + hparams['question_ending']
        pass

    def task_normal_test(self):
        self.train_fr = hparams['data_dir'] + hparams['test_name'] + '.' + hparams['src_ending']
        self.train_to = hparams['data_dir'] + hparams['test_name'] + '.' + hparams['tgt_ending']
        self.train_ques = hparams['data_dir'] + hparams['test_name'] + '.' + hparams['question_ending']
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
        if self.embedding_matrix is not None and self.embedding_matrix_is_loaded:
            print('embedding already loaded.')
            return
            pass
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
        i = self.start_epoch

        while True:
            self.this_epoch = i
            self.printable = 'step #' + str(i+1)
            self.do_test_not_train = False
            #self.score = 0.0

            self.input_lang, self.output_lang, self.pairs = self.prepareData(self.train_fr, self.train_to,
                                                                             lang3=self.train_ques, reverse=False,
                                                                             omit_unk=self.do_hide_unk)
            #self.model_0_wra.test_embedding()

            #self.first_load = True
            self.epoch_length = self.starting_epoch_length
            if self.epoch_length > len(self.pairs): self.epoch_length = len(self.pairs) - 1

            self.train_iters(None, None, self.epoch_length, print_every=self.print_every, learning_rate=lr)
            self.start = 0

            print('auto save.')
            print('%.2f' % self.score,'score')

            self.save_checkpoint(num=len(self.pairs))
            self.saved_files += 1
            self.validate_iters()
            self.start = 0
            #self.task_babi_files()
            mode = 'train'
            self.task_choose_files(mode=mode)
            if i % 3 == 0 and False: self._test_embedding(exit=False)
            if not self.do_sample_on_screen:
                print('-----')
                self._show_sample()
                print('-----')

            i += 1
            if i > num and not self.do_load_babi:
                break

        self.input_lang, self.output_lang, self.pairs = self.prepareData(self.train_fr, self.train_to,lang3=self.train_ques, reverse=False, omit_unk=self.do_hide_unk)

        self.update_result_file()
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

            l_in = self.open_sentences( lang1)
            l_out = self.open_sentences( lang2)
            if lang3 is not None:
                l_ques = self.open_sentences(lang3)

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
        if load_vocab_file is not None and self.vocab_lang is None:
            self.vocab_lang = Lang(load_vocab_file, limit=hparams['num_vocab_total'])
            print('vocab init.')
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
        ''' NOTE: pairs switch from train to embedding all the time. '''

        if self.do_load_once and len(self.pairs_train) is not 0 and len(self.pairs_valid) is not 0:
            return self.input_lang, self.output_lang, self.pairs

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
            #####
            if self.vocab_lang.n_words <= 3:
                v = self.open_sentences(self.vocab_lang.name)
                for word in v:
                    self.vocab_lang.addSentence(word.strip())
                    #print(word)
            #####
            self.input_lang = self.vocab_lang
            self.output_lang = self.vocab_lang

            skip_count = 0
            new_pairs = []
            for p in range(len(self.pairs)):
                #print(self.pairs[p])
                skip = False

                a = []
                b = []
                c = []
                if len(self.pairs[p][0].split(' ')) > hparams['tokens_per_sentence']: skip = True
                if len(self.pairs[p][1].split(' ')) > hparams['tokens_per_sentence']: skip = True
                if lang3 is not None:
                    if len(self.pairs[p][2].split(' ')) > hparams['tokens_per_sentence']: skip = True
                for word in self.pairs[p][0].split(' '):
                    if word in self.vocab_lang.word2index and word not in self.blacklist:
                        a.append(word)
                    elif not omit_unk or self.do_skip_unk:
                        a.append(hparams['unk'])
                        skip = True
                for word in self.pairs[p][1].split(' '):
                    if word in self.vocab_lang.word2index and word not in self.blacklist:
                        b.append(word)
                    elif not omit_unk or self.do_skip_unk:
                        b.append(hparams['unk'])
                        skip = True
                pairs = [' '.join(a), ' '.join(b)]
                if lang3 is not None:
                    for word in self.pairs[p][2].split(' '):
                        if word in self.vocab_lang.word2index and word not in self.blacklist:
                            c.append(word)
                        elif not omit_unk or self.do_skip_unk:
                            c.append(hparams['unk'])
                            skip = True
                    pairs.append( ' '.join(c) )
                if skip is False or not self.do_skip_unk:
                    new_pairs.append(pairs)
                else:
                    skip_count += 1
            self.pairs = new_pairs

        else:
            for pair in self.pairs:
                self.input_lang.addSentence(pair[0])
                self.output_lang.addSentence(pair[1])

        print("Counted words:")
        print(self.input_lang.name, self.input_lang.n_words)
        print(self.output_lang.name, self.output_lang.n_words)
        print('skip count', skip_count)

        if self.do_load_embeddings:
            print('embedding option detected.')
            self.task_set_embedding_matrix()

        if self.epoch_length > len(self.pairs):
            self.epoch_length = len(self.pairs)

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
        if self.do_recurrent_output and (len(sent) == 0 or sent[-1] != EOS_token):
            #sent.append(EOS_token)
            #print(sent,'<===')
            pass
        if len(sent) == 0: sent.append(0)
        if self.do_load_recurrent:
            sent = sent[:MAX_LENGTH]
        return sent

        #return [lang.word2index[word] for word in sentence.split(' ')]

    def variables_for_batch(self, pairs, size, start):
        if start + size >= len(pairs) and start < len(pairs) - 1:
            size = len(pairs) - start #- 1
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
            if not hparams['split_sentences']:
                g1.append(g[0].squeeze(1))
            else:
                g1.append(g[0]) ## put every word in it's own list??

            g2.append(g[1].squeeze(1))
            if self.do_recurrent_output:
                g3.append(g[2].squeeze(1))
            else:
                #print(g[2][0],g[2], 'target', len(g[2]),self.input_lang.index2word[g[2][0].item()])
                g3.append(g[2][0])

        return (g1, g2, g3)

    def variableFromSentence(self,lang, sentence, add_eol=False, pad=0):
        indexes = self.indexesFromSentence(lang, sentence)
        if add_eol and len(indexes) < pad: indexes.append(EOS_token)
        sentence_len = len(indexes)
        while pad > sentence_len:
            indexes.append(UNK_token)
            pad -= 1
        result = Variable(torch.LongTensor(indexes).unsqueeze(1))#.view(-1, 1))
        #print(result.size(),'r')
        if hparams['cuda']:
            return result.cuda()
        else:
            return result

    def variablesFromPair(self,pair):
        pad = hparams['tokens_per_sentence']
        if hparams['split_sentences'] and not self.do_simple_input:
            l = pair[0].strip().split('.')
            sen = []
            max_len = 0
            for i in range(len(l)):
                if len(l[i].strip().split(' ')) > max_len: max_len =  len(l[i].strip().split(' '))
            for i in range(len(l)):
                if len(l[i]) > 0:
                    #line = l[i].strip()
                    l[i] = l[i].strip()
                    while len(l[i].strip().split(' ')) < max_len:
                        l[i] += " " + hparams['unk']
                    #print(l[i],',l')
                    add_eol = True
                    z = self.variableFromSentence(self.input_lang, l[i], add_eol=add_eol)
                    #print(z)
                    sen.append(z)
            #sen = torch.cat(sen, dim=0)
            #for i in sen: print(i.size())
            #print('---')
            input_variable = sen
            pass
        else:
            #pad = 1
            add_eol = True
            input_variable = self.variableFromSentence(self.input_lang, pair[0], pad=pad, add_eol=add_eol)
        question_variable = self.variableFromSentence(self.output_lang, pair[1], add_eol=True, pad=pad)

        if len(pair) > 2 or self.do_recurrent_output:
            #print(pair[2],',pair')
            #if (len(pair[2]) > 0) or True:
            pad = 0
            add_eol = False
            if self.do_recurrent_output:
                pad = hparams['tokens_per_sentence']
                add_eol = True
            target_variable = self.variableFromSentence(self.output_lang, pair[2],
                                                        add_eol=add_eol,
                                                        pad=pad)
            #print(target_variable, 'tv')
            if self.do_recurrent_output:
                target_variable = target_variable.unsqueeze(0)

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
        if state is None or True:
            state = self.make_state(converted=converted)
            if converted: print(converted, 'is converted.')
        basename = hparams['save_dir'] + hparams['base_filename']
        if self.do_load_babi or self.do_conserve_space or True:
            num = self.this_epoch * len(self.pairs) + num
            torch.save(state,basename+ '.best.pth')
            print('save', basename)
            #####
            if self.do_test_not_train:
                self.best_accuracy_dict[str((self.best_accuracy_record_offset + self.saved_files) * self.starting_epoch_length)] = str(self.score)
                print('offset',self.best_accuracy_record_offset, ', epoch', self.this_epoch)
                self.update_json_file()
            #####
            #if self.do_test_not_train: self.score_list.append('%.2f' % self.score)
            if ((self.best_accuracy_old is None and self.best_accuracy is not None) or
                    (self.best_accuracy_old is not None and self.best_accuracy >= self.best_accuracy_old)):
                update = basename + '.' + str(int(math.floor(self.best_accuracy * 100))) + '.best.pth'
                if os.path.isfile(update):
                    os.remove(update)
                torch.save(state, update)
                self.best_accuracy_old = self.best_accuracy
            return
        torch.save(state, basename + extra + '.' + str(num)+ '.pth')
        if is_best:
            os.system('cp '+ basename + extra +  '.' + str(num) + '.pth' + ' '  +
                      basename + '.best.pth')

    def move_high_checkpoint(self):
        basename = hparams['save_dir'] + hparams['base_filename']
        update = basename + '.' + str(int(math.floor(10000))) + '.best.pth'
        if os.path.isfile(update):
            os.system('cp ' + update + ' ' + basename + '.best.pth')

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

                if self.do_load_embeddings:
                    self.model_0_wra.load_embedding(self.embedding_matrix)
                    self.embedding_matrix_is_loaded = True
                if self.do_freeze_embedding:
                    self.model_0_wra.wrap_freeze_embedding()
                else:
                    self.model_0_wra.wrap_freeze_embedding(do_freeze=False)
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
        return optim.Adam(parameters, lr=hparams['learning_rate'],weight_decay=hparams['weight_decay'])
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

        if self._highest_reached_test(num=20, goal=10, threshold=threshold):
            time.ctime()
            t = time.strftime('%l:%M%p %Z on %b %d, %Y')
            print(t)
            print('no progress')
            print('list:', self.score_list)
            self.update_result_file()
            exit()

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
                    float(self.score_list[-2]) == 100 or
                    (float(self.score_list[-2]) == 100 and float(self.score_list[-1]) == 100) or
                    (float(self.score_list[-2]) == float(self.score_list[-1]) and
                     float(self.score_list[-1]) != 0.0))):

                self.do_skip_validation = False

                ''' adjust learning_rate to different value if possible. -- validation '''

                if (False and len(self.score_list) > 3 and float(self.score_list[-2]) == 100.00 and
                        float(self.score_list[-3]) == 100.00 and float(self.score_list[-1]) != 100):
                    self.move_high_checkpoint()
                    time.ctime()
                    t = time.strftime('%l:%M%p %Z on %b %d, %Y')
                    print(t)
                    print('list:', self.score_list)
                    self.update_result_file()
                    exit()

                ''' put convergence test here. '''
                if self._convergence_test(10,lst=self.score_list_training):# or self._convergence_test(4, value=100.00):
                    time.ctime()
                    t = time.strftime('%l:%M%p %Z on %b %d, %Y')
                    print(t)
                    print('converge')
                    print('list:', self.score_list)
                    self.update_result_file()
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

            if (self.lr_adjustment_num > 25 or self.epochs_since_adjustment > 300) and (self.do_recipe_lr or self.do_recipe_dropout):
                print('max adjustments -- quit', self.lr_adjustment_num)
                self.update_result_file()
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

    def _convergence_test(self, num, lst=None, value=None):
        if lst is None:
            lst = self.score_list
        if len(lst) < num: return False

        if value is None:
            value = float(lst[-1])

        for i in lst[- num:]:
            if float(i) != value:
                return False
        return True

    def _highest_reached_test(self, num=0, lst=None, goal=0, threshold=0):
        ''' must only call this fn once !! '''
        if lst is None:
            lst = self.score_list
        if len(lst) == 0: return False
        val = float(lst[-1])
        if val < threshold: return False
        if val > self._highest_validation_for_quit and val > threshold:
            self._highest_validation_for_quit = val
            self._count_epochs_for_quit = 0
        else:
            self._count_epochs_for_quit += 1

        if num != 0 and self._count_epochs_for_quit >= num:
            return True
        if goal != 0 and self._count_epochs_for_quit >= goal and self._highest_validation_for_quit == 100.00:
            return True

        return False

    def _test_embedding(self, num=None, exit=True):
        if num is None:
            num = 'garden' #55 #hparams['unk']
        num = self.variableFromSentence(self.output_lang, str(num))
        print('\n',num)
        self.model_0_wra.test_embedding(num)
        if exit: exit()

    def _print_control(self, iter):
        if self.do_print_control:
            if iter == 0: return True
            if iter % self.print_control_num == 0: return True
        else:
            return True
        return False

    def _as_minutes(self,s):
        m = math.floor(s / 60)
        s -= m * 60
        if m >= 60:
            h = math.floor(m / 60)
            m -= h * 60
            return '%dh %dm %ds' % (h,m,s)
        return '%dm %ds' % (m, s)

    def _time_since(self, since):
        now = time.time()
        s = now - since
        #if percent == 0.0 : percent = 0.001
        #es = s / (percent)
        #rs = es - s
        return ' - %s' % (self._as_minutes(s))

    def _shorten(self, sentence, just_duplicates=False):
        # assume input is list already
        # get longest mutter possible!!
        # trim start!!!
        ll = ''
        out = []
        if just_duplicates:
            for x in sentence:
                if x == ll:
                    pass
                else:
                    if x != hparams['eol']: out.append(x)
                ll = x
            return ' '.join(out)

        ## shorten !!!

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

    def _pad_list(self, lst, val=None):
        if val is None: val = hparams['unk']
        width = 0
        for i in lst:
            if len(i) > width:
                width = len(lst)
        if width == 0:
            return lst
        z = []
        for i in lst:
            if len(i) < width:
                for j in range(width - len(i)):
                    i.append(val)
            z.append(i)
        return z

    def set_dropout(self, p):
        print('dropout',p)
        if self.model_0_wra is not None:
            self.model_0_wra.model_1_enc.dropout.p = p
            #self.model_0_wra.model_4_att.dropout.p = p
            self.model_0_wra.model_5_ans.dropout.p = p

    '''
    def prune_tensor(self, input, size):
        if isinstance(input, list): return input
        if input is None: return input
        while len(input.size()) < size:
            input = input.unsqueeze(0)
        while len(input.size()) > size:
            input = input.squeeze(0)
        return input
    '''
    #######################################

    def train(self,input_variable, target_variable,question_variable, encoder, decoder, wrapper_optimizer, decoder_optimizer, memory_optimizer, attention_optimizer, criterion, max_length=MAX_LENGTH):

        if criterion is not None:
            wrapper_optimizer.zero_grad()
            self.model_0_wra.train()
            outputs, _, ans, _ = self.model_0_wra(input_variable, question_variable, target_variable, criterion)

            loss = 0

            if self.do_recurrent_output :
                #print('do_rec_out')
                target_variable = torch.cat(target_variable, dim=0)
                ans = prune_tensor(ans, 3)

                ans = ans.float().permute(1,0,2).contiguous()
                '''
                ans = ans.view(-1, self.output_lang.n_words)
                target_variable = target_variable.view(-1)
                loss += criterion(ans, target_variable)


                '''
                for i in range(len(target_variable)):
                    #print(target_variable[i])
                    target_v = target_variable[i].squeeze(0).squeeze(1)
                    loss += criterion(ans[i,:, :], target_v)
                    #print(target_v,'tv')
                    #print(ans[i,:,:], 'ans')
                    #exit()

            elif self.do_batch_process:
                target_variable = torch.cat(target_variable,dim=0)
                ans = ans.permute(1,0)
            else:
                target_variable = target_variable[0]
                ans = torch.argmax(ans,dim=1)
                #ans = ans[0]



            if not self.do_recurrent_output:
                loss = criterion(ans, target_variable)

            loss.backward()

            if True:
                clip = 50.0
                _ = torch.nn.utils.clip_grad_norm_(self.model_0_wra.parameters(), clip)

            wrapper_optimizer.step()


        else:
            #self.model_0_wra.eval()
            with torch.no_grad():
                self.model_0_wra.eval()
                outputs, _, ans, _ = self.model_0_wra(input_variable, question_variable, target_variable,
                                                      criterion)

                if not self.do_recurrent_output:
                    loss = None
                    ans = ans.permute(1,0)

                else:
                    loss = None

                    ans = ans.float().permute(1, 0, 2).contiguous()

                    ans = prune_tensor(ans, 3)
                    #print(ans)



            #self._test_embedding()

        #ansx = ans
        if self.do_recurrent_output:
            #print(ans.size())
            if len(ans.size()) is not 2:
                ans = ans.view(-1, self.output_lang.n_words)
                #print(ans.size(),'redo with view')
                #exit()
            ans = ans.permute(1,0)
            #print(ans,ans.size(),'ans')

        return outputs, ans , loss

    #######################################

    def train_iters(self, encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.001):
        ''' laundry list of things that must be done every training or testing session. '''

        save_thresh = 2
        #self.saved_files = 0
        step = 1
        save_num = 0
        print_loss_total = 0  # Reset every print_every
        num_right = 0
        num_tot = 0
        num_right_small = 0
        num_count = 0
        temp_batch_size = hparams['batch_size'] #0

        epoch_len = self.epoch_length
        epoch_start = self.this_epoch * self.epoch_length
        if epoch_start >= len(self.pairs):
            n = (len(self.pairs) // self.epoch_length)
            if n is not 0: e = self.this_epoch % n
            else: e = 0
            epoch_start = e * self.epoch_length
            #exit()

        epoch_stop = epoch_start + self.epoch_length

        if len(self.pairs) < epoch_stop:
            epoch_stop = len(self.pairs)
            epoch_len = len(self.pairs) - epoch_start

        if not self.do_test_not_train :
            print('limit pairs:', len(self.pairs),
                  '- end of this epoch:',epoch_stop,
                  '- epochs:', len(self.pairs) // self.epoch_length,
                  '- this epoch:', self.this_epoch + 1)

        self.time_str = self._as_minutes(self.time_num)

        if self.opt_1 is None or self.first_load:

            wrapper_optimizer = self._make_optimizer()
            self.opt_1 = wrapper_optimizer

        if self.do_recurrent_output: # and False:
            weight = torch.ones(self.output_lang.n_words)
            weight[self.output_lang.word2index[hparams['unk']]] = 0.0
            self.criterion = nn.NLLLoss(weight=weight)
            #self.criterion = nn.MSELoss()
        else:
            weight = torch.ones(self.output_lang.n_words)
            weight[self.output_lang.word2index[hparams['unk']]] = 0.0
            self.criterion = nn.CrossEntropyLoss(weight=weight) #size_average=False)

        if not self.do_batch_process:
            training_pairs = [self.variablesFromPair(
                self.pairs[epoch_start:epoch_stop][i]) for i in range(epoch_len)] ## n_iters

        if not self.do_test_not_train:
            criterion = self.criterion
        else:
            criterion = None

        self.load_checkpoint()

        start = 1
        if self.do_load_babi  and not self.do_chatbot_train:
            self.start = 0

        if self.start != 0 and self.start is not None and not self.do_batch_process:
            start = self.start + 1

        if self.do_load_babi and self.do_test_not_train:

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

                #if self.do_recurrent_output:
                #    temp_batch_size = len(input_variable)# * hparams['tokens_per_sentence']

            elif self.do_batch_process:
                continue
                pass

            outputs, ans, l = self.train(input_variable, target_variable, question_variable, encoder,
                                            decoder, self.opt_1, None,
                                            None, None, criterion)
            num_count += 1

            if self.do_recurrent_output and self.do_load_babi: # and  self.do_sample_on_screen:
                ans = ans.permute(1,0)
                ans = torch.argmax(ans,dim=1)
                for ii in range(len(target_variable)):
                    for jj in range(target_variable[ii].size(1)): # target_variable[i].size()[1]):
                        #print(i, j, temp_batch_size)
                        t_val = target_variable[ii][0,jj,0].item()
                        #print(t_val, EOS_token)
                        #o_val = ans[i][j].item()
                        o_val = ans[ ii * target_variable[ii].size(1) + jj].item()
                        #print( ans.size(),'ans', ii * target_variable[ii].size(1) + jj, 'index')

                        if int(o_val) == int(t_val):
                            num_right += 1
                            num_right_small += 1
                            if int(o_val) == EOS_token:
                                num_right_small += hparams['tokens_per_sentence'] - (jj + 1)
                                num_right += hparams['tokens_per_sentence'] - (jj + 1)
                                #print('full line', i, j, num_right_small)
                                break
                        else:
                            # next sentence
                            if int(o_val) == EOS_token and int(t_val) == UNK_token and jj > 0:
                                num_right_small += hparams['tokens_per_sentence'] - (jj + 1)
                                num_right += hparams['tokens_per_sentence'] - (jj + 1)
                                break
                            break
                            pass
                    #print( len(target_variable), target_variable[i].size(1), ans[i].size() ,'tv, ans out')

                num_tot += temp_batch_size * hparams['tokens_per_sentence']


                self.score = float(num_right / num_tot) * 100

            if self.do_load_babi and not self.do_recurrent_output : #and  self.do_sample_on_screen:

                for i in range(len(target_variable)):
                    #print(ans[i].size())
                    o_val = torch.argmax(ans[i], dim=0).item() #[0]
                    t_val = target_variable[i].item()

                    if int(o_val) == int(t_val):
                        num_right += 1
                        num_right_small += 1

                if self.do_batch_process: num_tot += temp_batch_size
                else: num_tot += 1

                self.score = float(num_right/num_tot) * 100

            if l is not None:
                print_loss_total += float(l.clone())

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0

                if self._print_control(iter):
                    print(epoch_start ,'iter = '+str(iter)+ ', num of iters = '+str(n_iters) # +", countdown = "+ str(save_thresh - save_num)
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
                            if self._print_control(iter):
                                print('======= save file '+ extra+' ========')
                    elif not self.do_load_babi and self._print_control(iter):
                        print('skip save!')
                self.time_elapsed_str = self._time_since(self.time_num)
                self.time_elapsed_num = time.time()
                if self._print_control(iter):
                    print('(%d %d%%) %.6f loss' % (iter, iter / n_iters * 100, print_loss_avg),self.time_elapsed_str, end=' ')
                    if self.do_batch_process: print('- batch-size', temp_batch_size, end=' ')
                    if self.do_auto_stop: print('- count', self._count_epochs_for_quit)
                    else: print('')

                #print(epoch_start, iter, temp_batch_size, epoch_stop)

                if not self.do_skip_validation and self.do_sample_on_screen: # and temp_batch_size > 0 and epoch_start + iter < epoch_stop:
                    self._show_sample(iter,epoch_start, epoch_stop, temp_batch_size)
                    '''
                    ############################
                    '''
                if self.do_recurrent_output:
                    num_right_small = math.floor(num_right_small / (hparams['tokens_per_sentence'] ))
                    pass

                if self._print_control(iter):
                    if self.do_load_babi and self.do_test_not_train:

                        print('current accuracy: %.4f' % self.score, '- num right '+ str(num_right_small ))
                        num_right_small = 0

                    if self.do_load_babi and not self.do_test_not_train:

                        print('training accuracy: %.4f' % self.score, '- num right '+ str(num_right_small))
                        num_right_small = 0

                num_right_small = 0

                if self._print_control(iter):
                    if self.lr_adjustment_num > 0 and (self.do_recipe_dropout or self.do_recipe_lr):
                        if self._recipe_switching % 2 == 0 or not self.do_recipe_dropout:
                            print('[ lr adjust:', self.lr_adjustment_num, '-', hparams['learning_rate'],', epochs', self.epochs_since_adjustment ,']')
                        if self._recipe_switching % 2 == 1 or not self.do_recipe_lr:
                            print('[ dropout adjust:', self.lr_adjustment_num,'-', hparams['dropout'],', epochs',self.epochs_since_adjustment,']')

                    if self.score_list is not None and len(self.score_list_training) > 0 and len(self.score_list) > 0:
                        print('[ last train:', self.score_list_training[-1],']',end='')
                        if self.do_test_not_train:
                            print('[ older valid:', self.score_list[-1],']')
                        else:
                            print('[ last valid:', self.score_list[-1],']')
                    elif len(self.score_list_training) >= 1 and self.do_skip_validation:
                        print('[ last train:', self.score_list_training[-1],'][ no valid ]')

                    print("-----")

        if self.do_batch_process or not self.do_test_not_train:
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

        if self._print_control(iter):
            if self.do_load_babi:
                print('train list:', ', '.join(self.score_list_training))
                print('valid list:', ', '.join(self.score_list))
            print('dropout:',hparams['dropout'])
            print('learning rate:', hparams['learning_rate'])
            print('weight decay:', hparams['weight_decay'])
            print(num_count, 'exec count')
            print('raw score:', num_right, num_tot, num_right_small, len(self.pairs))

    def _show_sample(self, iter=0, epoch_start=0, epoch_stop=hparams['batch_size'], temp_batch_size=hparams['batch_size']):
        ###########################
        if not self._print_control(iter): return

        if epoch_start + iter >= epoch_stop:
            choice = random.choice(self.pairs)
        else:
            choice = random.choice(self.pairs[epoch_start + iter: epoch_start + iter + temp_batch_size])
        print('src:', choice[0])
        question = None
        if self.do_load_babi:
            print('ques:', choice[1])
            print('ref:', choice[2])
        else:
            print('tgt:', choice[1])
        nums = self.variablesFromPair(choice)
        if self.do_load_babi:
            question = nums[1]
            target = nums[2]

        if not self.do_load_babi:
            question = nums[0]
            target = None
        words, _ = self.evaluate(None, None, nums[0], question=question, target_variable=target)
        # print(choice)
        if not self.do_load_babi or self.do_recurrent_output:
            print('ans:', words)
            print('try:', self._shorten(words, just_duplicates=True))
            # self._word_from_prediction()
        ############################
        pass

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

        if True:
            if hparams['split_sentences'] is not True:
                input_variable = prune_tensor(input_variable,2)
                b, i = input_variable.size()
                if b > i and i == 1: input_variable = input_variable.permute(1,0)
                input_variable = prune_tensor(input_variable,1)
                input_variable = [input_variable]

                #input_variable = [input_variable.squeeze(0).squeeze(0).permute(1, 0).squeeze(0)]
                #print(input_variable[0].size(), 'size')
            else:
                input_variable = [input_variable]

            question_variable = prune_tensor(question_variable, 2)
            ql = len(question_variable.size())
            if ql == 2:
                b, i = question_variable.size()
                if b > i and i == 1: question_variable = question_variable.permute(1,0)
            question_variable = question_variable.squeeze(0)
            question_variable = [question_variable]

            #print(question_variable[0].size())

            #question_variable = [question_variable.squeeze(0).squeeze(0).permute(1, 0).squeeze(0)]

            sos_token = [sos_token.squeeze(0).squeeze(0).squeeze(0)]


        #print(question_variable.squeeze(0).squeeze(0).permute(1,0).squeeze(0).size(),'iv')

        self.model_0_wra.eval()
        with torch.no_grad():
            outputs, _, ans , _ = self.model_0_wra( input_variable, question_variable, sos_token, None)

        outputs = [ans]
        #####################

        if not self.do_recurrent_output:
            decoded_words = []
            for di in range(len(outputs)):

                output = outputs[di]
                output = output.permute(1,0)

                ni = torch.argmax(output, dim=1)[0]

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

        else:
            decoded_words = []
            for db in range(len(outputs)):
                for di in range(len(outputs[db])):
                    output = outputs[db][di]
                    output = output.permute(1, 0)
                    #print(output.size(),'out')
                    ni = torch.argmax(output, dim=0)[0]
                    #print(ni, 'ni')
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
        #self.task_babi_valid_files()
        if self.do_load_once:
            self.pairs_train = self.pairs
            if len(self.pairs_valid) > 0:
                self.pairs = self.pairs_valid
            if self.epoch_length > len(self.pairs):
                self.epoch_length = len(self.pairs)
            #print('load once', len(self.pairs))
        mode = 'valid'
        self.task_choose_files(mode=mode)
        self.printable = 'validate'
        self.input_lang, self.output_lang, self.pairs = self.prepareData(self.train_fr, self.train_to,lang3=self.train_ques, reverse=False, omit_unk=self.do_hide_unk)
        self.do_test_not_train = True
        self.first_load = True
        self.load_checkpoint()
        lr = hparams['learning_rate']
        self.start = 0
        self.train_iters(None,None, self.epoch_length, print_every=self.print_every, learning_rate=lr)
        if len(self.score_list) > 0 and float(self.score_list[-1]) >= self.record_threshold: #100.00:
            self.best_accuracy = float(self.score_list[-1])
            self.save_checkpoint(num=len(self.pairs))
        if self.do_load_once:
            self.pairs_valid = self.pairs
            self.pairs = self.pairs_train
        pass

    def setup_for_interactive(self):
        self.do_interactive = True
        self.input_lang, self.output_lang, self.pairs = self.prepareData(self.train_fr, self.train_to,lang3=self.train_ques, reverse=False, omit_unk=self.do_hide_unk)
        layers = hparams['layers']
        dropout = hparams['dropout']
        pytorch_embed_size = hparams['pytorch_embed_size']
        sol_token = self.output_lang.word2index[hparams['sol']]

        self.model_0_wra = WrapMemRNN(self.input_lang.n_words, pytorch_embed_size, self.hidden_size,layers,
                                      dropout=dropout,do_babi=self.do_load_babi,
                                      freeze_embedding=self.do_freeze_embedding, embedding=self.embedding_matrix,
                                      print_to_screen=self.do_print_to_screen, recurrent_output=self.do_recurrent_output,
                                      sol_token=sol_token, cancel_attention=self.do_no_attention,
                                      simple_input=self.do_simple_input)

        if hparams['cuda']: self.model_0_wra = self.model_0_wra.cuda()

        self.load_checkpoint()

    def setup_for_babi_test(self):
        #hparams['base_filename'] = filename
        self.printable = hparams['base_filename']

        self.do_test_not_train = True
        #self.task_babi_files()
        #self.task_babi_test_files()
        mode = 'test'
        self.task_choose_files(mode=mode)
        self.input_lang, self.output_lang, self.pairs = self.prepareData(self.train_fr, self.train_to,lang3=self.train_ques, reverse=False,
                                                                         omit_unk=self.do_hide_unk)
        hparams['num_vocab_total'] = self.output_lang.n_words

        layers = hparams['layers']
        dropout = hparams['dropout']
        pytorch_embed_size = hparams['pytorch_embed_size']
        sol_token = self.output_lang.word2index[hparams['sol']]

        self.model_0_wra = WrapMemRNN(self.input_lang.n_words, pytorch_embed_size, self.hidden_size, layers,
                                      dropout=dropout, do_babi=self.do_load_babi,
                                      freeze_embedding=self.do_freeze_embedding, embedding=self.embedding_matrix,
                                      print_to_screen=self.do_print_to_screen, recurrent_output=self.do_recurrent_output,
                                      sol_token=sol_token, cancel_attention=self.do_no_attention,
                                      simple_input=self.do_simple_input)

        if hparams['cuda']: self.model_0_wra = self.model_0_wra.cuda()

        self.first_load = True
        self.load_checkpoint()
        lr = hparams['learning_rate']
        self.train_iters(None, None, self.epoch_length, print_every=self.print_every, learning_rate=lr)

    def update_result_file(self):

        if self.do_freeze_embedding: self._test_embedding(exit=False)

        basename = hparams['save_dir'] + hparams['base_filename'] + '.txt'
        ts = time.time()
        st_now = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

        st_start = datetime.datetime.fromtimestamp(self.time_num).strftime('%Y-%m-%d %H:%M:%S')

        epoch_time = ts - self.time_num
        if self.saved_files > 0: epoch_time = epoch_time / self.saved_files

        if not os.path.isfile(basename):
            cpu_info = cpuinfo.get_cpu_info()['hz_advertised']

            with open(basename, 'w') as f:
                f.write(self.args['basename'] + '\n')
                f.write(str(hparams['units']) + ' hidden size \n')
                f.write(str(self.output_lang.n_words) + ' vocab size \n')
                f.write(cpu_info + ' \n\n')
                f.write('hparams \n')
                f.write(json.dumps(hparams) + '\n')
                f.write('------\n')
                f.write('commandline options \n')
                f.write(json.dumps(self.args) + '\n')
                f.write('\n')
            f.close()

        with open(basename,'a') as f:
            #f.write('\n')
            f.write('------\n')
            f.write('start time:     ' + st_start + '\n')
            f.write('quit/log time:  ' + st_now + '\n')
            f.write('elapsed time:   ' + self.time_elapsed_str + '\n')
            f.write('save count:     ' + str(self.saved_files) + '\n')
            f.write('time per epoch: ' + self._as_minutes(epoch_time) + '\n')
            f.write('this epoch/step ' + str(self.this_epoch) + '\n')
            f.write('train results:' + '\n')
            f.write(','.join(self.score_list_training))
            f.write('\n')
            f.write('valid results:' + '\n')
            f.write(','.join(self.score_list))
            f.write('\n')
            #f.write('\n')
        f.close()
        print('\nsee file:', basename, '\n')
        pass

    def update_json_file(self):
        basename = hparams['save_dir'] + hparams['base_filename'] + '.json'
        if len(self.best_accuracy_dict) > 0:
            with open(basename, 'w') as z:
                z.write(json.dumps(self.best_accuracy_dict))
            z.close()

    def read_json_file(self):
        basename = hparams['save_dir'] + hparams['base_filename'] + '.json'
        if os.path.isfile(basename):
            with open(basename) as z:
                json_data = json.load(z)
            self.best_accuracy_dict = json_data # json.loads(json_data)


if __name__ == '__main__':

    n = NMT()

    try:
        mode = ''
        if (not n.do_review and not n.do_load_babi) or n.do_load_recurrent:
            #n.task_normal_train()
            mode = 'train'
            n.task_choose_files(mode=mode)
        elif not n.do_load_babi:
            mode = 'test'
            n.task_choose_files(mode=mode)
            #n.task_review_set()
        elif n.do_load_babi and not n.do_test_not_train:
            mode = 'train'
            n.task_choose_files(mode=mode)
            #n.task_babi_files()
        elif n.do_load_babi and n.do_test_not_train:
            mode = 'test'
            n.task_choose_files(mode=mode)
            #n.task_babi_test_files()
            print('load test set -- no training.')
            print(n.train_fr)

        n.input_lang, n.output_lang, n.pairs = n.prepareData(n.train_fr, n.train_to,lang3=n.train_ques, reverse=False,
                                                             omit_unk=n.do_hide_unk)


        if n.do_load_babi:
            hparams['num_vocab_total'] = n.output_lang.n_words

        layers = hparams['layers']
        dropout = hparams['dropout']
        pytorch_embed_size = hparams['pytorch_embed_size']
        sol_token = n.output_lang.word2index[hparams['sol']]

        token_list = []
        if False:
            for i in word_lst: token_list.append(n.output_lang.word2index[i])

        n.model_0_wra = WrapMemRNN(n.vocab_lang.n_words, pytorch_embed_size, n.hidden_size,layers,
                                   dropout=dropout, do_babi=n.do_load_babi, bad_token_lst=token_list,
                                   freeze_embedding=n.do_freeze_embedding, embedding=n.embedding_matrix,
                                   print_to_screen=n.do_print_to_screen, recurrent_output=n.do_recurrent_output,
                                   sol_token=sol_token, cancel_attention=n.do_no_attention,
                                   simple_input=n.do_simple_input)

        #print(n.model_0_wra)
        #n.set_dropout(0.1334)
        #exit()

        if hparams['cuda'] :n.model_0_wra = n.model_0_wra.cuda()

        #n._test_embedding()

        if n.do_test_not_train and n.do_load_babi:
            print('test not train')
            n.setup_for_babi_test()
            exit()

        if n.do_train:
            lr = hparams['learning_rate']
            n.train_iters(None, None, n.epoch_length, print_every=n.print_every, learning_rate=lr)


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

    except KeyboardInterrupt:
        n.update_result_file()

