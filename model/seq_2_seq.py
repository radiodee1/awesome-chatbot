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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

---------------------------

Some code came from a Pytorch tutorial on creating a chatbot by Mathew Inkawhich.
Specifically the code for the Attention Mechanism was used.
It can be found at the following site:

https://pytorch.org/tutorials/beginner/chatbot_tutorial.html

'''

use_cuda = torch.cuda.is_available()
UNK_token = 0
SOS_token = 1
EOS_token = 2
MAX_LENGTH = hparams['tokens_per_sentence']

#hparams['teacher_forcing_ratio'] = 0.0
teacher_forcing_ratio = hparams['teacher_forcing_ratio'] #0.5
hparams['layers'] = 2
hparams['pytorch_embed_size'] = hparams['units']
#hparams['dropout'] = 0.3

word_lst = ['.', ',', '!', '?', "'", hparams['unk']]

blacklist = ['re', 've', 's', 't', 'll', 'm', 'don', 'd']

def plot_vector(vec):
    fig, ax = plt.subplots()
    lst_x = []
    lst_y = []
    vec = prune_tensor(vec, 1)

    for i in range(len(vec)):
        lst_x.append(i)
        lst_y.append(vec[i].item())
    ax.plot(lst_x, lst_y, 'b' + '-')
    plt.show()
    pass


def encoding_positional(embedded_sentence, sum=False):

    _, slen, elen = embedded_sentence.size()

    slen2 = slen
    elen2 = elen

    if slen == 1 or elen == 1:
        #exit()
        pass

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

################# pytorch modules ###############

class Encoder(nn.Module):
    def __init__(self, source_vocab_size, embed_dim, hidden_dim, n_layers, dropout, embed=None):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed = embed# nn.Embedding(source_vocab_size, embed_dim, padding_idx=1)
        self.sum_encoder = True
        self.gru = nn.GRU(embed_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def load_embedding(self, embedding):
        self.embed = embedding

    def forward(self, source, hidden=None):
        embedded = self.embed(source)  # (batch_size, seq_len, embed_dim)
        embedded = self.dropout(embedded)
        #print(embedded.size(),'emb')
        encoder_out, encoder_hidden = self.gru(embedded, hidden)  # (seq_len, batch, hidden_dim*2)
        # sum bidirectional outputs, the other option is to retain concat features
        if self.sum_encoder:
            encoder_out = (encoder_out[:, :, :self.hidden_dim] +
                           encoder_out[:, :, self.hidden_dim:])
        else:
            encoder_out = torch.cat(
                [
                    encoder_out[1, :, :self.hidden_dim],
                    encoder_out[1, :, self.hidden_dim:]
                ],
                dim=1
            )
        return encoder_out, encoder_hidden


class Attn(torch.nn.Module):
    def __init__(self,  hidden_size):
        method = 'general'
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        #print(encoder_output.size(),'eo')
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        #attn_energies = attn_energies.t()
        #print(attn_energies.size(),'att', attn_energies)
        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=0).unsqueeze(1)

class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, hidden_dim, n_layers, dropout, embed=None, cancel_attention=False):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.embed = embed # nn.Embedding(target_vocab_size, embed_dim, padding_idx=1)
        self.attention = Attn(hidden_dim)

        gru_in_dim = hidden_dim
        linear_in_dim = hidden_dim * 2
        if cancel_attention:
            gru_in_dim = hidden_dim
            linear_in_dim = hidden_dim

        self.gru = nn.GRU(gru_in_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.out = nn.Linear(hidden_dim, target_vocab_size)

        self.concat_out = nn.Linear(linear_in_dim, target_vocab_size)
        self.maxtokens = hparams['tokens_per_sentence']
        self.cancel_attention = cancel_attention
        self.decoder_hidden_z = None
        self.dropout_o = nn.Dropout(dropout)
        self.dropout_e = nn.Dropout(dropout)

    def load_embedding(self, embedding):
        self.embed = embedding



    def forward(self, encoder_out, decoder_hidden, encoder_output=None):

        l, s, hid = encoder_out.size()

        all_out = []
        for k in range(l):

            outputs = []

            decoder_hidden_x = prune_tensor(decoder_hidden[k,:,:],3)

            encoder_out_x = prune_tensor(encoder_out[k,:,:],3)

            token = EOS_token

            for i in range(self.maxtokens):
                output = Variable(torch.LongTensor([token]))
                output = prune_tensor(output, 3)

                output, decoder_hidden_x, mask, out_x = self.new_inner(
                    output,
                    encoder_out_x[:,i,:].unsqueeze(0),
                    decoder_hidden_x
                )

                output = prune_tensor(output, 3)
                outputs.append(output)

                token = torch.argmax(output, dim=2)

            some_out = torch.cat(outputs, dim=0)
            all_out.append(some_out)

        val_out = torch.cat(all_out, dim=1)

        return val_out

    def new_inner(self, output, encoder_out, decoder_hidden):

        embedded = self.embed(output)
        embedded = prune_tensor(embedded, 3)
        embedded = self.dropout_e(embedded)

        ## CHANGE HIDDEN STATE HERE ##
        decoder_hidden = decoder_hidden[:, : self.n_layers].permute(1, 0, 2)

        rnn_output, decoder_hidden = self.gru(embedded, decoder_hidden)

        if not self.cancel_attention:

            mask = None
            context = self.attention(rnn_output, encoder_out)

            context = prune_tensor(context[0,0,:],3)

            #print(context.size(), encoder_out.size(),'c,eo')

            context = context.bmm(encoder_out.transpose(0,1))

            concat_list = [
                rnn_output,
                context
            ]
            #for i in concat_list: print(i.size(), self.n_layers)
            #exit()

            attn_out = torch.cat(concat_list, dim=2)  # dim=2
            attn_out = self.concat_out(attn_out)
            attn_out = self.dropout_o(attn_out)

            out_x = attn_out

            output = out_x.clone()
        else:
            context = None
            mask = None

            rnn_output = self.concat_out(rnn_output)
            rnn_output = self.dropout_o(rnn_output)
            out_x = rnn_output # torch.tanh(rnn_output)

            output = out_x.clone()

        decoder_hidden = decoder_hidden.contiguous()
        decoder_hidden = decoder_hidden.permute(1,0,2)

        return output, decoder_hidden, mask, out_x


#################### Wrapper ####################

class WrapMemRNN(nn.Module):
    def __init__(self,vocab_size, embed_dim,  hidden_size, n_layers, dropout=0.3, do_babi=True, bad_token_lst=[],
                 freeze_embedding=False, embedding=None, recurrent_output=False,print_to_screen=False, sol_token=0,
                 cancel_attention=False, freeze_encoder=False, freeze_decoder=False):

        super(WrapMemRNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.do_babi = do_babi
        self.print_to_screen = print_to_screen
        self.bad_token_lst = bad_token_lst
        self.embedding = embedding
        self.freeze_embedding = freeze_embedding
        self.teacher_forcing_ratio = hparams['teacher_forcing_ratio']
        self.recurrent_output = True #recurrent_output
        self.sol_token = sol_token
        position = hparams['split_sentences']
        gru_dropout = dropout * 0.0 #0.5
        self.cancel_attention = cancel_attention

        self.embed = nn.Embedding(vocab_size,hidden_size,padding_idx=1)

        self.model_1_seq = Encoder(vocab_size,embed_dim, hidden_size,
                                          2, dropout,embed=self.embed)

        self.model_6_dec = Decoder(vocab_size, embed_dim, hidden_size,2, dropout, self.embed,
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
        self.prediction = None  # final single word prediction
        self.memory_hops = hparams['babi_memory_hops']
        #self.inv_idx = torch.arange(100 - 1, -1, -1).long() ## inverse index for 100 values

        self.reset_parameters()

        if self.embedding is not None:
            self.load_embedding(self.embedding)
        self.share_embedding()

        if self.freeze_embedding or self.embedding is not None:
            self.new_freeze_embedding()

        if freeze_encoder:
            self.new_freeze_encoding()
        if freeze_decoder:
            self.new_freeze_decoding()
        pass

    def load_embedding(self, embedding):
        #embedding = np.transpose(embedding,(1,0))
        e = torch.from_numpy(embedding)
        #e = e.permute(1,0)
        self.embed.weight.data.copy_(e) #torch.from_numpy(embedding))

    def share_embedding(self):

        self.model_1_seq.load_embedding(self.embed)
        self.model_6_dec.load_embedding(self.embed)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():

            weight.data.uniform_(-stdv, stdv)
            if len(weight.size()) > 1:
                init.xavier_normal_(weight)

        init.uniform_(self.embed.state_dict()['weight'], a=-(3**0.5), b=3**0.5)
        init.xavier_normal_(self.next_mem.state_dict()['weight'])

    def forward(self, input_variable, question_variable, target_variable, criterion=None):

        question_variable, hidden = self.wrap_question_module(input_variable)

        question_variable = prune_tensor(question_variable,3)
        hidden = prune_tensor(hidden,3)

        if self.print_to_screen:
            ''' here we test the plot_vector() function. '''
            print(question_variable.size(),'qv')
            out = prune_tensor(question_variable[0][0], 1)
            plot_vector(out)
            exit()

        ans = self.model_6_dec(question_variable, hidden)

        outputs = None

        return outputs, None, ans, None

    def new_freeze_embedding(self, do_freeze=True):
        self.embed.weight.requires_grad = not do_freeze
        self.model_1_seq.embed.weight.requires_grad = not do_freeze
        self.model_6_dec.embed.weight.requires_grad = not do_freeze
        if do_freeze: print('freeze embedding')
        pass

    def new_freeze_decoding(self, do_freeze=True):
        for weight in self.model_6_dec.parameters():
            weight.requires_grad = not do_freeze

        if do_freeze: print('freeze decoding')
        pass

    def new_freeze_encoding(self, do_freeze=True):
        for weight in self.model_1_seq.parameters():
            weight.requires_grad = not do_freeze

        if do_freeze: print('freeze encoding')
        pass

    def test_embedding(self, num=None):

        if num is None:
            num = 55  # magic number for testing = garden
        e = self.embed(num)
        print('encoder 0:')
        print(e.size(), 'test embedding')
        print(e[0, 0, 0:10])  # print first ten values

    def wrap_question_module(self, question_variable):

        prev_h2 = []
        prev_h3 = []

        for ii in question_variable:

            ii = ii.squeeze(1)
            ii = prune_tensor(ii, 2)

            out2, hidden2 = self.model_1_seq(ii, None) #, prev_h2[-1])

            prev_h2.append(prune_tensor(out2,3))

            #h = None
            h = Variable(torch.zeros([4, 1, self.hidden_size]))

            #print(ii.size(),'ii')

            if ii.size(1) > 0:
                h = prune_tensor(hidden2[:,-1,:], 3)

            #print(h.size(),'h')
            prev_h3.append(h)

        prev_h2 = torch.cat(prev_h2, dim=0)
        prev_h3 = torch.cat(prev_h3, dim=0)

        return prev_h2, prev_h3




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
        self.best_loss_graph = 0
        self.tag = ''

        self.input_lang = None
        self.output_lang = None
        self.question_lang = None
        self.vocab_lang = None

        self.print_every = hparams['steps_to_stats']
        self.epoch_length = 1000
        self.starting_epoch_length = self.epoch_length #1000
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
        self._skipped = 0

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
        self.best_accuracy_graph_size = self.epoch_length
        self.best_loss_old = None
        self.best_loss_dict = {}
        self.best_loss_record_offset = 0
        self.best_loss_graph_size = self.epoch_length
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

        self.do_train = False
        self.do_infer = False
        self.do_review = False
        self.do_train_long = False
        self.do_interactive = False
        self.do_convert = False
        self.do_plot = False
        self.do_load_babi = True #False
        self.do_hide_unk = False
        self.do_conserve_space = False
        self.do_test_not_train = False
        self.do_freeze_embedding = False
        self.do_freeze_decoding = False
        self.do_freeze_encoding = False
        self.do_load_embeddings = False
        self.do_auto_stop = False
        self.do_skip_validation = False
        self.do_print_to_screen = False
        self.do_recipe_dropout = False
        self.do_recipe_lr = False
        self.do_batch_process = True
        self.do_sample_on_screen = True
        self.do_recurrent_output = True
        self.do_load_recurrent = False
        self.do_no_positional = False
        self.do_no_attention = False
        self.do_skip_unk = False
        self.do_autoencode_words = False
        self.do_record_loss = False
        self.do_print_control = False
        self.do_load_once = True

        self.do_clip_grad_norm = True

        self.printable = ''

        parser = argparse.ArgumentParser(description='Train some NMT values.')
        parser.add_argument('--mode', help='mode of operation. (train, infer, review, long, interactive, plot)')
        parser.add_argument('--printable', help='a string to print during training for identification.')
        parser.add_argument('--basename', help='base filename to use if it is different from settings file.')
        parser.add_argument('--autoencode', help='enable auto encode from the command line with a ratio.')
        parser.add_argument('--autoencode-words', help='enable auto encode on a word to word basis.', action='store_true')
        parser.add_argument('--train-all', help='(broken) enable training of the embeddings layer from the command line',
                            action='store_true')
        #parser.add_argument('--convert-weights',help='convert weights', action='store_true')
        parser.add_argument('--load-babi', help='Load three babi input files instead of two chatbot data files. (default)',
                            action='store_true')
        parser.add_argument('--load-recurrent',help='load files from "train.big" recurrent filenames', action='store_true')
        parser.add_argument('--hide-unk', help='hide all unk tokens', action='store_true')
        parser.add_argument('--skip-unk',help='skip input that contains the unk token', action='store_true')
        parser.add_argument('--use-filename', help='use base filename as basename for saved weights.', action='store_true')
        parser.add_argument('--conserve-space', help='save only one file for all training epochs.',
                            action='store_true')
        parser.add_argument('--babi-num', help='number of which babi test set is being worked on')
        parser.add_argument('--units', help='Override UNITS hyper parameter.')
        parser.add_argument('--test',help='Disable all training. No weights will be changed and no new weights will be saved.',
                            action='store_true')
        parser.add_argument('--lr', help='learning rate')
        parser.add_argument('--freeze-embedding', help='Stop training on embedding elements',action='store_true')
        parser.add_argument('--freeze-decoding', help='Stop training on decoder', action='store_true')
        parser.add_argument('--freeze-encoding', help='Stop training on encoder', action='store_true')
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
        parser.add_argument('--recurrent-output', help='use recurrent output module. (default)', action='store_true')
        parser.add_argument('--no-split-sentences', help='do not do positional encoding on input', action='store_true')
        parser.add_argument('--decoder-layers', help='number of layers in the recurrent output decoder (1 or 2)')
        parser.add_argument('--start-epoch', help='Starting epoch number if desired.')
        parser.add_argument('--print-control', help='set print control num to space out output.')
        parser.add_argument('--no-attention', help='disable attention if desired.', action='store_true')
        parser.add_argument('--json-record-offset', help='starting record number for json file')
        parser.add_argument('--no-vocab-limit', help='no vocabulary size limit.', action='store_true')
        parser.add_argument('--record-loss', help='record loss for later graphing.', action='store_true')

        self.args = parser.parse_args()
        self.args = vars(self.args)
        # print(self.args)

        if self.args['printable'] is not None:
            self.printable = str(self.args['printable'])
        if self.args['mode'] is None or self.args['mode'] not in ['train', 'infer', 'review', 'long', 'interactive', 'plot']:
            self.args['mode'] = 'long'
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
        if self.args['autoencode_words'] is not False: self.do_autoencode_words = True
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
        if self.args['freeze_encoding'] == True: self.do_freeze_encoding = True
        if self.args['freeze_decoding'] == True: self.do_freeze_decoding = True
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
        if self.args['no_split_sentences'] is True:
            self.do_no_positional = True
            hparams['split_sentences'] = False
        if self.args['decoder_layers'] is not None: hparams['decoder_layers'] = int(self.args['decoder_layers'])
        if self.args['start_epoch'] is not None: self.start_epoch = int(self.args['start_epoch'])
        if self.args['no_attention'] is not False: self.do_no_attention = True
        if self.args['skip_unk'] is not False: self.do_skip_unk = True
        if self.args['print_control'] is not None:
            self.do_print_control = True
            self.print_control_num = float(self.args['print_control'])
        if self.args['json_record_offset'] is not None:
            self.best_accuracy_record_offset = int(self.args['json_record_offset'])
        if self.args['no_vocab_limit']: hparams['num_vocab_total'] = None
        if self.args['record_loss']: self.do_record_loss = True
        if self.printable == '': self.printable = hparams['base_filename']
        if hparams['cuda']: torch.set_default_tensor_type('torch.cuda.FloatTensor')

        ''' reset lr vars if changed from command line '''
        self.lr_low = hparams['learning_rate'] #/ 100.0
        self.lr_increment = self.lr_low / 4.0
        if self.args['lr_adjust'] is not None:
            self.lr_adjustment_num = int(self.args['lr_adjust'])
            hparams['learning_rate'] = self.lr_low + float(self.lr_adjustment_num) * self.lr_increment

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
                                                                             omit_unk=self.do_hide_unk,
                                                                             skip_unk=self.do_skip_unk)

            #self.first_load = True
            self.epoch_length = self.starting_epoch_length
            if self.epoch_length > len(self.pairs): self.epoch_length = len(self.pairs) - 1

            self.train_iters(None, None, self.epoch_length, print_every=self.print_every, learning_rate=lr)
            self.start = 0

            print('auto save.')
            print('%.2f' % self.score,'score')

            self.save_checkpoint(num=len(self.pairs))
            self.saved_files += 1
            self.this_epoch = 0
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

        self.input_lang, self.output_lang, self.pairs = self.prepareData(self.train_fr, self.train_to,
                                                                         lang3=self.train_ques, reverse=False,
                                                                         omit_unk=self.do_hide_unk,
                                                                         skip_unk=self.do_skip_unk)

        self.update_result_file()
        pass

    def task_interactive(self):

        print('-------------------')
        try:
            while True:
                line = input("> ")
                line = tokenize_weak.format(line)
                print(line)
                pad = hparams['tokens_per_sentence']
                add_eol = False
                line = self.variableFromSentence(self.input_lang, line, add_eol=add_eol, pad=pad)

                out , _ =self.evaluate(None, None, line, question=line)
                print(out)
        except EOFError:
            print()
            exit()

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

        if hparams['autoencode'] > 0.0 and not self.do_autoencode_words:
            a = hparams['autoencode']
            #self.pairs = [ [p[0], p[0], p[0]] for p in self.pairs]
            self.pairs = [ [p[0], p[0], p[0]] if random.uniform(0.0,1.0) <= a else [p[0], p[1], p[2]] for p in self.pairs]
            self.output_lang = self.input_lang

        if hparams['autoencode'] > 0.0 and self.do_autoencode_words:
            a = hparams['autoencode']

            self.pairs = self._make_autoencode_pairs(a, self.pairs)
            self.output_lang = self.input_lang

        return self.input_lang, self.output_lang, self.pairs

    def _make_autoencode_pairs(self, a, pairs_in):
        pairs_out = []
        for p in pairs_in:
            p_from = p[0] # in
            p_ques = p[1] # ques
            p_to = p[2] # out

            p_from = p_from.split(' ')
            p_to = p_to.split(' ')
            p_ques = p_ques.split(' ')

            #print(p_from, p_ques, p_to)

            while len(p_from) > len(p_to):
                p_to.append(hparams['unk'])
            while len(p_to) > len(p_from):
                p_from.append(hparams['unk'])

            for i in range(len(p_from)):
                if random.uniform(0.0,1.0) > a :#and i < hparams['tokens_per_sentence']:
                    p_to[i] = p_from[i]
                else:
                    pass
            #print(p_from, len(p_from), p_ques, len(p_ques), p_to, len(p_to))

            pairs_out.append([' '.join(p_from), ' '.join(p_ques), ' '.join(p_to)])
            #pairs_out.append([p_from, p_ques, p_to])

        return pairs_out


    def prepareData(self,lang1, lang2,lang3=None, reverse=False, omit_unk=False, skip_unk=False):
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

            new_pairs = []
            for p in range(len(self.pairs)):
                #print(self.pairs[p])
                a = []
                b = []
                c = []

                skip = False

                if len(self.pairs[p][0].split(' ')) > hparams['tokens_per_sentence']: skip = True
                if len(self.pairs[p][1].split(' ')) > hparams['tokens_per_sentence']: skip = True
                if lang3 is not None:
                    if len(self.pairs[p][2].split(' ')) > hparams['tokens_per_sentence']: skip = True
                for word in self.pairs[p][0].split(' '):
                    if word in self.vocab_lang.word2index and word not in blacklist:
                        a.append(word)
                    elif skip_unk:
                        skip = True
                    elif not omit_unk:
                        a.append(hparams['unk'])
                for word in self.pairs[p][1].split(' '):
                    if word in self.vocab_lang.word2index and word not in blacklist:
                        b.append(word)
                    elif skip_unk:
                        skip = True
                    elif not omit_unk:
                        b.append(hparams['unk'])
                pairs = [' '.join(a), ' '.join(b)]
                if lang3 is not None:
                    for word in self.pairs[p][2].split(' '):
                        if word in self.vocab_lang.word2index and word not in blacklist:
                            c.append(word)
                        elif skip_unk:
                            skip = True
                        elif not omit_unk:
                            c.append(hparams['unk'])
                    pairs.append( ' '.join(c) )
                if not skip: new_pairs.append(pairs)
            self.pairs = new_pairs

        else:
            for pair in self.pairs:
                self.input_lang.addSentence(pair[0])
                self.output_lang.addSentence(pair[1])

        print("Counted words:")
        print(self.input_lang.name, self.input_lang.n_words)
        print(self.output_lang.name, self.output_lang.n_words)

        if self.do_load_embeddings:
            print('embedding option detected.')
            self.task_set_embedding_matrix()

        '''
        if self.epoch_length > len(self.pairs) and not self.do_test_not_train:
            print('reset epoch length:', len(self.pairs))
            self.epoch_length = len(self.pairs)
        '''
        return self.input_lang, self.output_lang, self.pairs


    def indexesFromSentence(self,lang, sentence, skip_unk=False):
        s = sentence.split(' ')
        sent = []
        for word in s:
            if word in lang.word2index:
                if word == hparams['eol']: word = EOS_token
                elif word == hparams['sol']: word = SOS_token
                else: word = lang.word2index[word]
                sent.append(word)
            elif skip_unk:
                print('-')
                return None
            elif not self.do_hide_unk:
                sent.append(lang.word2index[hparams['unk']])
        if len(sent) >= MAX_LENGTH and not self.do_load_babi:
            sent = sent[:MAX_LENGTH]
            sent[-1] = EOS_token
        if self.do_load_babi and False:
            sent.append(EOS_token)
            #print(sent,'<<<<')
        if len(sent) == 0: sent.append(0)
        if self.do_load_recurrent:
            sent = sent[:MAX_LENGTH]
        return sent

        #return [lang.word2index[word] for word in sentence.split(' ')]

    def variables_for_batch(self, pairs, size, start, skip_unk=False):
        e = self.epoch_length * self.this_epoch + self.epoch_length
        if start + size > e  and start < e :
            size = e - start #- 1
            print('process size', size,'next')
        if size == 0 or start >= len(pairs):
            print('empty return.')
            return self.variablesFromPair(('','',''))
        g1 = []
        g2 = []
        g3 = []

        group = []

        if skip_unk:
            num = 0
            self._skipped = 0
            while len(g1) < size:

                if start + num >= len(pairs): break

                #print(start + num, end=', ')

                triplet = pairs[start + num]
                x = self.variablesFromPair(triplet, skip_unk=skip_unk)
                if x is not None:
                    if not hparams['split_sentences']:
                        g1.append(x[0].squeeze(1))
                    else:
                        g1.append(x[0])
                    g2.append(x[1].squeeze(1))
                    g3.append(x[2].squeeze(1))
                else:
                    self._skipped += 1
                num += 1
            #print(self._skipped)
            return (g1, g2, g3)
            pass
        else:
            group = pairs[start:start + size]

        for i in group:
            g = self.variablesFromPair(i)
            #print(g[0])
            if not hparams['split_sentences']:
                g1.append(g[0].squeeze(1))
            else:
                g1.append(g[0])
            g2.append(g[1].squeeze(1))
            g3.append(g[2].squeeze(1))

        return (g1, g2, g3)

    def variableFromSentence(self, lang, sentence, add_eol=False, pad=0, skip_unk=False):
        max = hparams['tokens_per_sentence']
        indexes = self.indexesFromSentence(lang, sentence, skip_unk=skip_unk)
        if indexes is None and skip_unk: return indexes
        if add_eol and len(indexes) < pad: indexes.append(EOS_token)
        sentence_len = len(indexes)
        while pad > sentence_len:
            indexes.append(UNK_token)
            pad -= 1
        result = Variable(torch.LongTensor(indexes).unsqueeze(1))#.view(-1, 1))
        #print(result.size(),'r')
        if hparams['cuda']:
            return result[:max].cuda()
        else:
            return result[:max]

    def variablesFromPair(self,pair, skip_unk=False):
        if hparams['split_sentences'] and False :
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
                    z = self.variableFromSentence(self.input_lang, l[i], skip_unk=skip_unk)
                    if skip_unk and z is None: return None
                    #print(z)
                    sen.append(z)
            #sen = torch.cat(sen, dim=0)
            #for i in sen: print(i.size())
            #print('---')
            input_variable = sen
            pass
        else:
            pad = hparams['tokens_per_sentence']
            input_variable = self.variableFromSentence(self.input_lang, pair[0], pad=pad, skip_unk=skip_unk)

        pad = hparams['tokens_per_sentence']
        question_variable = self.variableFromSentence(self.output_lang, pair[1], pad=pad, skip_unk=skip_unk)

        if len(pair) > 2:
            #print(pair[2],',pair')
            #if (len(pair[2]) > 0) or True:
            pad = 0
            add_eol = False
            if self.do_recurrent_output:
                pad = hparams['tokens_per_sentence']
                add_eol = True
            target_variable = self.variableFromSentence(self.output_lang, pair[2],
                                                        add_eol=add_eol,
                                                        pad=pad,
                                                        skip_unk=skip_unk)
            if self.do_recurrent_output:
                target_variable = target_variable.unsqueeze(0)

        else:
            if skip_unk and (input_variable is None or question_variable is None):
                return None
            return (input_variable, question_variable)

        if skip_unk and (input_variable is None or question_variable is None or target_variable is None):
            return None
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
        if self.do_load_babi or self.do_conserve_space or self.do_train_long or self.do_recurrent_output:
            num = self.this_epoch * len(self.pairs) + num
            torch.save(state,basename+ '.best.pth')
            #####
            if self.do_test_not_train:
                self.best_accuracy_dict[
                    str((self.best_accuracy_record_offset + self.saved_files) * self.best_accuracy_graph_size)
                ] = str(self.score)

                self.best_loss_dict[
                    str((self.best_loss_record_offset + self.saved_files)* self.best_loss_graph_size)
                ] = str(self.best_loss_graph)

                print('offset', self.best_accuracy_record_offset, ', epoch', self.this_epoch)
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
                    self.model_0_wra.new_freeze_embedding()
                else:
                    self.model_0_wra.new_freeze_embedding(do_freeze=False)
                if self.do_freeze_encoding:
                    self.model_0_wra.new_freeze_encoding()
                else:
                    self.model_0_wra.new_freeze_encoding(do_freeze=False)
                if self.do_freeze_decoding:
                    self.model_0_wra.new_freeze_decoding()
                else:
                    self.model_0_wra.new_freeze_decoding(do_freeze=False)

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

        if self._highest_reached_test(num=20, goal=10):
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

    def _highest_reached_test(self, num=0, lst=None, goal=0):
        ''' must only call this fn once !! '''
        if lst is None:
            lst = self.score_list
        if len(lst) == 0: return False
        val = float(lst[-1])
        if val > self._highest_validation_for_quit:
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


    #######################################

    def train(self,input_variable, target_variable, question_variable, encoder, decoder, wrapper_optimizer, decoder_optimizer, memory_optimizer, attention_optimizer, criterion, max_length=MAX_LENGTH):

        if criterion is not None:
            wrapper_optimizer.zero_grad()
            self.model_0_wra.train()
            outputs, _, ans, _ = self.model_0_wra(input_variable, question_variable, target_variable,
                                                  criterion)
            loss = 0

            if self.do_recurrent_output:
                ##lz = len(target_variable[0])

                #target_variable = Variable(torch.LongTensor(target_variable)) #torch.cat(target_variable, dim=0)

                ansx = Variable(ans.data.max(dim=2)[1])

                ans = ans.float().permute(1,0,2).contiguous()

                #ans = ans.view(-1, self.output_lang.n_words)
                #target_variable = target_variable.view(-1)

                for i in range(len(target_variable)):
                    target_v = target_variable[i].squeeze(0).squeeze(1)
                    loss += criterion(ans[i, :, :], target_v)
                #print( ans.size(),'ans', target_variable.size(), 'tv')

            elif self.do_batch_process:
                target_variable = torch.cat(target_variable,dim=0)
                ans = ans.permute(1,0)
            else:
                target_variable = target_variable[0]
                print(len(ans),ans.size())
                ans = torch.argmax(ans,dim=1)
                #ans = ans[0]

            #ans = ans.permute(1,0)

            #print(ans.size(), target_variable.size(),'criterion')

            if not self.do_recurrent_output:
                loss = criterion(ans, target_variable)

            loss.backward()
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
                    ansx = Variable(ans.data.max(dim=2)[1])
                    ans = ans.permute(1,0,2)

            #self._test_embedding()

        if self.do_recurrent_output:
            ans = ansx.permute(1,0)
            #print(ans.size(),'ans')

        if self.do_clip_grad_norm:
            clip = 50.0
            _ = torch.nn.utils.clip_grad_norm_(self.model_0_wra.parameters(), clip)
            #print('clip')

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
        temp_batch_size = 0

        epoch_len = self.epoch_length
        epoch_start = self.this_epoch * self.epoch_length
        if epoch_start >= len(self.pairs):
            n = (len(self.pairs) // self.epoch_length)
            if n == 0: n = 1
            e = self.this_epoch % n
            epoch_start = e * self.epoch_length
            #exit()
            #length

        epoch_stop = epoch_start + epoch_len

        if len(self.pairs) < epoch_stop:
            epoch_stop = len(self.pairs)
            #epoch_len = len(self.pairs) - epoch_start

        nn_iters = self.epoch_length

        if not self.do_test_not_train:
            print('limit pairs:', len(self.pairs),
                  '- end of this step:',epoch_stop,
                  '- steps:', len(self.pairs) // self.epoch_length,
                  '- this step:', self.this_epoch + 1,
                  '- step len:', self.epoch_length)

        self.time_str = self._as_minutes(self.time_num)

        if self.opt_1 is None or self.first_load:

            wrapper_optimizer = self._make_optimizer()
            self.opt_1 = wrapper_optimizer

        weight = torch.ones(self.output_lang.n_words)
        weight[self.output_lang.word2index[hparams['unk']]] = 0.0
        self.criterion = nn.CrossEntropyLoss(weight=weight) #size_average=False)
        #self.criterion = nn.NLLLoss(weight=weight)


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
            step = 1 # hparams['batch_size']
            if self.start_epoch is 0: start = 0

        for iter in range(epoch_start, epoch_stop + 1, step):

            if self.do_batch_process and (iter ) % hparams['batch_size'] == 0 and iter < len(self.pairs):

                skip_unk = self.do_skip_unk
                group = self.variables_for_batch(self.pairs, hparams['batch_size'], iter, skip_unk=skip_unk)

                input_variable = group[0]
                question_variable = group[1]
                target_variable = group[2]

                temp_batch_size = len(input_variable)
                #print(temp_batch_size,'temp')
                #if self.do_recurrent_output:
                #    temp_batch_size = len(input_variable)# * hparams['tokens_per_sentence']

            elif self.do_batch_process:
                continue
                pass

            outputs, ans, l = self.train(input_variable, target_variable, question_variable, encoder,
                                            decoder, self.opt_1, None,
                                            None, None, criterion)
            num_count += 1

            if self.do_recurrent_output and self.do_load_babi:

                for i in range(len(target_variable)):
                    for j in range(target_variable[i].size()[1]):
                        t_val = target_variable[i][0,j,0].item()
                        o_val = ans[i][j].item()

                        if int(o_val) == int(t_val):
                            num_right += 1
                            num_right_small += 1
                            if int(o_val) == EOS_token:
                                num_right_small += hparams['tokens_per_sentence'] - (j + 1)
                                num_right += hparams['tokens_per_sentence'] - (j + 1)
                                #print('full line', i, j, num_right_small)
                                break
                        else:
                            # next sentence
                            break

                num_tot += temp_batch_size * hparams['tokens_per_sentence']


                self.score = float(num_right / num_tot) * 100

            if self.do_load_babi and not self.do_recurrent_output:

                for i in range(len(target_variable)):
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
                    print('iter = '+str(iter)+ ', num of iters = '+str(n_iters)
                          + ', ' + self.printable + ', saved files = ' + str(self.saved_files)
                          + ', low loss = %.6f' % self.long_term_loss, end=' ')

                    print()

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

                        if not self.do_test_not_train:
                            self.best_loss_graph = print_loss_avg

                        if not self.do_test_not_train and not self.do_load_babi:
                            self.save_checkpoint(num=iter,extra=extra)
                            self.saved_files += 1
                            print('======= save file '+ extra+' ========')
                    elif not self.do_load_babi:
                        print('skip save!')
                self.time_elapsed_str = self._time_since(self.time_num)
                self.time_elapsed_num = time.time()

                if self._print_control(iter):
                    print('(%d %d%%) %.6f loss' % (iter, (iter - epoch_start) / self.epoch_length * 100, print_loss_avg),self.time_elapsed_str, end=' ')
                    if self.do_batch_process: print('- batch-size', temp_batch_size, end=' ')
                    if self.do_auto_stop: print('- count', self._count_epochs_for_quit)
                    else: print('')

                #print(epoch_start, iter, temp_batch_size, epoch_stop)

                if not self.do_skip_validation and self.do_sample_on_screen and self._print_control(iter): # and temp_batch_size > 0 and epoch_start + iter < epoch_stop:
                    ###########################
                    self._show_sample(iter, epoch_start, epoch_stop, temp_batch_size)

                    ############################
                if self.do_recurrent_output:
                    num_right_small = math.floor(num_right_small / (hparams['tokens_per_sentence'] ))
                    pass

                if self.do_load_babi and self.do_test_not_train:
                    if self._print_control(iter):
                        print('current accuracy: %.4f' % self.score, '- num right '+ str(num_right_small ))
                    num_right_small = 0

                if self.do_load_babi and not self.do_test_not_train:
                    if self._print_control(iter):
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
        print('weight decay:', hparams['weight_decay'])
        print(num_count, 'exec count')
        print('raw score:', num_right, num_tot, num_right_small, len(self.pairs))

    def _show_sample(self, iter=0, epoch_start=0, epoch_stop=hparams['batch_size'], temp_batch_size=hparams['batch_size']):
        ###########################
        if epoch_start + iter >= epoch_stop:
            choice = random.choice(self.pairs)
        else:
            choice = random.choice(self.pairs[epoch_start + iter: epoch_start + iter + temp_batch_size])
        print('src:', choice[0])
        question = None
        if self.do_load_babi:
            #print('ques:', choice[1])
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
                        if not self.do_print_to_screen: break
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
        self.input_lang, self.output_lang, self.pairs = self.prepareData(self.train_fr, self.train_to,
                                                                         lang3=self.train_ques, reverse=False,
                                                                         omit_unk=self.do_hide_unk,
                                                                         skip_unk=self.do_skip_unk)
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
        self.input_lang, self.output_lang, self.pairs = self.prepareData(self.train_fr, self.train_to,
                                                                         lang3=self.train_ques, reverse=False,
                                                                         omit_unk=self.do_hide_unk,
                                                                         skip_unk=self.do_skip_unk)
        layers = hparams['layers']
        dropout = hparams['dropout']
        pytorch_embed_size = hparams['pytorch_embed_size']
        sol_token = self.output_lang.word2index[hparams['sol']]

        self.model_0_wra = WrapMemRNN(self.input_lang.n_words, pytorch_embed_size, self.hidden_size,layers,
                                      dropout=dropout,do_babi=self.do_load_babi,
                                      freeze_embedding=self.do_freeze_embedding, embedding=self.embedding_matrix,
                                      print_to_screen=self.do_print_to_screen, recurrent_output=self.do_recurrent_output,
                                      sol_token=sol_token, cancel_attention=self.do_no_attention,
                                      freeze_decoder=self.do_freeze_decoding, freeze_encoder=self.do_freeze_encoding)
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
        self.input_lang, self.output_lang, self.pairs = self.prepareData(self.train_fr, self.train_to,
                                                                         lang3=self.train_ques, reverse=False,
                                                                         omit_unk=self.do_hide_unk,
                                                                         skip_unk=self.do_skip_unk)
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
                                      freeze_decoder=self.do_freeze_decoding, freeze_encoder=self.do_freeze_encoding)
        if hparams['cuda']: self.model_0_wra = self.model_0_wra.cuda()

        self.first_load = True
        self.load_checkpoint()
        lr = hparams['learning_rate']
        self.train_iters(None, None, self.epoch_length, print_every=self.print_every, learning_rate=lr)

    def update_result_file(self):

        self._test_embedding(exit=False)

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
        basename_loss = hparams['save_dir'] + hparams['base_filename'] + '.loss.json'
        if len(self.best_accuracy_dict) > 0:
            with open(basename, 'w') as z:
                z.write(json.dumps(self.best_accuracy_dict))
                z.write('\n')
            z.close()
        if len(self.best_loss_dict) > 0:
            with open(basename_loss, 'w') as z:
                z.write(json.dumps(self.best_loss_dict))
                z.write('\n')
            z.close()

    def read_json_file(self):
        basename = hparams['save_dir'] + hparams['base_filename'] + '.json'
        basename_loss = hparams['save_dir'] + hparams['base_filename'] + '.loss.json'
        if os.path.isfile(basename):
            with open(basename) as z:
                json_data = json.load(z)
            self.best_accuracy_dict = json_data

            y = min(int(k) for k, v in self.best_accuracy_dict.items())
            if int(y) != self.epoch_length:
                self.best_accuracy_graph_size = int(y)
            else:
                self.best_accuracy_graph_size = self.epoch_length
            x = max(int(k) for k, v in self.best_accuracy_dict.items() )
            x = int(int(x) / self.best_accuracy_graph_size)

            if self.args['json_record_offset'] is None:
                self.best_accuracy_record_offset = x
            if self.args['start_epoch'] is None:
                self.start_epoch = x
        if os.path.isfile(basename_loss):
            with open(basename_loss) as z:
                json_data = json.load(z)
            self.best_loss_dict = json_data

            y = min(int(k) for k, v in self.best_loss_dict.items())
            if int(y) != self.epoch_length:
                self.best_loss_graph_size = int(y)
            else:
                self.best_loss_graph_size = self.epoch_length
            x = max(int(k) for k, v in self.best_loss_dict.items() )
            x = int(int(x) / self.best_loss_graph_size)
            self.best_loss_record_offset = x


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
                                                             omit_unk=n.do_hide_unk, skip_unk=n.do_skip_unk)


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
                                   freeze_decoder=n.do_freeze_decoding, freeze_encoder=n.do_freeze_encoding)

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
        if not n.do_interactive:
            n.update_result_file()
        else:
            print()

