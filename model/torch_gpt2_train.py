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
import itertools
import matplotlib.pyplot as plt
import heapq
import matplotlib.patches as mpatches
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
import numpy as np
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel


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
to the BeamSearch class and came from:

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

blacklist_vocab = ['re', 've', 's', 't', 'll', 'm', 'don', 'd']
blacklist_sent = blacklist_vocab #+ ['i']
blacklist_supress = [] #[['i', 0.0001], ['you', 1.0]]

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


def prune_tensor( input, size):
    if isinstance(input, (list)): return input
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


#################### Wrapper ####################

class WrapMemRNN: #(nn.Module):
    def __init__(self,vocab_size, embed_dim,  hidden_size, n_layers, dropout=0.3, do_babi=True, bad_token_lst=[],
                 freeze_embedding=False, embedding=None, recurrent_output=False,print_to_screen=False, sol_token=0,
                 cancel_attention=False, freeze_encoder=False, freeze_decoder=False):

        #super(WrapMemRNN, self).__init__()

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
        beam_width = 0 if hparams['beam'] is None else hparams['beam']

        #self.embed = nn.Embedding(vocab_size,hidden_size,padding_idx=1)
        #self.embed.weight.requires_grad = not self.freeze_embedding

        # Load pre-trained model (weights)
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.past = None

        '''
        self.model_1_seq = Encoder(vocab_size,embed_dim, hidden_size,
                                          2, dropout,embed=self.embed)

        self.model_6_dec = Decoder(vocab_size, embed_dim, hidden_size,2, dropout, self.embed,
                                   cancel_attention=self.cancel_attention)

        self.beam_helper = BeamHelper(beam_width, hparams['tokens_per_sentence'])
        '''

        self.opt_1 = None
        self.opt_2 = None

        self.input_var = None  # for input
        self.answer_var = None  # for answer
        self.q_q_last = None # question
        self.inp_c = None  # extra input
        self.inp_c_seq = None
        self.all_mem = None
        self.last_mem = None  # output of mem unit
        self.prediction = None  # final single word prediction
        #self.memory_hops = hparams['babi_memory_hops']
        #self.inv_idx = torch.arange(100 - 1, -1, -1).long() ## inverse index for 100 values
        self.pass_no_token = False

        self.reset_parameters()

        if self.embedding is not None:
            self.load_embedding(self.embedding, not self.freeze_embedding)
        self.share_embedding()

        if self.freeze_embedding or self.embedding is not None:
            self.new_freeze_embedding()

        if freeze_encoder:
            self.new_freeze_encoding()
        if freeze_decoder:
            self.new_freeze_decoding()
        pass

    def load_embedding(self, embedding, requires_grad=True):
        #embedding = np.transpose(embedding,(1,0))
        #e = torch.from_numpy(embedding)

        #self.embed.weight.data.copy_(e) #torch.from_numpy(embedding))
        #self.embed.weight.requires_grad = requires_grad
        pass

    def share_embedding(self):
        pass
        #self.model_1_seq.load_embedding(self.embed, not self.freeze_embedding)
        #self.model_6_dec.load_embedding(self.embed, not self.freeze_embedding)

    def reset_parameters(self):
        return


    def __call__(self, input_variable, question_variable, target_variable, length_variable, criterion=None):

        if length_variable is not None and input_variable.size(0) == length_variable.size(0):
            out_lst = []
            token_num = 0 ## -1 ??
            for i in range(input_variable.size(0) ):
                self.past = None
                len = length_variable[i] - 1

                iv = input_variable[i,:len]
                iv = prune_tensor(iv, 2)

                #print(iv.size(),'ivsize')
                try:
                    ans, self.past = self.model(iv, self.past)
                except:
                    iv = torch.LongTensor([[0]])
                    ans, self.past = self.model(iv, self.past) ## space char?
                    #self.past = None
                    #print('bang char?')
                ans = ans[:,token_num,:]
                out_lst.append(ans)

            out_ans = torch.cat(out_lst, dim=0)

            #out_ans = out_ans.transpose(1,0)
            #print(out_ans.size(), out_lst[0].size(),'ans,out')
            #print(out_lst, len(out_lst), out_lst[0].size())
            return None, None, out_ans, None


    def new_freeze_embedding(self, do_freeze=True):
        #self.embed.weight.requires_grad = not do_freeze
        #self.model_1_seq.embed.weight.requires_grad = not do_freeze
        #self.model_6_dec.embed.weight.requires_grad = not do_freeze
        #self.embed.weight.requires_grad = not do_freeze
        #if do_freeze: print('freeze embedding')
        pass

    def new_freeze_decoding(self, do_freeze=True):
        '''
        for weight in self.model_6_dec.parameters():
            weight.requires_grad = not do_freeze

        if do_freeze: print('freeze decoding')
        '''
        pass

    def new_freeze_encoding(self, do_freeze=True):
        '''
        for weight in self.model_1_seq.parameters():
            weight.requires_grad = not do_freeze
        '''
        #if do_freeze: print('freeze encoding')
        pass

    def test_embedding(self, num=None):
        pass
        '''
        if num is None or True:
            num = torch.LongTensor([0]) #EOS_token  # magic number for testing = garden
        e = self.embed(num)
        print('encoder :',num)
        print(not self.embed.weight.requires_grad,': grad freeze')
        print(e.size(), 'test embedding')
        print(e[ 0, 0:10])  # print first ten values
        '''



######################## end pytorch modules ####################


class Lang:
    def __init__(self, name, limit=None):
        self.name = name
        self.limit = limit
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0  # Count SOS and EOS

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

        global teacher_forcing_ratio, MAX_LENGTH

        self.model_0_wra = None
        self.past = None
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.words_end = ['.', '?', '!', '"']

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
        #self.memory_hops = 5
        self.start = 0
        self.this_epoch = 0
        self.true_epoch = 0
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
        self.do_no_vocabulary = True

        self.do_clip_grad_norm = False

        self.printable = ''

        parser = argparse.ArgumentParser(description='Train some NMT values.')
        parser.add_argument('--mode', help='mode of operation. (preset, long, interactive, plot)')
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
        #parser.add_argument('--batch',help='enable batch processing. (default)',action='store_true')
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
        parser.add_argument('--beam', help='activate beam search for eval phase.')
        parser.add_argument('--single', help='force single execution instead of batch execution.', action='store_true')
        parser.add_argument('--teacher-forcing', help='set forcing for recurrent output')
        parser.add_argument('--multiplier', help='learning rate multiplier for decoder.')
        parser.add_argument('--length', help='number of tokens per sentence.')
        parser.add_argument('--no-vocab', help='use open ended vocabulary length tokens.', action='store_true')

        self.args = parser.parse_args()
        self.args = vars(self.args)
        # print(self.args)

        hparams['unk'] = ' '

        if self.args['printable'] is not None:
            self.printable = str(self.args['printable'])
        if self.args['mode'] is None or self.args['mode'] not in ['preset', 'long', 'interactive', 'plot']:
            self.args['mode'] = 'preset'
            pass
        if self.args['mode'] == 'preset':
            ''' some preset flags for a typical run '''
            print('preset.')
            self.do_train = False
            self.do_record_loss = True
            self.do_load_babi = True
            self.do_load_recurrent = True
            self.do_train_long = True
            self.do_recurrent_output = True
            self.do_skip_unk = True
            self.do_hide_unk = True
            self.args['mode'] = 'long'
        #if self.args['mode'] == 'infer': self.do_infer = True
        #if self.args['mode'] == 'review': self.do_review = True
        if self.args['mode'] == 'long': self.do_train_long = True
        if self.args['mode'] == 'interactive': self.do_interactive = True
        if self.args['mode'] == 'plot':
            self.do_review = True
            self.do_plot = True
        if self.args['basename'] is not None:
            name = str(self.args['basename'])
            print(name)
            name = name.split('/')[-1]
            name = name.split('.')[0]
            hparams['base_filename'] = name #self.args['basename']
            print('set name:',hparams['base_filename'])
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
        #if self.args['batch'] is not False:
        #    self.do_batch_process = True
        #    print('batch operation now enabled by default.')
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
        if self.args['beam'] is not None:
            hparams['beam'] = int(self.args['beam'])
        if self.args['single']:
            hparams['single'] = True
        if self.args['teacher_forcing'] is not None and not self.do_test_not_train:  # self.args['test']:
            hparams['teacher_forcing_ratio'] = float(self.args['teacher_forcing'])
            teacher_forcing_ratio = float(self.args['teacher_forcing'])
        if self.args['multiplier'] is not None:
            hparams['multiplier'] = float(self.args['multiplier'])
        if self.args['length'] is not None:
            hparams['tokens_per_sentence'] = int(self.args['length'])
            MAX_LENGTH = hparams['tokens_per_sentence']
        if self.args['no_vocab']:
            self.do_no_vocabulary = True
        if self.printable == '': self.printable = hparams['base_filename']

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

    def task_interactive(self, l=None, call_from_script=False):
        four_spaces = '    '
        token_num = 1
        if not call_from_script:
            print('-------------------')
        try:
            if True:
                if not call_from_script:
                    line = input("> ")
                    line = tokenize_weak.format(line)
                    print(line)
                elif l is not None:
                    line = l

                num = 0
                text_1 = line
                text_2 = ""
                zlist = ''
                self.past = None
                # decode_list = []

                if False:
                    word = self.random_word()
                else:
                    word = ''
                space_character = ' '  ## no space!!??

                while num < hparams['tokens_per_sentence']:

                    indexed_tokens_2 = self.tokenizer.encode(word + space_character + text_1 + ' ? ' + text_2)

                    tokens_tensor_2 = torch.tensor([indexed_tokens_2])

                    with torch.no_grad():
                        predictions_1, self.past = self.model_0_wra.model(tokens_tensor_2, past=self.past)

                    if False:
                        token_num = predictions_1.size(1)

                    #zlist = ''
                    xlist = ''
                    for i in range(token_num):
                        ii = i  # 0 ## i
                        p_index = torch.argmax(predictions_1[0, ii, :], dim=-1).item()
                        p_token = self.tokenizer.decode([p_index])
                        zlist += '[' + str(p_index) + ']'
                        xlist += p_token

                    xlist = xlist.strip()
                    xlist = xlist.replace(',', '')
                    xlist = xlist.replace('.', '')
                    #print(zlist)

                    #print('out >', xlist)
                    text_2 += xlist + ' '
                    if xlist.strip() in self.words_end or text_2.endswith(four_spaces):
                        break

                    num += 1

                print(zlist)
                print('out >', text_2)
                return text_2

        except EOFError:
                print()
                exit()

    def get_sentence(self, i):
        return self.task_interactive(l=i, call_from_script=True)

    def task_convert(self):
        hparams['base_filename'] += '.small'
        self.save_checkpoint(is_best=False,converted=True)

    ################################################

    def prep_blacklist_supress(self):
        global blacklist_supress
        if self.do_no_vocabulary: return

        bl = []
        if isinstance(blacklist_supress[0][0],str):
            for i in blacklist_supress:
                i[0] = self.output_lang.word2index[i[0]]
                bl.append([i[0], i[1]])
            blacklist_supress = bl
        #print(blacklist_supress)

    def open_sentences(self, filename):
        t_yyy = []
        with open(filename, 'r') as r:
            for xx in r:
                t_yyy.append(xx.lower())
        return t_yyy

    def count_sentences(self, filename):
        print('count vocab:', filename)
        #z = self.open_sentences(filename)
        z = self.tokenizer.__len__()
        return z #len(z)

    def readLangs(self, lang1, lang2, lang3=None, reverse=False, load_vocab_file=None, babi_ending=False):
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


        if True:
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
                print(self.tokenizer.__len__(), 'max')
                self.output_lang = Lang('lang')
                for i in range(self.tokenizer.__len__()):
                    self.output_lang.addWord(self.tokenizer.decode([i]))

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

                if not self.do_no_vocabulary:
                    if len(self.pairs[p][0].split(' ')) > hparams['tokens_per_sentence']: skip = True
                    if len(self.pairs[p][1].split(' ')) > hparams['tokens_per_sentence']: skip = True
                    if lang3 is not None:
                        if len(self.pairs[p][2].split(' ')) > hparams['tokens_per_sentence']: skip = True

                for word in self.pairs[p][0].split(' '):
                    if (word in self.vocab_lang.word2index and word not in blacklist_vocab) or self.do_no_vocabulary:
                        a.append(word)
                    elif skip_unk:
                        skip = True
                    elif not omit_unk:
                        a.append(hparams['unk'])
                for word in self.pairs[p][1].split(' '):
                    if (word in self.vocab_lang.word2index and word not in blacklist_vocab) or self.do_no_vocabulary:
                        b.append(word)
                    elif skip_unk:
                        skip = True
                    elif not omit_unk:
                        b.append(hparams['unk'])
                pairs = [' '.join(a), ' '.join(b)]
                if lang3 is not None:
                    for word in self.pairs[p][2].split(' '):
                        if (word in self.vocab_lang.word2index and word not in blacklist_vocab) or self.do_no_vocabulary:
                            c.append(word)
                        elif skip_unk:
                            skip = True
                        elif not omit_unk:
                            c.append(hparams['unk'])
                    pairs.append( ' '.join(c) )
                if not skip or self.do_no_vocabulary: new_pairs.append(pairs)
            self.pairs = new_pairs

        else:
            for pair in self.pairs:
                self.input_lang.addSentence(pair[0])
                self.output_lang.addSentence(pair[1])

        print("Counted words:")
        print(self.input_lang.name, self.input_lang.n_words)
        print(self.output_lang.name, self.output_lang.n_words)

        print("Num pairs", len(self.pairs))

        if self.do_load_embeddings:
            print('embedding option detected.')
            self.task_set_embedding_matrix()

        if hparams['beam'] is not None:
            self.prep_blacklist_supress()

        return self.input_lang, self.output_lang, self.pairs

    '''
    def chop_word_for_index(self,lang, word):
        w = word

        l = []
        i = 0
        j = len(w)
        #flag = False
        for _ in range(len(w)): # i == start character
            flag = False
            j = len(w) - i

            for _ in range(len(w)): # j == word size

                if i >= len(w) or j < 0: break

                part = w[i: i + j]
                part = ''.join(part)
                if part in lang.word2index:

                    index = lang.word2index[part]
                    l.append(index)
                    i += (len(part) )
                    #j = len(w) - (len(part) -1)

                    flag = True
                    break
                else:
                    j -= 1
            if not flag: i += 1



        l.extend([lang.word2index[hparams['eow']]  ]) #, lang.word2index[hparams['unk']]])
        return l
    '''

    def indexesFromSentence(self,lang, sentence, skip_unk=False, add_sos=True, add_eos=False, return_string=False, pad=-1, no_padding=False):
        if pad == -1:
            MAX_LENGTH = hparams['tokens_per_sentence']
        else:
            MAX_LENGTH = pad
        s = sentence.split(' ')

        sent = self.tokenizer.encode(sentence)
        '''
        if False:
            sent = []
            if add_sos and len(s) > 0 and s[0] != hparams['sol']: sent = [ SOS_token ]
            for word in s:
                if not self.do_no_vocabulary:
                    if word in lang.word2index and word not in blacklist_sent:
                        if word == hparams['eol']: word = EOS_token
                        elif word == hparams['sol']: word = SOS_token
                        else: word = lang.word2index[word]
                        sent.append(word)
                    elif skip_unk:
                        print('-')
                        return None
                    elif not self.do_hide_unk:
                        sent.append(lang.word2index[hparams['unk']])
    
                if self.do_no_vocabulary:
                    sent.extend(self.chop_word_for_index(lang, word))

            if len(sent) >= MAX_LENGTH and add_eos:
                sent = sent[:MAX_LENGTH]
                if len(sent) > 1 and (sent[-1] != EOS_token or sent[-1] == UNK_token):
                    sent[-1] = EOS_token
            elif add_eos:
                if sent[-1] != EOS_token : #or sent[-1] != UNK_token:
                    sent.append(EOS_token)
                if sent[-1] == UNK_token:
                    sent[-1] = EOS_token
                #print(sent,'<<<<')
        '''
        if len(sent) == 0: sent.append(0)
        if pad == -1 and not no_padding:
            while len(sent) < MAX_LENGTH:
                sent.append(0)
        if self.do_load_recurrent:
            sent = sent[:MAX_LENGTH]

        if not self.model_0_wra.model.train: print(sent)

        if return_string:
            return sentence
        return sent



    def pairs_from_batch(self, pairs):
        #print(len(pairs), len(pairs[0]), len(pairs[0][0]),'pairs')
        out = []
        lengths = []
        for i in pairs:

            #a = self.variableFromSentence(None, i[0])
            #b = self.variableFromSentence(None, i[1])
            #c = self.variableFromSentence(None, i[2])
            token = self.tokenizer.encode(' ')[0]

            i[0] = ' ' + i[0] ## input
            i[1] = ' ' + i[1] ## unused
            #i[2] = ' ' + i[2] ## not good for predicted word !!

            a = self.tokenizer.encode(i[0])
            b = self.tokenizer.encode(i[1])
            c = self.tokenizer.encode(i[2])

            d = len(a) #+ 1 ## single extra space!!

            e = token
            for i in range(len(c)):
                if c[i] != token: e = c[i]
            c = [e]

            if True:
                while len(a) < hparams['tokens_per_sentence']:
                    a.append(token)
                while len(b) < hparams['tokens_per_sentence']:
                    b.append(token)
                while len(c) < hparams['tokens_per_sentence']:
                    c.append(c[0])

                a = a[:hparams['tokens_per_sentence']]
                b = b[:hparams['tokens_per_sentence']]
                c = c[:hparams['tokens_per_sentence']]
                #print(c, 'c')

            out.append([a,b,c])
            lengths.append(d)
            #print(a,b,c)
        #exit()
        out = torch.LongTensor(out)
        lengths = torch.LongTensor(lengths)
        return out, lengths

    def variables_for_batch(self, pairs, size, start, skip_unk=False, pad_and_batch=True):
        e = self.epoch_length * self.this_epoch + self.epoch_length
        length = 0

        if start + size > e  and start < e :
            size = e - start #- 1
            print('process size', size,'next')
        if size == 0 or start >= len(pairs):
            print('empty return.')
            return self.variablesFromPair(('','',''), length)

        g1 = []
        g2 = []
        g3 = []

        group = []

        if skip_unk:
            num = 0
            pairs2 = []
            self._skipped = 0
            while len(g1) < size and len(pairs2) < size:

                if start + num >= len(pairs): break

                #print(start + num, end=', ')

                triplet = pairs[start + num]

                if not pad_and_batch:
                    x = self.variablesFromPair(triplet, skip_unk=skip_unk)
                else:
                    #x = triplet
                    x = [
                        self.indexesFromSentence(self.output_lang, triplet[0], skip_unk=skip_unk, add_eos=True, return_string=True),
                        self.indexesFromSentence(self.output_lang, triplet[1], skip_unk=skip_unk, add_eos=False, return_string=True),
                        self.indexesFromSentence(self.output_lang, triplet[2], skip_unk=skip_unk, add_eos=False, return_string=True)
                    ]
                    if x[0] is None: x = None

                if x is not None and not pad_and_batch:
                    if not hparams['split_sentences']:
                        g1.append(x[0].squeeze(1))
                    else:
                        g1.append(x[0])
                    g2.append(x[1].squeeze(1))
                    g3.append(x[2].squeeze(1))
                else:
                    self._skipped += 1

                if pad_and_batch and x is not None:
                    pairs2.append(x)
                else:
                    self._skipped += 1
                num += 1
            #print(self._skipped)

            if pad_and_batch:
                #print('pairs2')
                return self.pairs_from_batch(pairs2) #self.tokenizer.encode(pairs2) #self.pad_and_batch(pairs2)

            return (g1, g2, g3, length)
            pass
        else:
            group = pairs[start:start + size]
            if pad_and_batch:
                #print('group')
                return self.pairs_from_batch(group) #self.tokenizer.encode(group) #self.pad_and_batch(group)

        for i in group:
            g = self.variablesFromPair(i)
            #print(g[0])
            if not hparams['split_sentences']:
                g1.append(g[0].squeeze(1))
            else:
                g1.append(g[0])
            g2.append(g[1].squeeze(1))
            g3.append(g[2].squeeze(1))

        return (g1, g2, g3, length)

    def variableFromSentence(self, lang, sentence, add_eos=True, pad=0, skip_unk=False):
        max = hparams['tokens_per_sentence']

        indexes = self.indexesFromSentence(None, sentence, skip_unk=skip_unk, add_eos=add_eos, pad=pad)
        if indexes is None and skip_unk: return indexes

        sentence_len = len(indexes)
        while pad > sentence_len:
            indexes.append(UNK_token)
            pad -= 1
        result = Variable(torch.LongTensor(indexes).unsqueeze(1))

        if hparams['cuda']:
            return result[:max].cuda()
        else:
            return result[:max]

    def variablesFromPair(self,pair, skip_unk=False):

        if True:
            pad = hparams['tokens_per_sentence']
            input_variable = self.variableFromSentence(self.input_lang, pair[0], pad=pad, skip_unk=skip_unk)


        pad = hparams['tokens_per_sentence']
        question_variable = self.variableFromSentence(self.output_lang, pair[1], pad=pad, skip_unk=skip_unk)

        if len(pair) > 2:
            #print(pair[2],',pair')
            #if (len(pair[2]) > 0) or True:
            pad = 0
            add_eos = False
            if self.do_recurrent_output:
                pad = hparams['tokens_per_sentence']
                add_eos = True
            target_variable = self.variableFromSentence(self.output_lang, pair[2],
                                                        add_eos=add_eos,
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
                    'state_dict_1_seq': self.model_0_wra.model.state_dict(),
                    'state_dict_6_dec': None, # self.model_0_wra.model_6_dec.state_dict(),
                    'embedding': None, #self.model_0_wra.embed.state_dict(),
                    'optimizer_1': self.model_0_wra.opt_1.state_dict(),
                    'optimizer_2': None, #self.model_0_wra.opt_2.state_dict(),
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
                    'state_dict_1_seq': self.model_0_wra.model.state_dict(),
                    'state_dict_6_dec': None,# self.model_0_wra.model_6_dec.state_dict(),
                    'embedding': None, #self.model_0_wra.embed.state_dict(),
                    'optimizer_1': None , # self.opt_1.state_dict(),
                    'optimizer_2': None,
                    'best_loss': self.best_loss,
                    'long_term_loss': self.long_term_loss,
                    'tag': self.tag,
                    'score': self.score
                }
            ]
        #print(z)
        return z
        pass

    def save_checkpoint(self, state=None, is_best=True, num=0, converted=False, extra='', interrupt=False):
        if state is None or True:
            state = self.make_state(converted=converted)
            if converted: print(converted, 'is converted.')
        basename = hparams['save_dir'] + hparams['base_filename']
        if interrupt:
            basename += '.interrupt'
        if self.do_load_babi or self.do_conserve_space or self.do_train_long or self.do_recurrent_output:
            num = self.this_epoch * len(self.pairs) + num
            torch.save(state,basename+ '.best.pth')
            #####
            if self.do_test_not_train and not interrupt:
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

                self.model_0_wra.model.load_state_dict(checkpoint[0]['state_dict_1_seq'])
                #self.model_0_wra.model_6_dec.load_state_dict(checkpoint[0]['state_dict_6_dec'])

                if self.model_0_wra.opt_1 is not None:
                    #####
                    try:
                        self.model_0_wra.opt_1.load_state_dict(checkpoint[0]['optimizer_1'])
                        if self.model_0_wra.opt_1.param_groups[0]['lr'] != hparams['learning_rate']:
                            raise Exception('new optimizer...')
                    except:
                        #if self.do_freeze_embedding: self.model_0_wra.new_freeze_embedding()
                        self.model_0_wra.opt_1 = self._make_optimizer(self.model_0_wra.model)
                '''
                if self.model_0_wra.opt_2 is not None:
                    #####
                    try:
                        self.model_0_wra.opt_2.load_state_dict(checkpoint[0]['optimizer_2'])
                        if self.model_0_wra.opt_2.param_groups[0]['lr'] != hparams['learning_rate']:
                            raise Exception('new optimizer...')
                    except:
                        if self.do_freeze_embedding: self.model_0_wra.new_freeze_embedding()
                        lm = hparams['multiplier']
                        self.model_0_wra.opt_2 = self._make_optimizer(self.model_0_wra.model_6_dec, lm)
                '''
                print("loaded checkpoint '"+ basename + "' ")
                if self.do_recipe_dropout:
                    self.set_dropout(hparams['dropout'])

            else:
                print("no checkpoint found at '"+ basename + "'")

    def _make_optimizer(self, module=None, lr=1.0):
        print('new optimizer', hparams['learning_rate'] * lr)
        if module is None:
            module = self.model_0_wra.model
        parameters = filter(lambda p: p.requires_grad, module.parameters())
        if False:
            print('===')
            for p in module.parameters():
                if p.requires_grad : print(p.size())
            print('===')
        return optim.Adam(parameters, lr=float(hparams['learning_rate'] * lr) , weight_decay=hparams['weight_decay'])
        #return optim.SGD(parameters, lr=hparams['learning_rate'])



    def _test_embedding(self, num=None, exit=True):
        '''
        if num is None:
            num = 'dave' #55 #hparams['unk']
        num = self.variableFromSentence(self.output_lang, str(num), pad=1)
        print('\n',num)
        '''
        self.model_0_wra.test_embedding()
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
                    if x != hparams['eol'] and x != hparams['sol'] and x != hparams['eow']: # and x.strip() != '':
                        out.append(x)
                    if x == hparams['eow']:
                        #print('!!')
                        out.append(' ')
                ll = x

            if self.do_no_vocabulary:
                return ''.join(out)
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
        if self.do_no_vocabulary:
            return ''.join(out)

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

    def train(self,input_variable, target_variable, question_variable,length_variable, encoder, decoder, wrapper_optimizer_1, wrapper_optimizer_2, memory_optimizer, attention_optimizer, criterion, mask, max_target_length):
        #max_target_length = [hparams['tokens_per_sentence'] for _ in max_target_length]
        #question_variable = None
        self.model_0_wra.past = None

        if criterion is not None : #or not self.do_test_not_train:
            wrapper_optimizer_1.zero_grad()
            #wrapper_optimizer_2.zero_grad()

            self.model_0_wra.model.train()
            #self.model_0_wra.model_6_dec.train()

            outputs, _, ans, _ = self.model_0_wra(input_variable, None, target_variable, length_variable, criterion)
            loss = 0
            n_tot = 0

            if True:

                target_variable = target_variable.squeeze(0)


                if True:
                    #print(ans.size(), target_variable,'atv')

                    a_var = ans
                    t_var = target_variable[:,0] ## -1

                #if True:

                    try:
                        l = criterion(a_var, t_var)
                        loss += l
                        n_tot += t_var.size(0)
                    except ValueError as e:
                        print('skip for size...', z)
                        print(e)
                        print(a_var.size(), t_var.size(),'a,t')
                        exit()
                        pass
                        #print(l, loss, n_tot, 'loss')



            loss.backward()
            print('backward')
            wrapper_optimizer_1.step()
            #wrapper_optimizer_2.step()


        else:
            #self.model_0_wra.eval()
            with torch.no_grad():
                self.model_0_wra.model.eval()
                #self.model_0_wra.model_6_dec.eval()
                outputs, _, ans, _ = self.model_0_wra(input_variable, None, target_variable, length_variable,
                                                      criterion)
                if outputs is not None and ans is None:
                    return None, outputs, None

                if not self.do_recurrent_output:
                    loss = None
                    ans = ans.permute(1,0)

                else:
                    loss = None
                    #ansx = Variable(ans.data.max(dim=2)[1])
                    ans = prune_tensor(ans, 3)
                    ans = ans.permute(1,0,2)

            #self._test_embedding()



        if self.do_clip_grad_norm:
            pass
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
            #if not self.do_test_not_train:
            #    self.true_epoch = n
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
            self.true_epoch = self.this_epoch // (len(self.pairs) // self.epoch_length) + 1

        self.time_str = self._as_minutes(self.time_num)

        if self.model_0_wra.opt_1 is None or self.first_load:

            wrapper_optimizer_1 = self._make_optimizer(self.model_0_wra.model)
            self.model_0_wra.opt_1 = wrapper_optimizer_1

        '''
        if self.model_0_wra.opt_2 is None or self.first_load:
            lm = hparams['multiplier']
            wrapper_optimizer_2 = self._make_optimizer(self.model_0_wra.model_6_dec,lm)
            self.model_0_wra.opt_2 = wrapper_optimizer_2
        '''
        #weight = torch.ones(self.output_lang.n_words)
        #weight[self.output_lang.word2index[hparams['unk']]] = 0.0

        self.criterion = nn.CrossEntropyLoss( size_average=False)

        #self.criterion = self.maskNLLLoss

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
            for param_group in self.model_0_wra.opt_1.param_groups:
                print(param_group['lr'], 'lr opt_1')
            '''
            for param_group in self.model_0_wra.opt_2.param_groups:
                print(param_group['lr'], 'lr_opt_2')
            '''
            print(self.output_lang.n_words, 'num words')

        print(self.train_fr,'loaded file')

        print("-----")

        if self.do_load_babi:
            if self.do_test_not_train:
                self.model_0_wra.model.eval()
                #self.model_0_wra.model_6_dec.eval()

            else:
                self.model_0_wra.model.train()
                #self.model_0_wra.model_6_dec.train()

        if self.do_batch_process:
            step = 1
            if self.start_epoch is 0: start = 0

        for iter in range(epoch_start, epoch_stop + 1, step):

            if self.do_batch_process and (iter ) % hparams['batch_size'] == 0 and iter < len(self.pairs):

                skip_unk = self.do_skip_unk
                group, lengths = self.variables_for_batch(self.pairs, hparams['batch_size'], iter, skip_unk=skip_unk)

                #for i in group: print(i.size() if not isinstance(i,list) else ('->', i[0].size()), len(i))
                #print('---')

                group = group.transpose(1,0)
                #print(group.size(),'group')

                input_variable = group[0]
                question_variable = None #group[2]
                target_variable = group[2]

                length_variable = lengths #group[3]
                mask_variable = None #group[4]
                max_target_length_variable = None #group[5]

                target_variable = prune_tensor(target_variable, 3)
                #print(input_variable)
                #print(temp_batch_size,'temp')
                #if self.do_recurrent_output:
                #    temp_batch_size = len(input_variable)# * hparams['tokens_per_sentence']

            elif self.do_batch_process:
                continue
                pass

            outputs, ans, l = self.train(input_variable, target_variable, question_variable, length_variable, encoder,
                                            decoder, self.model_0_wra.opt_1, None ,
                                            None, None, criterion, mask_variable, max_target_length_variable)

            target_variable = target_variable.unsqueeze(1).transpose(-1,0)

            #print(ans.size(),'ans', target_variable.size(),'ans,tv')

            input_variable = input_variable.permute(1,0)


            temp_batch_size = len(input_variable)

            num_count += 1

            #print(len(max_target_length_variable))

            if self.do_recurrent_output and self.do_load_babi and False:

                for i in range(len(ans)):
                    num_tot += int(max_target_length_variable[i])  # += temp_batch_size * hparams['tokens_per_sentence']

                    for j in range(ans[i].size(0)):
                        t_val = target_variable[i][0,j,0].item()

                        o_val = ans[i][j].item()
                        l_val = length_variable[i].item()

                        if int(o_val) == int(t_val):
                            num_right += 1 * float(1/l_val )
                            num_right_small += 1 * float(1/l_val)
                            if int(o_val) == EOS_token:
                                #num_right_small += 1 - (j+ 1) / l_val #hparams['tokens_per_sentence'] #- (j + 1)
                                #num_right += 1 - (j+ 1) / l_val # hparams['tokens_per_sentence'] #- (j + 1)
                                #print('full line', i, j, num_right_small)
                                break
                        else:
                            # next sentence
                            break

                    # if ignore_break: num_tot += 1


                self.score = float(num_right / num_tot) * 100

            if l is not None:
                print_loss_total += float(l) #.clone())
            else:
                print('loss,', l)

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0

                if self._print_control(iter):
                    print('iter = '+str(iter)+ ', num of iters = '+str(n_iters)
                          + ', ' + self.printable + ', saved files = ' + str(self.saved_files)
                          + ', low loss = %.6f' % self.long_term_loss + ',', end=' ')
                    print('true-epoch =', str(self.true_epoch) + ',','pairs = '
                          + str(len(self.pairs)), end=' ')

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
                pass
                #self._auto_stop()

        if self.do_load_babi:
            print('train list:', ', '.join(self.score_list_training[-100:]))
            print('valid list:', ', '.join(self.score_list[-100:]))
        print('dropout:',hparams['dropout'])
        print('learning rate:', hparams['learning_rate'])
        print('weight decay:', hparams['weight_decay'])
        print(num_count, 'exec count')
        print('raw score:', num_right, num_tot, num_right_small, len(self.pairs))

    def _show_sample(self, iter=0, epoch_start=0, epoch_stop=hparams['batch_size'], temp_batch_size=hparams['batch_size']):
        ###########################
        group = []
        #flag = False
        #while len(group) < 4 and not flag:
        if True:
            if epoch_start + iter >= epoch_stop:
                choice = random.choice(self.pairs)
            else:
                choice = random.choice(self.pairs[epoch_start + iter: epoch_start + iter + temp_batch_size])

            group, length = self.variables_for_batch([choice], 1, 0, skip_unk=self.do_skip_unk)

            group = group.transpose(1,0)
            #print('----')
            #print(group.size(),'gr')
            #print('choice', choice)
            #print('group', group)
            #exit()


        input_variable = group[0]
        ques_variable = None  # group[2]
        target_variable = group[2]
        lengths = length #group[3]
        mask = None #group[4]
        max_target_length = None # group[5]

        #training_batches = self.batch2TrainData(self.output_lang, [choice])
        #input_variable, lengths, target_variable, mask, max_target_len = training_batches

        pad = hparams['tokens_per_sentence']
        #input_variable = self.variableFromSentence(self.output_lang, choice[0], pad=pad, add_eos=True)

        #lengths = Variable(torch.LongTensor([pad]))
        #ques_variable = self.variableFromSentence(self.output_lang, hparams['unk'],  add_eos=True)


        print('src:', choice[0])
        question = None
        if self.do_load_babi:
            #print('ques:', choice[1])
            print('ref:', '[' + choice[2] + ']')
        else:
            print('tgt:', choice[1])
        '''
        nums = self.variablesFromPair(choice)
        if self.do_load_babi:
            question = nums[1]
            target = nums[2]
        if not self.do_load_babi:
            question = nums[0]
            target = None
        '''
        #words, _ = self.evaluate(None, None, input_variable, question=ques_variable, target_variable=target_variable, lengths=lengths)
        #print(self.tokenizer.decode(input_variable))

        choice[0] = choice[0].encode("utf-8")
        choice[0] = choice[0].decode('utf-8')

        words = self.get_sentence(choice[0])
        # print(choice)
        if not self.do_load_babi or self.do_recurrent_output:
            print('ans:', words)
            print('try:', self._shorten(words, just_duplicates=True))
            # self._word_from_prediction()
        ############################
        pass

    def evaluate(self, encoder, decoder, sentence, question=None, target_variable=None, lengths=None, max_length=MAX_LENGTH):

        #print('eval')
        #exit()
        input_variable = sentence
        #question_variable = Variable(torch.LongTensor([UNK_token])) # [UNK_token]
        target_variable = prune_tensor(target_variable, 4).transpose(-1,0)

        t_var = target_variable[0].permute(1,0,2)

        #question_variable = question

        self.model_0_wra.model.eval()
        #self.model_0_wra.model_6_dec.eval()

        print(input_variable.size(), 'ivsize', t_var.size())
        with torch.no_grad():
            self.model_0_wra.past = None
            outputs, _, ans , _ = self.model_0_wra( input_variable, None, t_var, lengths, None)
            ans = torch.argmax(ans, dim=2)

        if hparams['beam'] is None:
            outputs = [ans]

        else:
            outputs = prune_tensor(outputs, 4).transpose(0,2)

        #####################
        #print(outputs[0].size(), 'out')
        decoded_words = self.tokenizer.decode([outputs[0].item()])
        print(decoded_words)

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

    def setup_for_interactive(self, name=None):

        if name is not None:
            hparams['base_filename'] = name
        print(hparams['base_filename'], ': load')

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

        if True:
            self.input_lang, self.output_lang, self.pairs = self.prepareData(self.train_fr, self.train_to,
                                                                             lang3=self.train_ques, reverse=False,
                                                                             omit_unk=self.do_hide_unk,
                                                                             skip_unk=self.do_skip_unk)
            hparams['num_vocab_total'] = self.output_lang.n_words


        words = hparams['num_vocab_total']

        layers = hparams['layers']
        dropout = hparams['dropout']
        pytorch_embed_size = hparams['pytorch_embed_size']
        sol_token = SOS_token #self.output_lang.word2index[hparams['sol']]

        self.model_0_wra = WrapMemRNN(words, pytorch_embed_size, self.hidden_size, layers,
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
            f.write(' '.join(sys.argv) + '\n')
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
        if len(self.best_loss_dict) > 0 and self.do_record_loss:
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
            if self.best_accuracy_graph_size is None or self.best_accuracy_graph_size < 1:
                self.best_accuracy_graph_size = 1
            x = int(int(x) / self.best_accuracy_graph_size)

            if self.args['json_record_offset'] is None:
                self.best_accuracy_record_offset = x
            if self.args['start_epoch'] is None:
                self.start_epoch = x
        if os.path.isfile(basename_loss) and self.do_record_loss:
            with open(basename_loss) as z:
                json_data = json.load(z)
            self.best_loss_dict = json_data

            y = min(int(k) for k, v in self.best_loss_dict.items())
            if int(y) != self.epoch_length:
                self.best_loss_graph_size = int(y)
            else:
                self.best_loss_graph_size = self.epoch_length
            x = max(int(k) for k, v in self.best_loss_dict.items() )
            if self.best_loss_graph_size is None or self.best_loss_graph_size < 1:
                self.best_loss_graph_size = 1
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

        if n.do_interactive:
            n.input_lang, n.output_lang, n.pairs = n.prepareData(n.train_fr, n.train_to,lang3=n.train_ques, reverse=False,
                                                                 omit_unk=n.do_hide_unk, skip_unk=n.do_skip_unk)
            words = n.output_lang.n_words
            hparams['num_vocab_total'] = words

        if n.do_load_babi:
            v_name = hparams['data_dir'] + hparams['vocab_name']
            v_name = v_name.replace('big', hparams['babi_name'])
            hparams['num_vocab_total'] = n.count_sentences(v_name)
            #hparams['num_vocab_total'] = n.output_lang.n_words

            words = hparams['num_vocab_total']

        layers = hparams['layers']
        dropout = hparams['dropout']
        pytorch_embed_size = hparams['pytorch_embed_size']
        sol_token = SOS_token #n.output_lang.word2index[hparams['sol']]

        token_list = []
        if False:
            for i in word_lst: token_list.append(n.output_lang.word2index[i])

        n.model_0_wra = WrapMemRNN(words, pytorch_embed_size, n.hidden_size,layers,
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


        if n.do_infer:
            n.load_checkpoint()
            choice = random.choice(n.pairs)[0]
            print(choice)
            words, _ = n.evaluate(None,None,choice)
            print(words)

    except KeyboardInterrupt as e:
        if not n.do_interactive:
            n.update_result_file()
            n.save_checkpoint(interrupt=True)
            #raise
            #print( e.strerror)
        else:
            print(e)

