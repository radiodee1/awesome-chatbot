#!/usr/bin/python3

import numpy as np
from settings import hparams
from attention_decoder import AttentionDecoder
from keras.preprocessing import text, sequence
from keras.models import  Model
from keras.layers import Embedding, Input, LSTM, Bidirectional, TimeDistributed, Flatten, dot
from keras.layers import Conv1D, Activation, RepeatVector, Permute, Merge, Dense ,Reshape, Lambda, Dropout
from keras.layers import Concatenate, Add, Multiply, Average
from keras.constraints import min_max_norm

from keras import optimizers
from keras.utils import to_categorical
from gensim.models.keyedvectors import KeyedVectors
import argparse
from random import randint
from keras import backend as K
import tensorflow as tf
import tensorflow.contrib.eager as tfe

#from keras.engine.topology import merge
import pandas as pd
import os
import sys
import csv
import tensorflow as tf
#print(hparams)

#tfe.enable_eager_execution()


words = hparams['num_vocab_total']
text_fr = hparams['data_dir'] + hparams['test_name'] + '.' + hparams['src_ending']
text_to = hparams['data_dir'] + hparams['test_name'] + '.' + hparams['tgt_ending']

train_fr = hparams['data_dir'] + hparams['train_name'] + '.' + hparams['src_ending']
train_to = hparams['data_dir'] + hparams['train_name'] + '.' + hparams['tgt_ending']

vocab_fr = hparams['data_dir'] + hparams['vocab_name'] + '.' + hparams['src_ending']
vocab_to = hparams['data_dir'] + hparams['vocab_name'] + '.' + hparams['tgt_ending']
oov_token = hparams['unk']
batch_size = hparams['batch_size']
units = hparams['units']
tokens_per_sentence = hparams['tokens_per_sentence']
raw_embedding_filename = hparams['raw_embedding_filename']

base_file_num = str(hparams['base_file_num'])
batch_constant = int(hparams['batch_constant'])
learning_rate = hparams['learning_rate']
filename = None
model = None

printable = ''

print(sys.argv)

    #print(printable)
#exit()

if batch_size % units != 0 or batch_constant % units != 0:
    print('batch size and batch constant must be mult of',units)
    exit()


class ChatModel:
    def __init__(self):
        # put everything here.

        self.word_embeddings = None
        self.glove_model = None
        self.embedding_matrix = None
        self.filename = hparams['save_dir'] + hparams['base_filename'] + '-' + base_file_num + '.h5'

        self.model = None
        self.model_encoder = None
        self.model_inference = None

        self.name_model = ''
        self.name_encoder = '.encode'
        self.name_infer = '.infer'

        self.uniform_low = -1.0
        self.uniform_high = 1.0
        self.trainable = True
        self.skip_embed = True

        self.vocab_list = None
        self.vocab_dict = None
        self.embed_mode = hparams['embed_mode']

        self.load_words(hparams['data_dir'] + hparams['embed_name'])

        self.train_fr = None
        self.train_to = None

        self.do_train = False
        self.do_infer = False
        self.do_review = False
        self.do_train_long = False

        self.load_good = False

        self.vocab_list, self.vocab_dict = self.load_vocab(vocab_fr)

        self.model, self.model_encoder, self.model_inference = self.embedding_model(self.model,
                                                                                    self.model_encoder,
                                                                                    self.model_inference,
                                                                                    global_check=True)
        self.printable = ''


        parser = argparse.ArgumentParser(description='Train some NMT values.')
        parser.add_argument('--mode',help='mode of operation. (train, infer, review, long)')
        parser.add_argument('--printable',help='a string to print during training for identification.' )
        self.args = parser.parse_args()
        self.args = vars(self.args)

        if self.args['printable'] is not None:
            self.printable = str(self.args['printable'])
        if self.args['mode'] == 'train': self.do_train = True
        if self.args['mode'] == 'infer': self.do_infer = True
        if self.args['mode'] == 'review' : self.do_review = True
        if self.args['mode'] == 'long': self.do_train_long = True


    def task_autoencode(self):
        self.train_fr = hparams['data_dir'] + hparams['train_name'] + '.' + hparams['src_ending']
        self.train_to = hparams['data_dir'] + hparams['train_name'] + '.' + hparams['src_ending']
        pass

    def task_normal_train(self):
        self.train_fr = hparams['data_dir'] + hparams['train_name'] + '.' + hparams['src_ending']
        self.train_to = hparams['data_dir'] + hparams['train_name'] + '.' + hparams['tgt_ending']
        pass

    def task_review_weights(self, stop_at_fail=False):
        num = hparams['base_file_num']
        for i in range(100):
            local_filename = hparams['save_dir'] + hparams['base_filename'] + '-' + str(num) + '.h5'
            if os.path.isfile(local_filename):
                ''' load weights '''

                print('==============================')
                print('here:',local_filename)
                self.load_model_file(local_filename)

                self.model_infer(self.train_fr)
                num = num + hparams['steps_to_stats'] * 10
            else:
                if stop_at_fail: break
        pass

    def task_train_epochs(self,num=0):
        if num == 0:
            num = hparams['epochs']
        for i in range(num):
            self.printable = ' epoch #' + str(i)
            self.train_model_categorical(check_sentences=False)
            self.save_model(self.filename)
        pass

    def open_sentences(self, filename):
        t_yyy = []
        with open(filename, 'r') as r:
            for xx in r:
                t_yyy.append(xx)
        #r.close()
        return t_yyy

    def categorical_input_one(self,filename,vocab_list, vocab_dict, length, start=0, batch=-1, shift_output=False):
        tokens = tokens_per_sentence #units #tokens_per_sentence #units
        text_x1 = self.open_sentences(filename)
        out_x1 = np.zeros(( length * tokens))
        #if batch == -1: batch = batch_size
        if start % units != 0 or (length + start) % units != 0:
            print('bad batch size',start % units, start+length % units, units)
            exit()
        # print(filename)
        for ii in range(length):
            num = 0
            i = text_x1[start + ii].split()
            words = len(i)
            if words >= tokens: words = tokens - 1
            for index_i in range(words):

                if index_i < words and i[index_i].lower() in vocab_list:
                    vec = vocab_dict[i[index_i].lower()]
                    #vec = to_categorical(vec,len(vocab_list))
                    out_x1[ num + (ii * tokens)] = vec
                    num += 1
                else:
                    vec = 0

        if shift_output:
            # print('stage: start shift y')
            out_y_shift = np.zeros(( length * tokens))
            out_y_shift[ : length * tokens - 1] = out_x1[ 1:]
            out_x1 = out_y_shift

        #### test ####
        # print(out_x1.shape,  'sentences')

        return out_x1

    def load_words(self,filename):
        pass
        self.glove_model = KeyedVectors.load_word2vec_format(filename, binary=False)



    def find_vec(self,word):

        if self.vocab_dict[word] > len(self.vocab_list):
            #return np.zeros((hparams['embed_size']))
            return np.random.uniform(low=self.uniform_low,
                                     high=self.uniform_high,
                                     size=(hparams['embed_size']))

        return self.embedding_matrix[self.vocab_dict[word]]


    def find_closest_word(self,vec):

        diff = self.word_embeddings[0] - vec
        delta = np.sum(diff * diff, axis=1)
        i = np.argmin(delta)
        if i < 0 or i >= len(self.vocab_list) :
            print('unknown index',i)
            i = 0
        #print(self.word_embeddings.iloc[i].name)
        return self.vocab_list[int(i)]

    def find_closest_index(self,vec):
        diff = self.word_embeddings[0] - vec
        delta = np.sum(diff * diff, axis=1)
        i = np.argmin(delta)
        return i

    def load_word_vectors(self):
        ''' do after all training, before every eval. also before stack_sentences '''
        self.load_weights_to_matrix()
        if self.embed_mode != 'mod' :
            if self.load_good:
                self.word_embeddings = self.model.get_layer('embedding_1').get_weights()
                #print('stage: embedding_1')
            else:
                print('stage: early load...')
                exit()
            if self.word_embeddings is None: self.set_embedding_matrix()
        else:
            #print(self.word_embeddings.shape)
            if self.word_embeddings is None: self.set_embedding_matrix()
            self.word_embeddings = np.expand_dims(self.embedding_matrix,0)

    def set_embedding_matrix(self):
        print('stage: set_embedding_matrix')
        self.skip_embed = False
        if self.embed_mode == 'mod': self.skip_embed = True
        embed_size = int(hparams['embed_size'])

        self.trainable = hparams['embed_train']  ## toggle trainable here
        embeddings_index = {}
        glove_data = hparams['data_dir'] + hparams['embed_name']
        if not os.path.isfile(glove_data) or self.embed_mode == 'zero':
            self.embedding_matrix = None  # np.zeros((len(self.vocab_list),embed_size))
            self.trainable = True
        else:
            # load embedding
            f = open(glove_data)
            for line in range(len(self.vocab_list)):
                if line == 0: continue
                word = self.vocab_list[line]
                # print(word, line)
                if word in self.glove_model.wv.vocab:
                    #print('fill with values',line)
                    values = self.glove_model.wv[word]
                    value = np.asarray(values, dtype='float32')
                    embeddings_index[word] = value
                else:
                    print('fill with random values',line, word)
                    value = np.random.uniform(low=self.uniform_low, high=self.uniform_high, size=(embed_size,))
                    # value = np.zeros((embed_size,))
                    embeddings_index[word] = value
            f.close()

            self.embedding_matrix = np.zeros((len(self.vocab_list), embed_size))
            for i in range(len(self.vocab_list)):
                word = self.vocab_list[i]
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all random.
                    self.embedding_matrix[i] = embedding_vector[:embed_size]
                else:
                    print('fill with random values',i,word)

                    self.embedding_matrix[i] = np.random.uniform(high=self.uniform_high, low=self.uniform_low,
                                                                 size=(embed_size,))
                    # self.embedding_matrix[i] = np.zeros((embed_size,))
        pass

    def load_weights_to_matrix(self,embeddings_index=None):
        ''' assume vectors in embedding layer 2 are in vocab_list order already. '''
        if self.load_good:
            #print('stage: set embedding data after load.')
            if embeddings_index is None:
                self.word_embeddings = self.model.get_layer('embedding_1').get_weights()
                embeddings_index = {}
                for i in range(len(self.vocab_list)):
                    word = self.vocab_list[i]
                    value = self.word_embeddings[0][i]
                    #print(value.shape, i, word)
                    embeddings_index[word] = value
            embed_size = int(hparams['embed_size'])
            self.embedding_matrix = np.zeros((len(self.vocab_list), embed_size))
            for i in range(len(self.vocab_list)):
                word = self.vocab_list[i]
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all random.
                    self.embedding_matrix[i] = embedding_vector[:embed_size]
                else:
                    print('fill with random values', i, word)
                    self.embedding_matrix[i] = np.random.uniform(high=self.uniform_high, low=self.uniform_low,
                                                                 size=(embed_size,))
                    # self.embedding_matrix[i] = np.zeros((embed_size,))
        pass


    def embedding_model(self,model=None, infer_encode=None, infer_decode=None, global_check=False):

        return_sequences_b = False

        if self.embed_mode == 'normal':
            self.skip_embed = False
            if self.word_embeddings is None: self.set_embedding_matrix()
            return_sequences_b = True
        elif self.embed_mode == 'mod':
            return_sequences_b = True

        if self.embedding_matrix is None:# and self.embed_mode != 'normal':
            self.load_word_vectors()

        if not self.embed_mode == 'mod':
            if model is not None and infer_encode is not None and infer_decode is not None:
                return model, infer_encode, infer_decode

            if (global_check and self.model is not None and
                    self.model_encoder  is not None and self.model_inference  is not None):
                return self.model, self.model_encoder, self.model_inference

        if not self.load_good :
            print('stage: construct model here.')
            self.model, self.model_encoder, self.model_inference \
                = self.embedding_model_lstm(words,
                                             self.embedding_matrix,
                                             self.embedding_matrix,
                                             self.trainable,
                                             skip_embed=self.skip_embed,
                                             return_sequences_b=return_sequences_b)

            if self.vocab_dict is None: self.load_vocab()

        self.load_good = True

        return self.model, self.model_encoder, self.model_inference

    def embedding_model_lstm(self, words,
                             embedding_weights_a=None,
                             embedding_weights_b=None,
                             trainable=False,
                             skip_embed=False,
                             return_sequences_b=False):

        lstm_unit_a = units
        lstm_unit_b = units  * 2
        embed_unit = int(hparams['embed_size'])

        x_shape = (tokens_per_sentence,)
        decoder_dim = units *2 # (tokens_per_sentence, units *2)

        valid_word_a = Input(shape=x_shape)
        valid_word_b = Input(shape=x_shape)

        embeddings_a = Embedding(words,embed_unit ,
                                 weights=[embedding_weights_a],
                                 input_length=tokens_per_sentence,
                                 trainable=trainable
                                 )

        embed_a = embeddings_a(valid_word_a)

        ### encoder for training ###
        lstm_a = Bidirectional(LSTM(units=lstm_unit_a,
                                    return_sequences=True#,
                                    #return_state=True,
                                    #recurrent_dropout=0.2,
                                    #input_shape=(None,)
                                    ),
                               merge_mode='concat',
                               trainable=True)

        recurrent_a = lstm_a(embed_a)

        #############
        #conv1d_b = Conv1D(tokens_per_sentence,lstm_unit_b)(recurrent_a)

        lstm_b = AttentionDecoder(units=lstm_unit_b , output_dim=decoder_dim,
                      kernel_constraint=min_max_norm(),
                      #return_sequences=return_sequences_b,
                      #return_state=True
                      )

        #recurrent_b, inner_lstmb_h, inner_lstmb_c  = lstm_b(recurrent_a) #recurrent_a ## <--- here
        recurrent_b = lstm_b(recurrent_a) #recurrent_a ## <--- here

        dense_b = Dense(embed_unit, input_shape=(tokens_per_sentence,),
                        activation='softmax' #softmax or relu
                        #name='dense_layer_b',
                        )

        decoder_b = dense_b(recurrent_b) # recurrent_b

        dropout_b = Dropout(0.15)(decoder_b)

        #model = Model([valid_word_a,valid_word_b], dropout_b) # decoder_b
        model = Model([valid_word_a], dropout_b) # decoder_b

        ### encoder for inference ###
        #model_encoder = Model(valid_word_a, lstm_a_states)

        ### decoder for inference ###

        input_h = Input(shape=(None, ))

        input_c = Input(shape=(None, ))

        inputs_inference = [input_h, input_c]

        #embed_b = embeddings_a(valid_word_b)
        #outputs_inference, outputs_inference_h, outputs_inference_c = lstm_b(embed_b)# ,
                                                                            #initial_state=inputs_inference)

        #outputs_states = [outputs_inference_h, outputs_inference_c]

        #dense_outputs_inference = dense_b(outputs_inference)

        ### inference model ###
        '''
        model_inference = Model([valid_word_b] + inputs_inference,
                                [dense_outputs_inference] +
                                outputs_states)
        '''


        ### boilerplate ###

        adam = optimizers.Adam(lr=learning_rate)

        # try 'categorical_crossentropy', 'mse', 'binary_crossentropy'
        model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['acc'])

        return model , None, None #, None, model_inference

    def predict_words(self,txt,stop_at_eol=False):
        eol = hparams['eol']
        self.vocab_list, self.vocab_dict = self.load_vocab(vocab_fr)

        self.model, self.model_encoder, self.model_inference = self.embedding_model(self.model,
                                                                                    self.model_encoder,
                                                                                    self.model_inference,
                                                                                    global_check=True)
        source_input = self._fill_vec(txt, shift_right=False)
        if self.embed_mode == 'mod':
            source_input = self.stack_sentences_categorical(source_input,self.vocab_list,shift_output=True)
        else:
            source_input = self.stack_sentences_categorical(source_input,self.vocab_list,shift_output=False)

        repeats = hparams['infer_repeat']

        if False:
            for _ in range(repeats):
                state = self.model_encoder.predict(source_input)

                #self.model_encoder.summary()
                #self.model_inference.summary()

                h = state[0]
                c = state[1]

                infer_lst = []
                txt_out = []
                t = txt.lower().split()
                out = self.vocab_dict[hparams['sol']]
                if self.embed_mode == 'mod':
                    #out = self.vocab_dict[hparams['sol']]
                    infer_lst.append(self.vocab_dict[hparams['sol']])
                    out = self.find_vec(hparams['sol'])
                    out = np.expand_dims(out,0)
                    out = np.expand_dims(out,0)
                else:
                    out = np.expand_dims(out,0)
                    #out = np.expand_dims(out,0)

                    #out = self.stack_sentences_categorical(out,self.vocab_list,shift_output=True)

                for i in range(len(t) - 0, len(t) * 3):
                    if True:
                        txt_out.append('|')
                        out_word = ''
                        state_out = [h,c]
                        #print(state_out,'so')

                        if self.embed_mode == 'normal':
                            #a = np.zeros((tokens_per_sentence))
                            #a[0] = int(out)

                            a = self._fill_sentence(out, infer_lst, pad_last_val=False)
                            out = np.expand_dims(a,0)
                            #print(out,'a')
                        out, h, c = self.model_inference.predict([out] + state_out)
                        if self.embed_mode == 'normal':
                            out = out[0,0,:]
                            out = self.find_closest_index(out)
                            infer_lst.append(out)
                            txt_out.append(str(out))
                            if int(out) < len(self.vocab_list):
                                txt_out.append(self.vocab_list[int(out)])
                                out_word = self.vocab_list[int(out)]
                        elif self.embed_mode == 'mod':

                            if False:
                                z_list = []
                                for z in range(tokens_per_sentence):
                                    close_word = out[0,z,:]
                                    z_list.append(self.find_closest_word(close_word))
                                print('**','--'.join(z_list),'**')

                            out_word = self.find_closest_index(out[0,0,:])
                            txt_out.append(str(out_word))
                            if int(out_word) < len(self.vocab_list):
                                txt_out.append(self.vocab_list[int(out_word)])
                            #out_word = np.array([out_word])
                            #out = self.stack_sentences_categorical(out_word, self.vocab_list,shift_output=True)
                            #if self.embed_mode == 'mod':
                            out = self.find_vec(self.vocab_list[int(out_word)])
                            out = np.expand_dims(out, 0)
                            out = np.expand_dims(out, 0)
                        if stop_at_eol and out_word == eol: break


                print('---greedy predict---')
                print(' '.join(txt_out))
                ####

        if True:
            for _ in range(repeats):
                print('---basic predict---')
                out = self.model.predict([source_input]) #, source_input])
                print(out.shape)
                t_out = []
                for i in range(tokens_per_sentence):
                    word = self.find_closest_word(out[0,i,:])
                    t_out.append(word)
                    t_out.append(str(self.find_closest_index(out[0,i,:])))
                    t_out.append('|')
                print(' '.join(t_out))
                #print(self.find_closest_word(out[0]))
        pass

    def _fill_vec(self, sent, shift_right=False):
        s = sent.lower().split()
        out = []
        l = np.zeros((tokens_per_sentence))
        for i in range(tokens_per_sentence):
            if i < len(s) and s[i] in self.vocab_list:
                out.append( self.vocab_dict[s[i]])
            pass
        out = np.array(out)
        start = 0
        stop = out.shape[0]
        if shift_right:
            start = tokens_per_sentence - out.shape[0]
            stop = tokens_per_sentence
        #l[:out.shape[0]] = out
        l[start:stop] = out
        out = l
        #print(out.shape,'check')
        return out

    def _fill_sentence(self, word, infer_lst=[],pad_last_val=False):
        #print(infer_lst,'<')
        out = np.zeros((tokens_per_sentence))
        for i in range(tokens_per_sentence):
            if len(infer_lst) == 0:
                out[i] = word
            elif i < len(infer_lst):
                out[i] = infer_lst[i]
            elif pad_last_val:
                out[i] = infer_lst[-1]

        if False:
            print('---')
            for i in out:
                print(self.vocab_list[int(i)], ' ', end='')
            print('---')
        return out

    def model_infer(self,filename):
        print('stage: try predict')
        f = filename.replace('from','to')
        self.load_word_vectors()
        self.load_vocab(vocab_fr)
        c = self.open_sentences(self.train_fr)
        f = self.open_sentences(self.train_to)
        g = randint(0, len(c))
        line = c[g]
        line = line.strip('\n')
        self.model, self.model_encoder, self.model_inference = self.embedding_model(self.model,
                                                                                    self.model_encoder,
                                                                                    self.model_inference,
                                                                                    global_check=True)
        print('----------------')
        print('index:',g)
        print('input:',line)
        print('ref:', f[g])
        self.predict_words(line, stop_at_eol=True)
        print('----------------')
        line = 'sol what is up ? eol'
        print('input:', line)
        self.predict_words(line, stop_at_eol=True)

        if False:
            word = 'the'
            print(word)
            vec = self.find_vec(word)
            i = self.find_closest_index(vec)
            print(self.vocab_list[i],self.find_closest_word(vec), 'close')

        if False:
            self.model_encoder.summary()
            self.model_inference.summary()


    def check_sentence(self,x2, y, lst=None, start = 0):
        self.load_word_vectors()
        print(x2.shape, y.shape, self.train_to)
        ii = tokens_per_sentence
        for k in range(10):
            print(k,lst[k])
        c = self.open_sentences(self.train_to)
        line = c[start]
        print(line)
        for j in range(start, start + 8):
            print("x >",j,end=' ')
            for i in range(ii):
                if self.embed_mode == 'normal':
                    vec_x = x2[j,i]
                else:
                    vec_x = x2[i + tokens_per_sentence * j]
                print(lst[int(vec_x)], ' ' , int(vec_x),' ',end=' ')
            print()
            print("y >",j, end=' ')
            for i in range(ii):
                if self.embed_mode == 'normal':
                    vec_y = y[j,i,:]
                else:
                    vec_y = y[i + tokens_per_sentence * j,:]
                vec_y2 = self.find_closest_index(vec_y)
                print(self.find_closest_word(vec_y), ' ', vec_y2,' ', end=' ')
            print()

    def three_input_mod(self,xx1, xx2, yy, dict):
        tot = len(xx1)
        steps = tot // tokens_per_sentence
        x1 = []
        x2 = []
        y = []

        for i in range(steps):

            end1 = False
            end2 = False
            for j in range(tokens_per_sentence):

                c = i * tokens_per_sentence + j
                if j == 0:
                    if xx1[c] != dict[hparams['sol']]:
                        print('bad sentence start')
                    if xx2[c] != dict[hparams['sol']]:
                        print('bad sentence start')
                if (not end1) and (not end2):
                    x1.append(xx1[c])
                    x2.append(xx2[c])
                    y.append(yy[c])
                    #print(yy[c].shape, 'yy')
                if xx1[c] == dict[hparams['eol']]: end1 = True
                if xx2[c] == dict[hparams['eol']]: end2 = True

        xx1 = np.array(x1)
        xx2 = np.array(x2)
        yy =  np.array(y)
        #print(yy.shape)
        #exit()
        return xx1, xx2, yy


    def stack_sentences_categorical(self,xx, vocab_list, shift_output=False):
        self.load_word_vectors()
        normal = False
        if self.embed_mode == 'normal': normal = True
        batch = tokens_per_sentence
        if self.embed_mode == 'mod':
            batch = tokens_per_sentence
        tot = xx.shape[0] // batch
        #print(xx.shape,'stack')
        out = None
        if not shift_output:
            out = np.zeros(( tot))
        else:
            if tot == 0: tot = 1
            out = np.zeros((tot,batch, hparams['embed_size']))
        if normal and not shift_output:
            out = np.zeros((tot,batch))
        for i in range(tot):
            for k in range(batch):
                start = i * batch
                end = (i + 1) * batch
                x = xx[i * batch + k]

                if not shift_output:
                    if not normal:
                        out[i] = np.array(x)
                    else:
                        #for j in range(batch):
                        out[i,k] = int(x)
                else:
                    if (int(x) < len(self.vocab_list) and
                            self.vocab_dict[self.vocab_list[int(x)]] < len(self.vocab_list)):
                        #for j in range(batch):
                        out[i,k,:] = self.find_vec(self.vocab_list[int(x)])
        if not shift_output:
            #print(out.shape)
            #out = np.swapaxes(out,0,1)
            pass
        else:
            pass

        if self.embed_mode == 'mod' :
            pass
            #out = np.expand_dims(out,0)
        return out

    def train_model_categorical(self, train_model=True, check_sentences=False):
        print('stage: arrays prep for test/train')
        list = self.vocab_list
        dict = self.vocab_dict

        if self.model is None: self.model, self.model_encoder, self.model_inference = \
            self.embedding_model(self.model,
                                  self.model_encoder,
                                  self.model_inference,
                                  global_check=True)


        if not check_sentences: self.model.summary()
        tot = len(self.open_sentences(self.train_fr))

        self.load_word_vectors()
        #global batch_constant

        all_vectors = False
        if self.embed_mode == 'mod':
            all_vectors = True
            length = int(hparams['batch_constant'])
        else:
            length = int(hparams['batch_constant']) #* int(hparams['units'])
        steps = tot // length
        if steps * length < tot: steps += 1
        #print( steps, tot, length, batch_size)
        for z in range(steps):
            try:
                s = (length) * z
                if tot < s + length: length = tot - s
                if length % int(hparams['units']) != 0:
                    i = length // int(hparams['units'])
                    length = i * int(hparams['units'])
                print('(',s,'= start,', s + length,'= stop )',steps,'total, at',z+1, 'steps', self.printable)
                x1 = self.categorical_input_one(self.train_fr,list,dict, length, s)  ## change this to 'train_fr' when not autoencoding
                x2 = self.categorical_input_one(self.train_to,list,dict, length, s)
                y =  self.categorical_input_one(self.train_to,list,dict, length, s, shift_output=True)

                x1 = self.stack_sentences_categorical(x1,list, shift_output=all_vectors)
                x2 = self.stack_sentences_categorical(x2,list, shift_output=all_vectors)
                y =  self.stack_sentences_categorical(y,list, shift_output=True)

                #x1, x2, y = self.three_input_mod(x1,x2,y, dict) ## seems to work better without this.

                if check_sentences:
                    self.check_sentence(x2, y, list, 0)
                    exit()
                if train_model:
                    if self.embed_mode == 'mod':

                        self.model.fit([x1], y)
                    else:
                        self.model.fit([x1], y)

                if (z + 1) % (hparams['steps_to_stats'] * 10) == 0 and z != 0:
                    hparams['base_file_num'] = hparams['base_file_num'] + hparams['steps_to_stats'] * 10
                    self._set_filename()
                    print(self.filename)
                    pass
                if (z + 1) % (hparams['steps_to_stats'] * 1) == 0 and z != 0:
                    self.save_model( self.filename)
                    self.model_infer(self.train_fr)

            except KeyboardInterrupt as e:
                print(repr(e))
                self.save_model(filename + ".backup")
                exit()
            finally:
                pass
        return model

    def _set_filename(self):
        self.filename = hparams['save_dir'] + hparams['base_filename'] + '-' + \
                        str(hparams['base_file_num']) + '.h5'



    def save_model(self, filename):
        print ('stage: save lstm model')
        if filename == None:
            base_file_num = hparams['base_file_num']
            filename = hparams['save_dir'] + hparams['base_filename']+'-'+base_file_num +'.h5'
        if filename.endswith('.h5'):
            basename = filename[:- len('.h5')]
            print(basename)
            if self.model is not None:
                self.model.save_weights(filename)
            if self.model_encoder is not None:
                self.model_encoder.save_weights(basename + self.name_encoder + '.h5')
            if self.model_inference is not None:
                self.model_inference.save_weights(basename + self.name_infer + '.h5')

            ## update master file num 1 ##
            filename = hparams['save_dir'] + hparams['base_filename']+'-'+str(1) +'.h5'

            basename = filename[:- len('.h5')]
            print(basename)
            if self.model is not None:
                self.model.save_weights(filename)
            if self.model_encoder is not None:
                self.model_encoder.save_weights(basename + self.name_encoder + '.h5')
            if self.model_inference is not None:
                self.model_inference.save_weights(basename + self.name_infer + '.h5')
        else:
            self.model.save(filename)


    def load_model_file(self, filename=None):
        print('stage: checking for load')
        basename = ''
        if filename is None:
            base_file_num = str(hparams['base_file_num'])
            filename = hparams['save_dir'] + hparams['base_filename']+'-'+base_file_num +'.h5'
        if filename.endswith('.h5'):
            basename = filename[:- len('.h5')]
        if os.path.isfile(filename):
            self.model.load_weights(filename)

            print ('stage: load works', filename)
        name_encoder = basename + self.name_encoder + '.h5'
        name_inference = basename + self.name_infer + '.h5'
        if os.path.isfile(name_encoder) :
            self.model_encoder.load_weights(name_encoder)
            print('stage: load', name_encoder)
        if os.path.isfile(name_inference) :
            self.model_inference.load_weights(name_inference)
            print('stage: load', name_inference)
        if not os.path.isfile(filename):
            self.model, self.model_encoder, self.model_inference = self.embedding_model()
            print('stage: load failed')
        #return model

    def load_vocab(self,filename, global_check=False):
        ''' assume there is one word per line in vocab text file '''

        if global_check and self.vocab_list is not None and self.vocab_dict is not None:
            return self.vocab_list, self.vocab_dict

        dict = {}
        list = self.open_sentences(filename)
        for i in range(len(list)):
            list[i] = list[i].strip()
            word = list[i]
            #dict[word] = self.model_embedding.precict(np.array([i]))
            dict[list[i]] = i
        self.vocab_list = list
        self.vocab_dict = dict # dict
        return list, dict




if __name__ == '__main__':

    c = ChatModel()

    if True:
        if hparams['autoencode']: c.task_autoencode()
        else: c.task_normal_train()

        print('stage: load vocab')
        filename = hparams['save_dir'] + hparams['base_filename'] + '-' + base_file_num + '.h5'

        c.load_vocab(vocab_fr)
        c.load_model_file()

    if c.do_train:
        c.train_model_categorical( check_sentences=False)

        c.save_model(filename)
        c.model_infer(c.train_fr)

    if c.do_infer:

        c.model_infer(c.train_fr)

    if c.do_train_long:
        c.task_train_epochs()
        c.save_model(c.filename)

    if c.do_review:
        c.task_review_weights(True)

    if False:
        c.load_word_vectors()
        print(c.embedding_matrix[0])
        z = c.embedding_matrix[0]
        print(c.find_closest_word(z))
        print(c.find_vec('unk'))
        print(c.vocab_dict['unk'])