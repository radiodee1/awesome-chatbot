#!/usr/bin/python3

import numpy as np
from settings import hparams
from keras.preprocessing import text, sequence
from keras.models import Sequential , Model
from keras.layers import Embedding, Input, LSTM, Bidirectional, TimeDistributed, Flatten, dot
from keras.layers import Activation, RepeatVector, Permute, Merge, Dense ,Reshape, Lambda
from keras.layers import Concatenate, Add, Multiply, Average
from keras.models import load_model
from keras import optimizers
from keras.utils import to_categorical
from gensim.models.keyedvectors import KeyedVectors

from random import randint
from keras import backend as K
import tensorflow as tf
#from keras.engine.topology import merge
import pandas as pd
import os
import sys
import csv
import tensorflow as tf
#print(hparams)

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
if len(sys.argv) > 1:
    printable = str(sys.argv[1])
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
        self.filename = None

        self.model = None
        self.model_encoder = None
        self.model_inference = None

        self.vocab_list = None
        self.vocab_dict = None

        self.vocab_list, self.vocab_dict = self.load_vocab(vocab_fr)

        self.load_words(hparams['data_dir'] + hparams['embed_name'])


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

                try:
                    #out_x1[ index_i + (ii * tokens)] = vec
                    pass
                except:
                    pass
                    #print(out_x1.shape, index_i, tokens, ii, words, start, length)
                    # exit()

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

        '''
        self.word_embeddings = pd.read_table(filename,
                                             sep=' ',
                                             index_col=0,
                                             header=None,
                                             quoting=csv.QUOTE_NONE,
                                             na_values=None,
                                             keep_default_na=False)
        w = []
        print(self.word_embeddings.shape)
        for i in range(self.word_embeddings.shape[0]):
            ii = self.word_embeddings[i,:]
            if ii[0] in self.vocab_list:
                w.extend(ii)
                pass
        self.word_embeddings = np.array(w)
        print(self.word_embeddings[0:10], 'top 10')
        self.word_matrix = self.word_embeddings.as_matrix()
        '''

    def find_vec(self,word):

        if self.vocab_dict[word] > len(self.vocab_list):
            return np.zeros((hparams['embed_size']))
            #return np.random.uniform(size=(hparams['embed_size']))

        return self.word_embeddings[0][self.vocab_dict[word]]
        #return self.glove_model.wv[word]
        #return self.word_embeddings.loc[word].as_matrix()

    def find_closest_word(self,vec):

        diff = self.word_embeddings[0] - vec
        delta = np.sum(diff * diff, axis=1)
        i = np.argmin(delta)
        if i < 0 or i > len(self.vocab_list) :
            print('unknown index',i)
            i = 0
        #print(self.word_embeddings.iloc[i].name)
        return self.vocab_list[int(i)]

        #return self.glove_model.wv.most_similar(positive=[vec],topn=1)[0][0]

    def find_closest_index(self,vec):
        diff = self.word_embeddings[0] - vec
        delta = np.sum(diff * diff, axis=1)
        i = np.argmin(delta)
        return i

    def load_word_vectors(self):
        ''' do after all training, before every eval. also before stack_sentences '''
        self.word_embeddings = self.model.get_layer('embedding_2').get_weights()

        #print(self.word_embeddings, len(self.word_embeddings[0]))
        #print(self.word_embeddings[0].shape)
        #for i in range(len(model.layers)):
        #    print(model.get_layer(i).name )

    def embedding_model(self,model=None, infer_encode=None, infer_decode=None, global_check=False):
        if model is not None and infer_encode is not None and infer_decode is not None:
            return model, infer_encode, infer_decode

        if (global_check and self.model is not None and
                self.model_encoder  is not None and self.model_inference  is not None):
            return self.model, self.model_encoder, self.model_inference

        embed_size = int(hparams['embed_size'])
        #lst, dict = self.load_vocab(vocab_fr)
        trainable = True
        embeddings_index = {}
        glove_data = hparams['data_dir'] + hparams['embed_name']
        if not os.path.isfile(glove_data) or hparams['use_embed'] == False:
            embedding_matrix = None # np.zeros((len(self.vocab_list),embed_size))
            trainable = True
        else:
            # load embedding
            f = open(glove_data)
            for line in range(len(self.vocab_list)):
                if line == 0: continue
                word = self.vocab_list[line]
                #print(word, line)
                if word in self.glove_model.wv.vocab:
                    values =  self.glove_model.wv[word]

                    value = np.asarray(values, dtype='float32')

                    embeddings_index[word] = value
                else:
                    #value = np.random.uniform(size=(embed_size,))
                    value = np.zeros((embed_size,))
                    embeddings_index[word] = value
            f.close()

            #print('Loaded %s word vectors.' % len(embeddings_index))

            embedding_matrix = np.zeros((len(self.vocab_list) +1, embed_size))
            for word, i in self.vocab_dict.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all random.
                    embedding_matrix[i] = embedding_vector[:embed_size]
                else:
                    #embedding_matrix[i] = np.random.uniform(size=(embed_size,))
                    embedding_matrix[i] = np.zeros((embed_size,))


        return self.embedding_model_lstm(len(self.vocab_list) +1, embedding_matrix, embedding_matrix, trainable)


    def embedding_model_lstm(self, words, embedding_weights_a=None, embedding_weights_b=None, trainable=False):

        x_shape = (None,units)
        lstm_unit_a =  units
        lstm_unit_b = units # * 2
        embed_unit = int(hparams['embed_size'])

        valid_word_a = Input(shape=(None,))
        valid_word_b = Input(shape=(None,))

        if  embedding_weights_a is not None:
            embeddings_a = Embedding(words,embed_unit ,
                                     weights=[embedding_weights_a],
                                     input_length=tokens_per_sentence,
                                     trainable=trainable
                                     )
        else:
            embeddings_a = Embedding(words, embed_unit,
                                     #weights=[embedding_weights_a],
                                     input_length=tokens_per_sentence,
                                     trainable=True
                                     )

        embed_a = embeddings_a(valid_word_a)

        ### encoder for training ###
        lstm_a = Bidirectional(LSTM(units=lstm_unit_a,
                                    return_sequences=True,
                                    return_state=True
                                    ), merge_mode='ave')

        #recurrent_a, lstm_a_h, lstm_a_c = lstm_a(valid_word_a)

        recurrent_a, rec_a_1, rec_a_2, rec_a_3, rec_a_4 = lstm_a(embed_a) #valid_word_a
        #print(len(recurrent_a),'len')

        concat_a_1 = Average()([rec_a_1, rec_a_3])
        concat_a_2 = Average()([rec_a_2, rec_a_4])

        lstm_a_states = [concat_a_1, concat_a_2]

        ### decoder for training ###

        if embedding_weights_b is not None:
            embeddings_b = Embedding(words, embed_unit,
                                     input_length=tokens_per_sentence, #lstm_unit_a,
                                     weights=[embedding_weights_b],
                                     trainable=trainable
                                     )
        else:
            embeddings_b = Embedding(words, embed_unit,
                                     input_length=tokens_per_sentence,  # lstm_unit_a,
                                     #weights=[embedding_weights_b],
                                     trainable=True
                                     )

        embed_b = embeddings_b(valid_word_b)

        lstm_b = LSTM(units=lstm_unit_b ,
                      #return_sequences=True,
                      return_state=True
                      )

        recurrent_b, inner_lstmb_h, inner_lstmb_c  = lstm_b(embed_b, initial_state=lstm_a_states)

        dense_b = Dense(embed_unit, #words
                        activation='softmax', #softmax
                        #name='dense_layer_b',
                        #batch_input_shape=(None,lstm_unit)
                        )


        decoder_b = dense_b(recurrent_b) # recurrent_b



        model = Model([valid_word_a,valid_word_b], decoder_b) # decoder_b

        ### encoder for inference ###
        model_encoder = Model(valid_word_a, lstm_a_states)

        ### decoder for inference ###

        input_h = Input(shape=(None,lstm_unit_b))
        input_c = Input(shape=(None,lstm_unit_b))

        inputs_inference = [input_h, input_c]

        #print(inputs_inference[0].shape,'zero')

        embed_b = embeddings_b(valid_word_b)

        outputs_inference, outputs_inference_h, outputs_inference_c = lstm_b(embed_b,
                                                                             initial_state=inputs_inference)

        outputs_states = [outputs_inference_h, outputs_inference_c]

        dense_outputs_inference = dense_b(outputs_inference)

        #print(dense_outputs_inference.shape,'out')

        ### inference model ###
        model_inference = Model([valid_word_b] + inputs_inference,
                                [dense_outputs_inference] +
                                outputs_states)

        ### boilerplate ###

        adam = optimizers.Adam(lr=learning_rate)

        # try 'sparse_categorical_crossentropy', 'mse', 'binary_crossentropy'
        model.compile(optimizer=adam, loss='mse')

        return model, model_encoder, model_inference


    def predict_words(self,txt):

        self.vocab_list, self.vocab_dict = self.load_vocab(vocab_fr)

        self.model, self.model_encoder, self.model_inference = self.embedding_model(self.model,
                                                                                    self.model_encoder,
                                                                                    self.model_inference,
                                                                                    global_check=True)
        source_input = self._fill_vec(txt, self.vocab_list, self.vocab_dict)
        state = self.model_encoder.predict(source_input)

        state_out = [
            np.expand_dims(np.expand_dims(state[-1][1],0),0),
            np.expand_dims(np.expand_dims(state[-1][2],0),0)
        ]
        txt_out = []
        t = txt.lower().split()
        for i in range(len(t) - 1, len(t) * 3):
            if i < len(t):
                ## this should happen just once
                vec = self.vocab_dict[t[i]]
                print(vec,self.vocab_list[vec],'vec')
                out = self.model_inference.predict([np.expand_dims(np.array([vec]),0)] + state_out)
                state = [out[1], out[2]]
                out = self.find_closest_index(out[0])
                txt_out.append(str(out))
            else:
                ## this should be repeated
                txt_out.append('|')
                state_out = [
                    np.expand_dims(state[0][-1], 0),
                    np.expand_dims(state[1][-1], 0)
                ]
                out = self.model_inference.predict([np.array([out])] + state_out)
                state = [out[1], out[2]]
                out = self.find_closest_index(out[0])
                txt_out.append(str(out))
                txt_out.append(self.vocab_list[out])
        print(' '.join(txt_out))

        print('------')
        out = self.model.predict([source_input, source_input])
        t_out = []
        for i in range(len(source_input)):
            word = self.find_closest_word(out[i])
            t_out.append(word)
        print(' '.join(t_out))
        print(self.find_closest_word(out[0]))
        pass


    def predict_word(self,txt, lst=None, dict=None, model=None, infer_enc=None, infer_dec=None):
        if lst is None or dict is None:
            lst, dict = self.load_vocab(vocab_fr)

        self.model,self.model_encoder,self.model_inference = self.embedding_model(self.model,
                                                                                  self.model_encoder,
                                                                                  self.model_inference,
                                                                                  global_check=True)

        source_input = self._fill_vec(txt,lst,dict)
        #print(source.shape,'source')
        state = self.model_encoder.predict(source_input)

        print(source_input.shape,'state')
        #vec = source
        txt_out = []
        switch = False
        vec = -1
        t = txt.lower().split()
        steps = 1
        #decode = False
        for i in range(0,len(t) * 3):
            if switch or t[i] in lst:
                if not switch:
                    #print(t[i])
                    steps = 1
                    #decode = True
                if vec == -1 :#len(vec) == 0:
                    vec = dict[t[i]]

                #print(vec)
                if len(state) > 0 :
                    #print(state[0][0])
                    predict , ws = self.predict_sequence(self.model_encoder, self.model_inference, state[0][0], steps,lst,dict)
                    state = []
                else:
                    predict, ws = self.predict_sequence(self.model_encoder, self.model_inference, vec, steps, lst, dict)
                #txt_out.append(ws)
                if switch or t[i] == hparams['eol']:
                    txt_out.append('|')
                    #vec = int(np.argmax(predict))

                    #print(self.find_closest_word(predict),'predict')
                    closest = self.find_closest_word(predict)
                    if closest in self.vocab_list:
                        vec = int(dict[closest])
                        txt_out.append(closest)
                    switch = True
                    steps = 1
                elif not switch:
                    pass
                    vec = -1
        print('output: ',' '.join(txt_out))


    def predict_sequence(self,infer_enc, infer_dec, source, n_steps,lst,dict, decode=False ,simple_reply=True):

        ws = ''
        source = np.array(source)
        source = np.expand_dims(source,0)
        state = infer_enc.predict(source)
        # start of sequence input
        #i = np.argmax(state[0])
        #ws = self.find_closest_word(state[0])# lst[int(i)]
        #print(state[0].shape, '< state[0]', state[1].shape,source.shape, ' ')
        yhat = np.zeros((1,1,hparams['embed_size']))
        target_seq = source# self.vocab_dict[hparams['sol']] #state[0] # np.zeros((1,1,units))

        state = [ np.expand_dims(state[0],0), np.expand_dims(state[1],0)  ]
        #target_seq = np.expand_dims(target_seq,0)
        #output = list()
        if not decode or True:
            for t in range(n_steps):
                target_values = [target_seq] + state
                yhat, h, c = infer_dec.predict(target_values)

                #output.append(yhat[0,:])
                state = [h, c]
                target_seq = h #yhat
                i = self.find_closest_word(yhat[0,:])
                if i in self.vocab_list:
                    ws = self.find_closest_word(yhat[0,:])#lst[int(i)]
                else:
                    ws = hparams['unk']
                #print(w,'< h')

        return yhat[0,:], ws


    def _fill_vec(self, sent, lst, dict):
        s = sent.lower().split()
        out = []
        l = np.zeros((len(s)))
        for i in s:
            if i in lst:
                out.append( dict[i])
            pass
        out = np.array(out)
        l[:out.shape[0]] = out
        out = l
        #print(out.shape,'check')
        return out


    def model_infer(self,filename):
        print('stage: try predict')
        lst, dict = self.load_vocab(vocab_fr)
        c = self.open_sentences(filename)
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
        self.predict_words(line) #, lst, dict, self.model, self.model_encoder,self.model_inference)
        print('----------------')
        line = 'sol what is up ? eol'
        print('input:', line)
        self.predict_words(line) #, lst, dict,self.model,self.model_encoder,self.model_inference)


    def check_sentence(self,x2, y, lst=None, start = 0):
        self.load_word_vectors()
        print(x2.shape, y.shape, train_to)
        ii = tokens_per_sentence
        for k in range(10):
            print(k,lst[k])
        c = self.open_sentences(train_to)
        line = c[start]
        print(line)
        for j in range(start, start + 8):
            print("x >",j,end=' ')
            for i in range(ii):
                vec_x = x2[i + tokens_per_sentence * j]
                print(lst[int(vec_x)], ' ' , int(vec_x),' ',end=' ')
            print()
            print("y >",j, end=' ')
            for i in range(ii):
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

        batch = units
        tot = xx.shape[0] // batch
        out = None
        if not shift_output:
            out = np.zeros(( tot))
        else:
            #out = np.zeros((tot,len(vocab_list)))
            out = np.zeros((tot, hparams['embed_size']))
        for i in range(tot):
            #start = i * batch
            #end = (i + 1) * batch
            x = xx[i]
            if not shift_output:
                out[i] = np.array(x)
            else:
                #out[i,:] = to_categorical(x, len(vocab_list))
                if (int(x) < len(self.vocab_list) and #self.vocab_list[int(x)] in self.vocab_list and
                        self.vocab_dict[self.vocab_list[int(x)]] < len(self.vocab_list)):
                    #print(x, int(x), self.vocab_list[int(x)])
                    out[i,:] = self.find_vec(self.vocab_list[int(x)])
        if not shift_output:
            #print(out.shape)
            #out = np.swapaxes(out,0,1)
            pass
        else:
            pass
        return out

    def train_model_categorical(self, model_in, list, dict,train_model=True, check_sentences=False):
        print('stage: arrays prep for test/train')

        if self.model is None: self.model, self.model_encoder, self.model_inference = self.embedding_model(self.model,
                                                                                                      self.model_encoder,
                                                                                                      self.model_inference,
                                                                                                      global_check=True)
        if not check_sentences: self.model.summary()
        tot = len(self.open_sentences(train_fr))

        self.load_word_vectors()
        #global batch_constant
        length = int(hparams['batch_constant']) * int(hparams['units'])
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
                print('(',s,'= start,', s + length,'= stop )',steps,'total, at',z+1, 'steps', printable)
                x1 = self.categorical_input_one(train_fr,list,dict, length, s)  ## change this to 'train_fr' when not autoencoding
                x2 = self.categorical_input_one(train_to,list,dict, length, s)
                y =  self.categorical_input_one(train_to,list,dict, length, s, shift_output=True)

                x1 = self.stack_sentences_categorical(x1,list)
                x2 = self.stack_sentences_categorical(x2,list)
                y =  self.stack_sentences_categorical(y,list, shift_output=True)

                x1, x2, y = self.three_input_mod(x1,x2,y, dict)

                if check_sentences:
                    self.check_sentence(x2, y, list, 0)
                    exit()
                if train_model:
                    self.model.fit([x1, x2], y, batch_size=16)
                if z % (hparams['steps_to_stats'] * 1) == 0 and z != 0:
                    self.model_infer(train_to)
            except KeyboardInterrupt as e:
                print(repr(e))
                self.save_model(self.model,filename + ".backup")
                exit()
            finally:
                pass
        return model



    def save_model(self,model, filename):
        print ('stage: save lstm model')
        if filename == None:
            filename = hparams['save_dir'] + hparams['base_filename']+'-'+base_file_num +'.h5'
        model.save(filename)


    def load_model_file(self,model, filename, lst):
        print('stage: checking for load')
        if filename == None:
            filename = hparams['save_dir'] + hparams['base_filename']+'-'+base_file_num +'.h5'
        if os.path.isfile(filename):
            model = load_model(filename)
            print ('stage: load works')
        else:
            #model, _, _ = embedding_model_lstm(words=len(lst))
            model, _, _ = self.embedding_model()

            print('stage: load failed')
        return model

    def load_vocab(self,filename, global_check=False):
        ''' assume there is one word per line in vocab text file '''

        if global_check and self.vocab_list is not None and self.vocab_dict is not None:
            return self.vocab_list, self.vocab_dict

        dict = {}
        list = self.open_sentences(filename)
        for i in range(len(list)):
            list[i] = list[i].strip()
            dict[list[i]] = i
        self.vocab_list = list
        self.vocab_dict = dict
        return list, dict

if __name__ == '__main__':

    c = ChatModel()

    if True:
        print('stage: load vocab')
        filename = hparams['save_dir'] + hparams['base_filename'] + '-' + base_file_num + '.h5'

        l, d = c.load_vocab(vocab_fr)
        c.model = c.load_model_file(model,filename, l)

        #model.summary()
        #exit()

    if True:
        c.train_model_categorical(model,l,d, check_sentences=False)

        c.save_model(c.model,filename)

        #print(c.find_closest_word(c.find_vec('str95bb')), 'str95bb')

    if True:

        c.load_word_vectors()
        c.model_infer(train_to)




