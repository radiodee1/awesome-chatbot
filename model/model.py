#!/usr/bin/python3

import numpy as np
from settings import hparams
from keras.preprocessing import text, sequence
from keras.models import Sequential , Model
from keras.layers import Embedding, Input, LSTM, Bidirectional, TimeDistributed, Flatten, dot
from keras.layers import Activation, RepeatVector, Permute, Merge, Dense #, TimeDistributedMerge
from keras.layers import Concatenate, Add, Multiply
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
#from keras.engine.topology import merge
import gensim.models.word2vec as w2v
import os
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
filename = None

if True:
    print ("stage: load w2v model")
    word2vec_book = w2v.Word2Vec.load(os.path.join(hparams['save_dir'], raw_embedding_filename + "_1.w2v"))
    words = len(word2vec_book.wv.vocab)
    vocab_size = words

if False:
    zzz_fr = None
    zzz_to = None
    '''
    with open(vocab_fr,'r') as r:
        zzz_fr = r.read()
        text_zzz_fr = []
        for x in zzz_fr.split('\n'):
            text_zzz_fr.append(x)

    with open(vocab_to, 'r') as r:
        zzz_to = r.read()
        text_zzz_to = []
        for x in zzz_to.split('\n'):
            text_zzz_to.append(x)
    
    with open(text_fr, 'r') as r:
        xxx = r.read()
        text_xxx = []
        for xx in xxx.split('\n'):
            text_xxx.append(xx)

    with open(text_to, 'r') as r:
        yyy = r.read()
        text_yyy = []
        for xx in yyy.split('\n'):
            text_yyy.append(xx)

    with open(train_fr, 'r') as r:
        xxx = r.read()
        train_xxx = []
        for xx in xxx.split('\n'):
            train_xxx.append(xx)

    with open(train_to, 'r') as r:
        yyy = r.read()
        train_yyy = []
        for xx in yyy.split('\n'):
            train_yyy.append(xx)
    '''

def open_sentences(filename):
    with open(filename, 'r') as r:
        yyy = r.read()
        t_yyy = []
        for xx in yyy.split('\n'):
            t_yyy.append(xx)
    return t_yyy

def word_and_vector_size_arrays(text_xxx, text_yyy, double_y=False, double_sentence_y=False):

    text_xxx = open_sentences(text_xxx)
    text_yyy = open_sentences(text_yyy)

    ls_xxx = np.array([])
    ls_yyy = np.array([])

    temp_xxx = np.array([])
    temp_yyy = np.array([])

    mult = 1
    if double_y:
        mult = 2

    for ii in range(len(text_xxx)):
        ############### x #######################
        i = text.text_to_word_sequence(text_xxx[ii])
        ls = np.array([])
        for word in range(tokens_per_sentence):  # i:
            if word + 1 <= len(i):
                if i[word] in word2vec_book.wv.vocab:  # tokenize_voc_fr.word_index:
                    w = word2vec_book.wv[i[word]]
                    if ls.shape[0] == 0:
                        ls = w
                    else:
                        ls = np.vstack((ls, w))

                else:
                    pad = np.zeros(units)
                    if ls.shape[0] == 0:
                        ls = pad
                    else:
                        ls = np.vstack((ls, pad))
            else:
                pad = np.zeros(units)
                if ls.shape[0] == 0:
                    ls = pad
                else:
                    ls = np.vstack((ls, pad))
            pass

        if ls_xxx.shape[0] == 0:
            ls_xxx = ls

        else:
            ls_xxx = np.dstack((ls_xxx, ls))

        ################# y ####################
        j = text.text_to_word_sequence(text_yyy[ii])
        ls = np.array([])
        for word in range(tokens_per_sentence):  # j:
            if word + 1 <= len(j):
                if j[word] in word2vec_book.wv.vocab:  # tokenize_voc_to.word_index:
                    w = word2vec_book.wv[j[word]]
                    if double_y:
                        reverse = w[::-1]
                        w = np.hstack((w, reverse))
                    if ls.shape[0] == 0:
                        ls = w
                    else:
                        ls = np.vstack((ls, w))
                else:
                    pad = np.zeros(units * mult)
                    if ls.shape[0] == 0:
                        ls = pad
                    else:
                        ls = np.vstack((ls, pad))
            else:
                pad = np.zeros(units * mult)
                if ls.shape[0] == 0:
                    ls = pad
                else:
                    ls = np.vstack((ls, pad))
            pass

        if double_sentence_y:
            ls = np.vstack((ls, ls[::-1]))
        if ls_yyy.shape[0] == 0:
            ls_yyy = ls
        else:
            ls_yyy = np.dstack((ls_yyy, ls))
        ############ batch #############

        if ii % (len(text_xxx) // batch_size) == 0:
            if temp_xxx.shape[0] == 0:
                temp_xxx = ls_xxx
            else:
                temp_xxx = np.dstack((temp_xxx, ls_xxx))
            ls_xxx = np.array([])
            ###############
            if temp_yyy.shape[0] == 0:
                temp_yyy = ls_yyy
            else:
                temp_yyy = np.dstack((temp_yyy, ls_yyy))
            ls_yyy = np.array([])

    return temp_xxx, temp_yyy






def get_batch(batch, x, y, batch_size=16):
    """ first batch starts at 0 """
    a = batch_size * batch
    b = batch_size * (batch + 1)
    x = x[:,:,a:b]
    y = y[:,:,a:b]
    return x, y


def swap_axes(x, y):
    x = np.swapaxes(x, 0, 2)
    #x = np.swapaxes(x, 1, 2)

    y = np.swapaxes(y, 0, 2)
    #y = np.swapaxes(y, 1, 2)

    #x = x[:,:,0]

    return x, y


def embedding_model_lstm():
    print (batch_size, tokens_per_sentence)
    '''
    embedding_matrix = np.zeros((len(word2vec_book.wv.vocab), units))
    for i in range(len(word2vec_book.wv.vocab)):
        embedding_vector = word2vec_book.wv[word2vec_book.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    '''

    x_shape = (units,tokens_per_sentence)

    valid_word = Input(shape=x_shape)
    valid_word_y = Input(shape=x_shape)


    '''
    embeddings_a = Embedding(words, units, weights=[embedding_matrix],
                             #input_length=tokens_per_sentence,
                             batch_size=batch_size, input_shape=x_shape[1:],
                             trainable=False
                             )

    embed_a = embeddings_a(valid_word)
    '''

    lstm_a = Bidirectional(LSTM(units=tokens_per_sentence,
                                input_shape=( None,units),
                                return_sequences=True))

    recurrent_a = lstm_a(valid_word)

    concat_a = Concatenate()([recurrent_a,valid_word_y])

    lstm_a2 = LSTM(units=tokens_per_sentence * 2,
                                input_shape=(None,units),
                                return_sequences=True)

    recurrent_a2 = lstm_a2(concat_a)

    time_dist_a = TimeDistributed(Dense(tokens_per_sentence, activation='softmax'))(recurrent_a2)


    #print (K.shape(recurrent_a2), 'not time dist')

    k_model = Model(inputs=[valid_word,valid_word_y], outputs=[time_dist_a])

    k_model.compile(optimizer='adam', loss='binary_crossentropy')

    return k_model

def embedding_model_lstm_softmax():
    print (batch_size, tokens_per_sentence)


    x_shape = (units,tokens_per_sentence)

    valid_word = Input(shape=x_shape)
    valid_word_y = Input(shape=x_shape)



    concat_a = Concatenate()([valid_word,valid_word_y])


    lstm_a = Bidirectional(LSTM(units=tokens_per_sentence * 2,
                                input_shape=( None,units),
                                return_sequences=True))

    recurrent_a = lstm_a(concat_a)

    #concat_a = Concatenate()([recurrent_a,valid_word_y])

    lstm_a2 = LSTM(units=tokens_per_sentence * 2,
                                input_shape=(None,units),
                                return_sequences=True)

    recurrent_a2 = lstm_a2(recurrent_a)

    time_dist_a = TimeDistributed(Dense(tokens_per_sentence, activation='softmax'))(recurrent_a2)


    #print (K.shape(recurrent_a2), 'not time dist')

    k_model = Model(inputs=[valid_word,valid_word_y], outputs=[time_dist_a])

    k_model.compile(optimizer='adam', loss='binary_crossentropy')


    return k_model


def train_embedding_model_api(model, x, y, predict=False, epochs=1, qnum=-1):
    z = x.shape[2] // batch_size
    num = 0
    for e in range(epochs):
        #print ('----')
        for i in range(z):
            xx , yy = get_batch(i, x, y)
            xx, yy = swap_axes(xx, yy)
            #model.train_on_batch(xx,yy)
            if not predict:
                #print (xx.shape, yy.shape)
                model.train_on_batch([xx,yy],yy)
                #model.fit(xx,yy)
            else:
                ypredict = model.predict([xx,yy], batch_size=batch_size)
                #print (ypredict.shape)
                for ii in ypredict:
                    #num += 1
                    if qnum != -1 and num > qnum: return
                    #print (ii,'<', ii.shape)

                    for j in range(units):
                        #print (j,'<<<<',i[:,j].shape)
                        z = word2vec_book.wv.most_similar(positive=[ii[:,j]],topn=1)
                        print (z[0][0], end=' ')
                    num += 1
        print('\n---- epoch ' + str(e) + ' ---------')

def inference_embedding_model_api(model, x, y):
    z = None
    num = 1
    xx, yy = get_batch(0, x, y, batch_size=tokens_per_sentence)
    #print (xx.shape)
    for k in range(xx.shape[0]):
        #print (xx[k,:,0].shape)
        print (word2vec_book.wv.most_similar(positive=[xx[k,:,0]], topn=1)[0][0], end=' ')


    print('\n--------------')
    if True:
        xx = np.expand_dims(xx[:,:,0], axis=0)
        xx = np.swapaxes(xx, 2,1)
    else:
        xx = np.expand_dims(xx[0,:, :], axis=0)

    single_word_y = np.zeros((1,50, 55))
    while z == None or  (z != hparams['eol'] and num < tokens_per_sentence):

        #single_word_y = np.zeros((1,50,55))

        single_word_y[0,:,num] = yy[0,:,num] ## note: dont make this to yy[0,:,0]
        #print (single_word_y[0,:,num].shape,"y here", yy[0,:,num].shape, num)
        z = model.predict([xx, single_word_y],batch_size=1)[0]
        yy = np.expand_dims(z, axis=0)

        z = word2vec_book.wv.most_similar(positive=[z[:,0]])
        #print(z[0])

        z = z[0][0]
        print (z, end=' ')
        num +=1
    print()
    pass

if True:
    print ('stage: arrays train')
    x, y = word_and_vector_size_arrays(train_fr, train_to)
    print ('stage: arrays test')
    x_test, y_test = word_and_vector_size_arrays(text_fr, text_to, double_y=False, double_sentence_y=False)
    #x = x_test
    #y = y_test

    #print (y.shape)

if True:
    model = embedding_model_lstm()
    filename = hparams['save_dir'] + hparams['base_filename'] + '-' + base_file_num + '.h5'
else:
    model = embedding_model_lstm_softmax()
    filename = hparams['save_dir'] + hparams['base_filename'] + '-softmax-' + base_file_num + '.h5'


if True:
    print('stage: checking for load')
    if filename == None:
        filename = hparams['save_dir'] + hparams['base_filename']+'-'+base_file_num +'.h5'
    if os.path.isfile(filename):
        model = load_model(filename)
        print ('stage: load works')
    else:
        print('stage: load failed')
    #exit()

if True:
    print ('stage: train')
    train_embedding_model_api(model, x, y, epochs=2)

if True:
    print ('stage: save lstm model')
    if filename == None:
        filename = hparams['save_dir'] + hparams['base_filename']+'-'+base_file_num +'.h5'
    model.save(filename)

if True:
    print ('stage: simple predict')
    train_embedding_model_api(model, x_test, y_test, predict=True, qnum=1)

if True:
    print ('\n-------')
    inference_embedding_model_api(model,x_test,y_test)

if False:
    print ('\n',len(word2vec_book.wv.vocab))

    print ( word2vec_book.wv.most_similar(positive=['sol'], topn=5))
    print ( word2vec_book.wv.most_similar(positive=['man'], topn=5))
    print ('k', word2vec_book.wv.most_similar(positive=['k'], topn=5))