#!/usr/bin/python3

import numpy as np
from settings import hparams
from keras.preprocessing import text, sequence
from keras.models import Sequential , Model
from keras.layers import Embedding, Input, LSTM, Bidirectional, TimeDistributed, Flatten, dot
from keras.layers import Activation, RepeatVector, Permute, Merge, Dense ,Reshape, Lambda
from keras.layers import Concatenate, Add, Multiply
from keras.models import load_model
from keras import optimizers
from keras.utils import to_categorical
from random import randint
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



def open_sentences(filename):
    with open(filename, 'r') as r:
        yyy = r.read()
        t_yyy = []
        for xx in yyy.split('\n'):
            t_yyy.append(xx)
    return t_yyy

def vector_input_three(filename_x1, filename_x2, filename_y ):
    text_x1 = open_sentences(filename_x1)
    text_x2 = open_sentences(filename_x2)
    text_y  = open_sentences(filename_y)
    out_x1 = np.zeros((units, len(text_x1) * tokens_per_sentence))
    out_x2 = np.zeros((units, len(text_x1) * tokens_per_sentence))
    out_y  = np.zeros((units, len(text_x1) * tokens_per_sentence))

    for ii in range(len(text_x1)):
        ################ x1 ##################
        i = text_x1[ii].split()
        words = len(i)
        for index_i in range(words):
            if index_i < len(i) and i[index_i] in word2vec_book.wv.vocab:
                vec = word2vec_book.wv[i[index_i]]
                #print(vec.shape,'vocab', i[index_i])
            else:
                vec = np.zeros((units))
                #print(vec.shape, 'fixed')
            ## add to output
            out_x1[:,index_i] = vec
        ############### x2 ##################
        i = text_x2[ii].split()
        words = len(i)
        for index_i in range(words):
            if index_i < len(i) and i[index_i] in word2vec_book.wv.vocab:
                vec = word2vec_book.wv[i[index_i]]
                #print(vec.shape, 'vocab', i[index_i])
            else:
                vec = np.zeros((units))
                #print(vec.shape, 'fixed')
            ## add to output
            out_x2[:, index_i] = vec
        ################# y ###############
        i = text_y[ii].split()
        words = len(i)
        for index_i in range(words):
            if index_i < len(i) and i[index_i] in word2vec_book.wv.vocab:
                vec = word2vec_book.wv[i[index_i]]
                #print(vec.shape, 'vocab', i[index_i])
            else:
                vec = np.zeros((units))
                #print(vec.shape, 'fixed')
            ## add to output
            out_y[:, index_i] = vec

    return out_x1, out_x2, out_y


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
        #i = text.text_to_word_sequence(text_xxx[ii])
        i = text_xxx[ii].split()
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
        #j = text.text_to_word_sequence(text_yyy[ii])
        j = text_yyy[ii].split()
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



def get_batch_three(batch, x, y,z, batch_size=16):
    """ first batch starts at 0 """
    a = batch_size * batch
    b = batch_size * (batch + 1)
    x = x[:,:,a:b]
    y = y[:,:,a:b]
    z = z[:,:,a:b] #np.zeros_like(y)
    temp = z[1:,:,:]
    z[:tokens_per_sentence -1,:,:] = temp
    return x, y, z


def get_batch(batch, x, y, batch_size=16):
    """ first batch starts at 0 """
    a = batch_size * batch
    b = batch_size * (batch + 1)
    x = x[:,:,a:b]
    y = y[:,:,a:b]
    return x, y


def swap_axes(x, y, z):
    x = np.swapaxes(x, 0, 2)
    x = np.swapaxes(x, 1, 2)

    y = np.swapaxes(y, 0, 2)
    y = np.swapaxes(y, 1, 2)


    z = np.swapaxes(z, 0, 2)
    z = np.swapaxes(z, 1, 2)
    return x, y, z

def embedding_model_lstm():

    #print (batch_size, tokens_per_sentence)

    '''
    embedding_matrix = np.zeros((len(word2vec_book.wv.vocab), units))
    for i in range(len(word2vec_book.wv.vocab)):
        embedding_vector = word2vec_book.wv[word2vec_book.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    '''

    #x_shape = (units,tokens_per_sentence)
    x_shape = (tokens_per_sentence,units)

    valid_word_a = Input(shape=(None,units))
    valid_word_b = Input(shape=(None,units))


    '''
    embeddings_a = Embedding(words, units, weights=[embedding_matrix],
                             #input_length=tokens_per_sentence,
                             batch_size=batch_size, input_shape=x_shape[1:],
                             trainable=False
                             )

    embed_a = embeddings_a(valid_word)
    '''

    tokens_partb = units

    ### encoder for training ###
    lstm_a = LSTM(units=tokens_partb,
                                return_state=True)

    recurrent_a, lstm_a_h, lstm_a_c = lstm_a(valid_word_a)

    lstm_a_states = [lstm_a_h , lstm_a_c]

    ### decoder for training ###

    lstm_b = LSTM(units=tokens_partb ,return_state=True ,
                                input_shape=x_shape[1:])

    recurrent_b, _, _ = lstm_b(valid_word_b, initial_state=lstm_a_states)

    reshape_b = Reshape((-1,tokens_partb))(recurrent_b)

    #permute_b = Permute((1,2))(reshape_b)
    #print(permute_b.shape)

    #lambda_b = Lambda(lambda x: K.squeeze(x, axis=0))(reshape_b)  # decrease 1 dimension

    #permute_b2 = Permute((1,2))(lambda_b)

    #print(lambda_b.shape)

    dense_b = Dense(tokens_partb, activation='softmax')

    decoder_b = dense_b(reshape_b)

    #decoder_b = dense_b(reshape_b)

    model = Model([valid_word_a,valid_word_b], decoder_b)

    ### encoder for inference ###
    model_encoder = Model(valid_word_a, lstm_a_states)

    ### decoder for inference ###

    input_h = Input(shape=(tokens_partb,))
    input_c = Input(shape=(tokens_partb,))

    inputs_inference = [input_h, input_c]

    outputs_inference, outputs_inference_h, outputs_inference_c = lstm_b(valid_word_b,
                                                                         initial_state=inputs_inference)

    outputs_states = [outputs_inference_h, outputs_inference_c]

    dense_outputs_inference = dense_b(outputs_inference)

    model_inference = Model([valid_word_b] + inputs_inference,
                            [dense_outputs_inference] + outputs_states)

    ### boilerplate ###

    adam = optimizers.Adam(lr=0.001)

    model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])
    model_encoder.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])
    model_inference.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])


    return model, model_encoder, model_inference






def predict_word(txt):
    model, infenc, infdec = embedding_model_lstm()
    switch = False
    vec = []
    t = txt.lower().split()
    for i in range(len(t) * 3):
        if switch or t[i] in word2vec_book.wv.vocab:
            if not switch: print(t[i])
            if len(vec) == 0:
                vec = word2vec_book.wv[t[i]]
                #vec = vec[:,0]
                vec = np.expand_dims(vec, 0)
                vec = np.expand_dims(vec, 0)
            predict = predict_sequence(infenc, infdec, vec, 1, units)
            word = word2vec_book.wv.most_similar(positive=[predict], topn=1)[0][0]
            if switch or t[i] == hparams['eol']:
                predict = np.expand_dims(predict,0)
                predict = np.expand_dims(predict,0)
                vec = predict
                print(vec.shape)
                switch = True
            else:
                vec = []


def predict_sequence(infenc, infdec, source, n_steps, simple_reply=True):
    # encode
    state = infenc.predict(source)
    # start of sequence input
    target_seq = np.zeros((1,1,units)) #np.array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
    # collect predictions
    output = list()
    for t in range(n_steps):
        # predict next char
        yhat, h, c = infdec.predict([target_seq] + state)
        # store prediction
        output.append(yhat[0,:])
        # update state
        state = [h, c]
        # update target sequence
        target_seq = yhat
        print(word2vec_book.wv.most_similar(positive=[yhat[0,:]], topn=1)[0])
    if not simple_reply: return np.array(output)
    else: return yhat[0,:]


def batch_train(model, x1, x2, y):

    batch = tokens_per_sentence
    print(x1.shape)
    for i in range(x1.shape[1]):
        xx1 = x1[:,i]
        xx2 = x2[:,i]
        yy = y[:,i]

        xx1 = np.expand_dims(xx1, 0)
        xx2 = np.expand_dims(xx2, 0)
        yy = np.expand_dims(yy, 0)

        xx1 = np.expand_dims(xx1, 0)
        xx2 = np.expand_dims(xx2, 0)
        yy = np.expand_dims(yy, 0)

        #print(xx1.shape)
        model.train_on_batch([xx1,xx2],yy)
        if i % batch == 0:
            print(model.evaluate([xx1,xx2], yy), i, end=' ')
    #x = y = x_test

    #print (y.shape)




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
    print ('stage: arrays prep for test')
    x1, x2, y = vector_input_three(text_to, text_to, text_to)
    model , _, _ = embedding_model_lstm()
    batch_train(model, x1, x2, y)

if True:
    print ('stage: save lstm model')
    if filename == None:
        filename = hparams['save_dir'] + hparams['base_filename']+'-'+base_file_num +'.h5'
    model.save(filename)

if True:
    print('stage: try predict')
    c = open_sentences(text_to)
    line = c[0]
    predict_word(line)
    print('----------------')
    predict_word('sol what is up ? eol')



if True:
    print ('\n',len(word2vec_book.wv.vocab))

    vec = word2vec_book.wv['sol']
    print ( word2vec_book.wv.most_similar(positive=[vec], topn=5))
    print ( word2vec_book.wv.most_similar(positive=['man'], topn=5))
    print ('k', word2vec_book.wv.most_similar(positive=['k'], topn=5))