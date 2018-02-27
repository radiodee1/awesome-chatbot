#!/usr/bin/python3

import numpy as np
from settings import hparams
from keras.preprocessing import text, sequence
from keras.models import Sequential , Model
from keras.layers import Embedding, Input, LSTM, Bidirectional, TimeDistributed, Flatten, dot
from keras.layers import Activation, RepeatVector, Permute, Merge, Dense #, TimeDistributedMerge
from keras.layers import Concatenate, Add, Multiply
from keras.models import load_model
from keras import optimizers
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
    #x = np.swapaxes(x, 1, 2)

    y = np.swapaxes(y, 0, 2)
    #y = np.swapaxes(y, 1, 2)


    z = np.swapaxes(z, 0, 2)
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

    x_shape = (units,tokens_per_sentence)

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

    ### encoder for training ###
    lstm_a = LSTM(units=tokens_per_sentence,
                                return_state=True,
                                #return_sequences=True
                  )

    recurrent_a, lstm_a_h, lstm_a_c = lstm_a(valid_word_a)

    lstm_a_states = [lstm_a_h , lstm_a_c]

    ### decoder for training ###

    lstm_b = LSTM(units=tokens_per_sentence ,return_state=True #,
                                #input_shape=(None,units),
                                #return_sequences=True
                  )

    recurrent_b, _, _ = lstm_b(valid_word_b, initial_state=lstm_a_states)

    dense_b = Dense(tokens_per_sentence, activation='softmax')

    decoder_b = dense_b(recurrent_b)
    model = Model([valid_word_a,valid_word_b], decoder_b)

    ### encoder for inference ###
    model_encoder = Model(valid_word_a, lstm_a_states)

    ### decoder for inference ###

    input_h = Input(shape=(tokens_per_sentence,))
    input_c = Input(shape=(tokens_per_sentence,))

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

def embedding_model_lstm_alternate():

    #print (batch_size, tokens_per_sentence)

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

    time_dist_a = TimeDistributed(Dense(tokens_per_sentence))(recurrent_a2)


    #print (K.shape(recurrent_a2), 'not time dist')

    k_model = Model(inputs=[valid_word,valid_word_y], outputs=[time_dist_a])

    k_model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])


    return k_model


def train_embedding_model_api(model, x, y, predict=False, epochs=1, qnum=-1):
    z = x.shape[2] // batch_size
    num = 0
    for e in range(epochs):
        #print ('----')
        for i in range(z):
            xx , yy, yymod = get_batch_three(i, x, y, y)
            xx, yy , yymod = swap_axes(xx, yy, yymod)
            #model.train_on_batch(xx,yy)
            if not predict:
                #print (xx.shape, yy.shape)
                model.train_on_batch([xx,yy],yymod)
                #model.fit([xx,yy],yymod)
                #model.evaluate([xx,yy],yy)
            else:
                ypredict = model.predict([xx,yy], batch_size=batch_size)
                #print (ypredict.shape)
                for ii in ypredict:
                    #num += 1
                    if qnum != -1 and num > qnum: return
                    #print (ii,'<', ii.shape)

                    for j in range(tokens_per_sentence):
                        #print (j,'<<<<',i[:,j].shape)
                        z = word2vec_book.wv.most_similar(positive=[ii[:,j]],topn=1)
                        print (z[0][0], end=' ')
                    num += 1
        print('---- epoch ' + str(e + 1) + ' ----')

def inference_embedding_model_api(model, x, y, input='', show_similarity=False):
    z = None
    num = 1 #0 # skip zero?
    xx = None
    if len(input) != 0:
        print(input)
        jj = np.zeros((1,units,tokens_per_sentence))
        ii = input.lower().split()
        for k in range(len(ii)):
            w2v = word2vec_book.wv[ii[k]]
            jj[0][:,k] = w2v
            xx = yy = jj
        pass
    else:
        xx, yy = get_batch(0, x, y, batch_size=tokens_per_sentence)
        #print (xx.shape)
        for k in range(xx.shape[0]):

            print (word2vec_book.wv.most_similar(positive=[xx[k,:,0]], topn=1)[0][0], end=' ')


    print('\n--------------')
    if True:
        xx = np.expand_dims(xx[:,:,0], axis=0)
        xx = np.swapaxes(xx, 2,1)
        #print(xx.shape)
        if xx.shape[2] != tokens_per_sentence:
            temp = np.zeros((1, units, tokens_per_sentence))
            for t in range(tokens_per_sentence):
                temp[:,:,t] = xx[0,:,0]
            #print(temp)
            xx = temp
    else:
        xx = np.expand_dims(xx[0,:, :], axis=0)
        #print(xx.shape)
    single_word_y = np.zeros((1,units, tokens_per_sentence))
    while z == None or  (z != hparams['eol'] and num < tokens_per_sentence):

        if False:
            single_word_y[0,:,num] = yy[0,:,num] ## note: dont make this to yy[0,:,0]
        else:
            #print(single_word_y.shape,"---")
            single_word_y[0,:,0] = yy[0,:,0] ## note: dont make this to yy[0,:,0]

        z = model.predict([xx, single_word_y],batch_size=1)
        #print(z.shape)
        z = z[0]
        yy = np.expand_dims(z, axis=0)
        #z = word2vec_book.wv.most_similar(positive=[z[:,0]])
        if True:
            z = word2vec_book.wv.most_similar(positive=[z[:,0]])
            #print(z)
        elif z[0][0] in word2vec_book.wv.vocab:
            result = word2vec_book.wv[z[0][0]]
            yy = np.expand_dims(result,axis=0)
            yy = np.expand_dims(yy,axis=2)
            #print(yy.shape,'<<<')

        if show_similarity: print(z[0])

        z = z[0][0]
        print (z, end=' ')
        num +=1
    print()
    pass

def inference_w_a_g(model, x, y, n=0, count_printout=False):
    xx, yy = get_batch(n, x,y)
    xx, yy, _ = swap_axes(xx,yy,yy)
    z = xx[0,:,:]
    #print(xx.shape, z.shape, z.shape[1])
    i = 0
    h = 0
    found = False
    switch = False
    out = np.zeros((1,units))
    while i < tokens_per_sentence * 2 and found == False:
        single_word = np.zeros((1,units,tokens_per_sentence))
        if (not switch) and i < tokens_per_sentence:
            vec_in = xx[0,:,i]
            word_in = word2vec_book.most_similar(positive=[vec_in], topn=1)[0][0]
            if count_printout: print(word_in,'= word in',i)
            single_word[0,:,0] = xx[0,:,i]
        else:
            single_word[0,:,0] = out
            word_end = word2vec_book.most_similar(positive=[out], topn=1)[0][0]
            if word_end == hparams['eol'] and i > h + 2: found = True
            if count_printout: print(word_end,'= re-used',i)
        xx_start = np.zeros((1,units,tokens_per_sentence))
        xx_start[0] = z
        out = model.predict([xx_start,single_word], batch_size=1)
        out = out[0,:,0]
        vec = word2vec_book.most_similar(positive=[out], topn=1)
        #print(vec)
        word_out = vec[0][0]
        if count_printout: print(word_out,'= word out',i)
        else: print(word_out, end=', ')
        if word_in == hparams['eol']:
            switch = True
            word_in = ''
            h = i
            #print('throw fit.')
        i += 1
    print()
    pass

if True:
    print ('stage: arrays prep for train')
    #x, y = word_and_vector_size_arrays(train_fr, train_to)
    print ('stage: arrays prep for test')
    x_test, y_test = word_and_vector_size_arrays(text_fr, text_to, double_y=False, double_sentence_y=False)
    x = x_test
    y = y_test
    #x = y = x_test

    #print (y.shape)

if True:
    model, model_b, model_c = embedding_model_lstm()
    filename = hparams['save_dir'] + hparams['base_filename'] + '-' + base_file_num + '.h5'
else:
    model = embedding_model_lstm_alternate()
    filename = hparams['save_dir'] + hparams['base_filename'] + '-concat-' + base_file_num + '.h5'
print(filename)

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
    train_embedding_model_api(model, x, y, epochs=20)

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
    inference_embedding_model_api(model,x_test,y_test, show_similarity=False)

if True:
    print()
    inference_embedding_model_api(model,x_test,y_test,input='sol so who are you eol', show_similarity=False)

if True:
    print()
    inference_w_a_g(model, x_test, y_test, n=0, count_printout=False)

if False:
    print ('\n',len(word2vec_book.wv.vocab))

    vec = word2vec_book.wv['sol']
    print ( word2vec_book.wv.most_similar(positive=[vec], topn=5))
    print ( word2vec_book.wv.most_similar(positive=['man'], topn=5))
    print ('k', word2vec_book.wv.most_similar(positive=['k'], topn=5))